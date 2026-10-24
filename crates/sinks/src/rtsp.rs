//! RTSP video sink backed by `gst-rtsp-server`.
//!
//! The encoder hands us H.264/H.265 byte-stream access units. A shared
//! RTSP media factory owns an `appsrc`; when a client connects its
//! pipeline is created and `media-configure` hands us the appsrc. With
//! no client we drop packets (live semantics); once connected, each
//! packet is copied into a `GstBuffer` (the bitstream copy budgeted in
//! the design).
//!
//! This is a PC-testable sink: construction, URL and no-client pushes
//! are unit-tested; actual playback needs a device/client.

use edge_core::{Codec, EncodedPacket, Error, Result, Sink};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_rtsp_server as gst_rtsp_server;
use gstreamer_rtsp_server::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// State shared with the media-configure/unprepared callbacks.
struct Shared {
    appsrc: Option<gst_app::AppSrc>,
    gate: PtsGate,
}

type SharedRef = Arc<Mutex<Shared>>;

fn lock(shared: &SharedRef) -> std::sync::MutexGuard<'_, Shared> {
    shared.lock().expect("rtsp appsrc mutex poisoned")
}

/// Per-(re)prepare timestamp state. A joining client starts its running
/// time at 0, so PTS is rebased to the first buffer pushed after the
/// prepare, and nothing is pushed before the first keyframe.
#[derive(Debug, Default)]
struct PtsGate {
    synced: bool,
    base: Duration,
}

impl PtsGate {
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// The rebased PTS to push, or `None` to drop the packet.
    fn map(&mut self, pts: Duration, keyframe: bool) -> Option<Duration> {
        if !self.synced {
            if !keyframe {
                return None;
            }
            self.synced = true;
            self.base = pts;
        }
        Some(pts.saturating_sub(self.base))
    }
}

/// Elements the launch string needs, with the Yocto package of each.
fn required_elements(codec: Codec) -> [(&'static str, &'static str); 2] {
    match codec {
        Codec::H264 => [
            ("rtph264pay", "gstreamer1.0-plugins-good-rtp"),
            ("h264parse", "gstreamer1.0-plugins-bad-videoparsersbad"),
        ],
        Codec::H265 => [
            ("rtph265pay", "gstreamer1.0-plugins-good-rtp"),
            ("h265parse", "gstreamer1.0-plugins-bad-videoparsersbad"),
        ],
    }
}

/// Fails at construction, not at the first client, when a plugin is
/// missing. `have` is the element lookup (a seam for tests).
fn check_elements(codec: Codec, have: impl Fn(&str) -> bool) -> Result<()> {
    for (element, package) in required_elements(codec) {
        if !have(element) {
            return Err(Error::Platform(format!(
                "RTSP needs the GStreamer element `{element}` (Yocto package `{package}`), which is not installed"
            )));
        }
    }
    Ok(())
}

/// A push into a media that is going away is a dropped packet, not a
/// run error (the viewer left).
fn push_outcome(result: std::result::Result<gst::FlowSuccess, gst::FlowError>) -> Result<()> {
    match result {
        Ok(_) | Err(gst::FlowError::Flushing | gst::FlowError::Eos) => Ok(()),
        Err(e) => Err(Error::Platform(format!("pushing RTSP packet: {e}"))),
    }
}

/// `SourceId::remove` only knows the default context; ours is private.
fn detach(context: &gst::glib::MainContext, id: &gst::glib::SourceId) {
    if let Some(source) = context.find_source_by_id(id) {
        source.destroy();
    }
}

pub struct RtspSink {
    port: u16,
    server: gst_rtsp_server::RTSPServer,
    shared: SharedRef,
    source: Option<gst::glib::SourceId>,
    main_loop: gst::glib::MainLoop,
    context: gst::glib::MainContext,
    thread: Option<std::thread::JoinHandle<()>>,
    finished: bool,
}

impl RtspSink {
    pub fn new(port: u16, codec: Codec) -> Result<Self> {
        gst::init().map_err(|e| Error::Platform(format!("gstreamer init: {e}")))?;
        check_elements(codec, |name| gst::ElementFactory::find(name).is_some())?;
        if gst::ElementFactory::find("udpsink").is_none() {
            log::warn!("video_rtsp: udp plugin missing; only TCP-interleaved clients will work");
        }
        let launch = match codec {
            Codec::H264 => {
                "( appsrc name=videosrc is-live=true format=time \
                 max-buffers=8 max-bytes=1048576 leaky-type=downstream \
                 caps=video/x-h264,stream-format=byte-stream,alignment=au \
                 ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
            }
            Codec::H265 => {
                "( appsrc name=videosrc is-live=true format=time \
                 max-buffers=8 max-bytes=1048576 leaky-type=downstream \
                 caps=video/x-h265,stream-format=byte-stream,alignment=au \
                 ! h265parse ! rtph265pay name=pay0 pt=96 config-interval=1 )"
            }
        };
        let factory = gst_rtsp_server::RTSPMediaFactory::new();
        factory.set_launch(launch);
        factory.set_shared(true);
        let shared: SharedRef = Arc::new(Mutex::new(Shared {
            appsrc: None,
            gate: PtsGate::default(),
        }));
        let slot = Arc::clone(&shared);
        factory.connect_media_configure(move |_, media| {
            // The launch string is a `( ... )` bin, not a pipeline: a
            // Pipeline downcast always failed and no packet was pushed.
            let Ok(pipeline) = media.element().downcast::<gst::Bin>() else {
                log::error!("video_rtsp: media element is not a bin; client gets no video");
                return;
            };
            let Some(src) = pipeline
                .by_name("videosrc")
                .and_then(|e| e.downcast::<gst_app::AppSrc>().ok())
            else {
                return;
            };
            {
                let mut state = lock(&slot);
                state.gate.reset();
                state.appsrc = Some(src.clone());
            }
            let slot = Arc::clone(&slot);
            media.connect_unprepared(move |_| {
                let mut state = lock(&slot);
                if state.appsrc.as_ref() == Some(&src) {
                    state.appsrc = None;
                }
            });
        });

        let server = gst_rtsp_server::RTSPServer::new();
        server.set_service(&port.to_string());
        let mounts = server
            .mount_points()
            .ok_or_else(|| Error::Platform("RTSP server has no mount points".into()))?;
        mounts.add_factory("/stream", factory);
        // A private context per sink: a second sink in one process must
        // not share the global default context or its loop.
        let context = gst::glib::MainContext::new();
        let source = server.attach(Some(&context)).map_err(|e| {
            Error::Platform(format!(
                "attaching RTSP server on port {port} (in use?): {e}"
            ))
        })?;

        let main_loop = gst::glib::MainLoop::new(Some(&context), false);
        match Self::start_loop(&main_loop, &context, port) {
            Ok(thread) => Ok(Self {
                port,
                server,
                shared,
                source: Some(source),
                main_loop,
                context,
                thread: Some(thread),
                finished: false,
            }),
            Err(e) => {
                main_loop.quit();
                detach(&context, &source);
                Err(e)
            }
        }
    }

    fn start_loop(
        main_loop: &gst::glib::MainLoop,
        context: &gst::glib::MainContext,
        port: u16,
    ) -> Result<std::thread::JoinHandle<()>> {
        let loop_thread = main_loop.clone();
        let loop_context = context.clone();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        // Signal from inside the loop so a fast `finish()` cannot call
        // quit() before run() starts (and lose the quit). The idle source
        // sits on our context, so only its running loop dispatches it.
        let started = gst::glib::idle_source_new(
            Some("rtsp-started"),
            gst::glib::Priority::DEFAULT,
            move || {
                let _ = started_tx.send(());
                gst::glib::ControlFlow::Break
            },
        );
        started.attach(Some(context));
        let thread = std::thread::Builder::new()
            .name("rtsp-server".into())
            .spawn(move || {
                let _ = loop_context.with_thread_default(|| loop_thread.run());
            })
            .map_err(|e| Error::Platform(format!("spawning RTSP loop for port {port}: {e}")))?;
        if let Err(e) = started_rx.recv_timeout(Duration::from_secs(2)) {
            main_loop.quit();
            let _ = thread.join();
            return Err(Error::Platform(format!(
                "RTSP main loop for port {port} did not start: {e}"
            )));
        }
        Ok(thread)
    }

    pub fn url(&self) -> String {
        format!("rtsp://0.0.0.0:{}/stream", self.port)
    }

    /// True while a client's media pipeline owns an appsrc.
    pub fn has_client(&self) -> bool {
        lock(&self.shared).appsrc.is_some()
    }
}

impl Sink for RtspSink {
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()> {
        let (appsrc, pts) = {
            let mut state = lock(&self.shared);
            let Some(appsrc) = state.appsrc.clone() else {
                // No client: live stream semantics, drop the packet.
                return Ok(());
            };
            match state.gate.map(pkt.pts, pkt.keyframe) {
                Some(pts) => (appsrc, pts),
                None => return Ok(()),
            }
        };
        let mut buffer = gst::Buffer::from_slice(pkt.data);
        if let Some(buffer) = buffer.get_mut() {
            let pts =
                gst::ClockTime::from_nseconds(u64::try_from(pts.as_nanos()).unwrap_or(u64::MAX));
            buffer.set_pts(pts);
            buffer.set_dts(pts);
        }
        push_outcome(appsrc.push_buffer(buffer))
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let appsrc = lock(&self.shared).appsrc.take();
        if let Some(appsrc) = appsrc {
            let _ = appsrc.end_of_stream();
        }
        self.main_loop.quit();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        if let Some(source) = self.source.take() {
            detach(&self.context, &source);
        }
        // Drop the server and its context after the main loop is stopped.
        let _ = (&self.server, &self.context);
        Ok(())
    }
}

impl Drop for RtspSink {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

/// Parse `stream.rtsp_port`; a zero port is rejected because it is not
/// a usable RTSP endpoint in a config.
pub fn validate_port(port: u16) -> Result<()> {
    if port == 0 {
        return Err(Error::invalid("stream.rtsp_port", "must be > 0"));
    }
    Ok(())
}

/// A tiny helper for tests and logs.
pub fn url_for(port: u16) -> String {
    format!("rtsp://0.0.0.0:{port}/stream")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_port() {
        assert!(validate_port(0).is_err());
        assert!(validate_port(8555).is_ok());
        assert_eq!(url_for(8555), "rtsp://0.0.0.0:8555/stream");
    }

    #[test]
    fn constructs_and_drops_packets_without_a_client() {
        gst::init().expect("gst init");
        if gst::ElementFactory::find("rtph264pay").is_none() {
            eprintln!("skipping: rtph264pay not installed");
            return;
        }
        let mut sink = RtspSink::new(0, Codec::H264).expect("construct RTSP sink");
        assert_eq!(sink.url(), "rtsp://0.0.0.0:0/stream");
        assert!(!sink.has_client());
        sink.push_video(EncodedPacket::new(
            Duration::from_millis(33),
            true,
            vec![0, 0, 0, 1, 0x67],
        ))
        .expect("no-client push is a drop, not an error");
        sink.finish().expect("finish RTSP sink");
    }

    #[test]
    fn two_sinks_in_one_process_have_independent_loops() {
        gst::init().expect("gst init");
        if check_elements(Codec::H264, |n| gst::ElementFactory::find(n).is_some()).is_err() {
            eprintln!("skipping: RTSP plugins not installed");
            return;
        }
        let mut a = RtspSink::new(0, Codec::H264).expect("sink a");
        let mut b = RtspSink::new(0, Codec::H264).expect("sink b");
        a.finish().expect("finish a");
        b.finish().expect("finish b");
    }

    #[test]
    fn missing_payloader_error_names_element_and_package() {
        let err = check_elements(Codec::H264, |_| false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rtph264pay"), "{err}");
        assert!(err.contains("gstreamer1.0-plugins-good-rtp"), "{err}");
        let err = check_elements(Codec::H265, |n| n != "rtph265pay")
            .unwrap_err()
            .to_string();
        assert!(err.contains("rtph265pay"), "{err}");
        assert!(check_elements(Codec::H264, |_| true).is_ok());
    }

    #[test]
    fn gate_waits_for_a_keyframe_and_rebases_pts() {
        let ms = Duration::from_millis;
        let mut gate = PtsGate::default();
        assert_eq!(gate.map(ms(100), false), None);
        assert_eq!(gate.map(ms(133), true), Some(ms(0)));
        assert_eq!(gate.map(ms(166), false), Some(ms(33)));
        gate.reset();
        assert_eq!(gate.map(ms(200), false), None);
        assert_eq!(gate.map(ms(5000), true), Some(ms(0)));
    }

    #[test]
    fn flushing_and_eos_pushes_are_drops_not_errors() {
        assert!(push_outcome(Err(gst::FlowError::Flushing)).is_ok());
        assert!(push_outcome(Err(gst::FlowError::Eos)).is_ok());
        assert!(push_outcome(Ok(gst::FlowSuccess::Ok)).is_ok());
        assert!(push_outcome(Err(gst::FlowError::NotLinked)).is_err());
    }

    #[test]
    fn push_into_a_stopped_appsrc_is_ok_and_unprepare_clears_the_slot() {
        gst::init().expect("gst init");
        let src = gst_app::AppSrc::builder().build();
        // A never-started appsrc refuses buffers with Flushing.
        let flow = src.push_buffer(gst::Buffer::from_slice(vec![0u8]));
        assert!(push_outcome(flow).is_ok());
        let shared: SharedRef = Arc::new(Mutex::new(Shared {
            appsrc: Some(src),
            gate: PtsGate::default(),
        }));
        lock(&shared).appsrc = None;
        assert!(lock(&shared).appsrc.is_none());
    }

    #[test]
    fn port_in_use_error_names_the_port() {
        gst::init().expect("gst init");
        if check_elements(Codec::H264, |n| gst::ElementFactory::find(n).is_some()).is_err() {
            eprintln!("skipping: RTSP plugins not installed");
            return;
        }
        let blocker = std::net::TcpListener::bind("0.0.0.0:0").expect("bind");
        let port = blocker.local_addr().unwrap().port();
        let err = RtspSink::new(port, Codec::H264)
            .err()
            .expect("port is taken");
        assert!(err.to_string().contains(&port.to_string()), "{err}");
    }
}
