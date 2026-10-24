//! GStreamer encoder stages.
//!
//! M1 uses the hybrid launch-string path: an `appsrc` takes the NVMM
//! `GstBuffer` (the very same buffer capture handed out), the encoder
//! converts/encodes it, and a byte-stream `appsink` returns packets.
//! The CPU H.264 stage is the Orin Nano product path: one `nvvidconv`
//! copy out of NVMM to I420 immediately before `x264enc`, and nothing
//! else in the pipeline touches pixels on the CPU.
//!
//! Element presence is checked before a launch string is built
//! (finding N16): a missing `x264enc` must name the package, not fail
//! later inside `gst_parse_launch`.

use crate::launch;
use crate::probe::EncoderKind;
use edge_core::{Codec, EncodedPacket, Encoder, EncoderConfig, Error, HwFrame, Result};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The GStreamer element and distro/Yocto package an encoder needs.
fn required_elements(
    kind: EncoderKind,
    codec: Codec,
) -> (&'static str, &'static str, &'static str) {
    match (kind, codec) {
        (EncoderKind::Cpu, Codec::H264) => {
            ("x264enc", "h264parse", "gstreamer1.0-plugins-ugly-x264")
        }
        (EncoderKind::Cpu, Codec::H265) => {
            ("x265enc", "h265parse", "gstreamer1.0-plugins-ugly-x265")
        }
        (EncoderKind::Nvenc, Codec::H264) => {
            ("nvv4l2h264enc", "h264parse", "the L4T multimedia plugins")
        }
        (EncoderKind::Nvenc, Codec::H265) => {
            ("nvv4l2h265enc", "h265parse", "the L4T multimedia plugins")
        }
    }
}

/// Fail before building a launch string when an element is missing.
/// The `find` closure keeps this testable without installing plugins.
pub fn check_encoder_elements(
    kind: EncoderKind,
    codec: Codec,
    find: impl Fn(&str) -> bool,
) -> Result<()> {
    let (encoder, parser, package) = required_elements(kind, codec);
    for element in ["appsrc", "appsink", parser, encoder] {
        if !find(element) {
            let hint = if element == encoder {
                format!("; install {package}")
            } else {
                String::new()
            };
            return Err(Error::Unsupported(format!(
                "encoder `{encoder}` needs GStreamer element `{element}`{hint}"
            )));
        }
    }
    Ok(())
}

fn element_available(name: &str) -> bool {
    // ElementFactory::find needs an initialised GStreamer; a failed
    // init means no elements, which is the honest answer.
    let _ = gst::init();
    gst::ElementFactory::find(name).is_some()
}

/// True when the host/image can build the CPU H.264 launch string.
pub fn software_h264_available() -> bool {
    element_available("x264enc") && element_available("h264parse")
}

pub struct GstEncoder {
    kind: EncoderKind,
    cfg: EncoderConfig,
    width: u32,
    height: u32,
    /// Optional override for tests: a system-memory launch string
    /// instead of the NVMM device string.
    launch_override: Option<String>,
    pipeline: Option<gst::Pipeline>,
    appsrc: Option<gst_app::AppSrc>,
    appsink: Option<gst_app::AppSink>,
}

impl GstEncoder {
    pub fn new(kind: EncoderKind, cfg: &EncoderConfig, width: u32, height: u32) -> Result<Self> {
        check_encoder_elements(kind, cfg.codec, element_available)?;
        Ok(Self {
            kind,
            cfg: cfg.clone(),
            width,
            height,
            launch_override: None,
            pipeline: None,
            appsrc: None,
            appsink: None,
        })
    }

    /// Test hook: build a different (system-memory) pipeline. The
    /// production launch string needs `nvvidconv`/NVMM, which a dev
    /// PC does not have.
    pub fn new_with_launch(
        kind: EncoderKind,
        cfg: &EncoderConfig,
        width: u32,
        height: u32,
        launch: impl Into<String>,
    ) -> Result<Self> {
        check_encoder_elements(kind, cfg.codec, element_available)?;
        Ok(Self {
            kind,
            cfg: cfg.clone(),
            width,
            height,
            launch_override: Some(launch.into()),
            pipeline: None,
            appsrc: None,
            appsink: None,
        })
    }

    /// Parse and start the pipeline on first use. Keeping this lazy
    /// lets the builder validate a profile on a dev PC that lacks
    /// `nvvidconv`; the first real frame then reports the parse error.
    fn ensure_started(&mut self) -> Result<()> {
        if self.pipeline.is_some() {
            return Ok(());
        }
        let launch = self.launch_override.clone().unwrap_or_else(|| {
            launch::encode_launch(self.kind, &self.cfg, self.width, self.height)
        });
        let pipeline = gst::parse::launch(&launch)
            .map_err(|e| Error::Platform(format!("parsing encoder launch string: {e}")))?
            .downcast::<gst::Pipeline>()
            .map_err(|_| {
                Error::Platform("encoder launch string did not build a pipeline".into())
            })?;
        let appsrc = pipeline
            .by_name("framesrc")
            .and_then(|e| e.downcast::<gst_app::AppSrc>().ok())
            .ok_or_else(|| {
                Error::Platform("encoder launch string has no appsrc `framesrc`".into())
            })?;
        let appsink = pipeline
            .by_name("enc")
            .and_then(|e| e.downcast::<gst_app::AppSink>().ok())
            .ok_or_else(|| Error::Platform("encoder launch string has no appsink `enc`".into()))?;
        // Defence in depth: even a hand-written launch string must
        // never make push_buffer block forever (E1). A full queue is a
        // surfaced FlowError, not a hung encoder thread.
        appsrc.set_property("block", false);
        appsrc.set_property("max-buffers", 2u64);
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| Error::Platform(format!("starting encoder pipeline: {e}")))?;
        self.pipeline = Some(pipeline);
        self.appsrc = Some(appsrc);
        self.appsink = Some(appsink);
        Ok(())
    }

    fn emit_sample(
        &mut self,
        sample: &gst::Sample,
        out: &mut dyn FnMut(EncodedPacket),
    ) -> Result<()> {
        let buffer = sample
            .buffer_owned()
            .ok_or_else(|| Error::Platform("encoder sample without a buffer".into()))?;
        let map = buffer
            .map_readable()
            .map_err(|e| Error::Platform(format!("mapping encoded buffer: {e}")))?;
        let data: Arc<[u8]> = Arc::from(map.as_slice());
        let keyframe = !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT);
        let pts = buffer
            .pts()
            .map_or(Duration::ZERO, |t| Duration::from_nanos(t.nseconds()));
        out(EncodedPacket::new(pts, keyframe, data));
        Ok(())
    }

    /// Surface a pipeline error from the bus. Without this, a caps
    /// negotiation failure only shows up as a stalled appsrc.
    fn poll_bus(&self) -> Result<()> {
        let Some(pipeline) = &self.pipeline else {
            return Ok(());
        };
        let Some(bus) = pipeline.bus() else {
            return Ok(());
        };
        while let Some(msg) =
            bus.timed_pop_filtered(gst::ClockTime::ZERO, &[gst::MessageType::Error])
        {
            if let gst::MessageView::Error(e) = msg.view() {
                return Err(Error::Platform(format!(
                    "encoder pipeline error from {:?}: {} ({:?})",
                    msg.src().map(|s| s.path_string()),
                    e.error(),
                    e.debug()
                )));
            }
        }
        Ok(())
    }

    fn drain(&mut self, out: &mut dyn FnMut(EncodedPacket), blocking: bool) -> Result<()> {
        self.poll_bus()?;
        let timeout = if blocking {
            gst::ClockTime::from_mseconds(200)
        } else {
            gst::ClockTime::ZERO
        };
        let Some(appsink) = self.appsink.clone() else {
            return Ok(());
        };
        while let Some(sample) = appsink.try_pull_sample(timeout) {
            self.emit_sample(&sample, out)?;
        }
        Ok(())
    }
}

impl Encoder for GstEncoder {
    fn encode(&mut self, frame: HwFrame, out: &mut dyn FnMut(EncodedPacket)) -> Result<()> {
        self.ensure_started()?;
        let buffer = {
            let gst_buffer = frame
                .buf
                .as_any()
                .downcast_ref::<crate::gst_source::GstFrameBuffer>()
                .ok_or_else(|| Error::Platform("gst encoder got a non-GStreamer buffer".into()))?;
            gst_buffer.gst_buffer().cloned().ok_or_else(|| {
                Error::Platform(
                    "gst encoder needs a dmabuf-backed GstBuffer; the PC copy \
                     fallback cannot feed an encoder"
                        .into(),
                )
            })?
        };
        let appsrc = self
            .appsrc
            .as_ref()
            .ok_or_else(|| Error::Platform("encoder appsrc not started".into()))?;
        appsrc
            .push_buffer(buffer)
            .map_err(|e| Error::Platform(format!("pushing frame into encoder: {e}")))?;
        // Non-blocking drain: x264 emits from its own thread, so a
        // packet may arrive on a later call. `finish` flushes.
        self.drain(out, false)
    }

    fn poll(&mut self, out: &mut dyn FnMut(EncodedPacket)) -> Result<()> {
        if self.pipeline.is_none() {
            return Ok(());
        }
        self.drain(out, false)
    }

    fn finish(&mut self, out: &mut dyn FnMut(EncodedPacket)) -> Result<()> {
        if self.pipeline.is_none() {
            return Ok(());
        }
        if let Some(appsrc) = &self.appsrc {
            appsrc
                .end_of_stream()
                .map_err(|e| Error::Platform(format!("ending encoder stream: {e}")))?;
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            self.drain(out, true)?;
            self.poll_bus()?;
            if self.appsink.as_ref().is_some_and(|s| s.is_eos()) {
                break;
            }
        }
        self.drain(out, false)?;
        // Leave the encoder reusable: the runner hands it back so the
        // pipeline can run again. An ended appsrc cannot be pushed to.
        if let Some(pipeline) = self.pipeline.take() {
            let _ = pipeline.set_state(gst::State::Null);
        }
        self.appsrc = None;
        self.appsink = None;
        Ok(())
    }
}

impl Drop for GstEncoder {
    fn drop(&mut self) {
        if let Some(pipeline) = &self.pipeline {
            let _ = pipeline.set_state(gst::State::Null);
        }
    }
}
