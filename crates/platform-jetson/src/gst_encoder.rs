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
    fps: u32,
    /// Optional override for tests: a system-memory launch string
    /// instead of the NVMM device string.
    launch_override: Option<String>,
    pipeline: Option<gst::Pipeline>,
    appsrc: Option<gst_app::AppSrc>,
    appsink: Option<gst_app::AppSink>,
    logged_first_buffer: bool,
    appsrc_dropped: u64,
    /// PTS (ns) of the first buffer pushed into the appsrc.
    first_in_pts: Option<u64>,
    /// The encoder's constant timestamp shift, learned from the first packet.
    pts_shift: Option<u64>,
}

/// Whole-second shift the encoder added to the timestamps: the first output
/// PTS minus the first input PTS, rounded to the nearest second.
///
/// `GstVideoEncoder` (x264enc calls `set_min_pts(1000 h)`) moves every output
/// PTS/DTS up by a constant so DTS stays positive; a measured Nano run saw
/// 3 600 000 s exactly. The shift is a whole number of seconds, anything
/// below that is pipeline latency or a leaked first frame, so rounding keeps
/// the mapping right even when the leaky appsrc dropped the first frames.
/// Encoders that do not shift (NVENC) give 0. The result is the same clock as
/// `FrameMeta.pts` and the detections.
pub fn encoder_pts_shift(first_in_ns: u64, first_out_ns: u64) -> u64 {
    const SEC: u64 = 1_000_000_000;
    let diff = first_out_ns.saturating_sub(first_in_ns);
    (diff + SEC / 2) / SEC * SEC
}

impl GstEncoder {
    pub fn new(kind: EncoderKind, cfg: &EncoderConfig, width: u32, height: u32) -> Result<Self> {
        check_encoder_elements(kind, cfg.codec, element_available)?;
        Ok(Self {
            kind,
            cfg: cfg.clone(),
            width,
            height,
            fps: 30,
            launch_override: None,
            pipeline: None,
            appsrc: None,
            appsink: None,
            logged_first_buffer: false,
            appsrc_dropped: 0,
            first_in_pts: None,
            pts_shift: None,
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
            fps: 30,
            launch_override: Some(launch.into()),
            pipeline: None,
            appsrc: None,
            appsink: None,
            logged_first_buffer: false,
            appsrc_dropped: 0,
            first_in_pts: None,
            pts_shift: None,
        })
    }

    /// Source frame rate for the appsrc caps (default 30). NVIDIA's
    /// `nvvidconv` refuses NVMM caps without a framerate.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = fps.max(1);
        self
    }

    /// Frames the leaky `appsrc` queue evicted because the encoder
    /// was behind (a proxy: counted when the queue was non-empty at push).
    pub fn appsrc_dropped(&self) -> u64 {
        self.appsrc_dropped
    }

    /// Parse and start the pipeline on first use. Keeping this lazy
    /// lets the builder validate a profile on a dev PC that lacks
    /// `nvvidconv`; the first real frame then reports the parse error.
    fn ensure_started(&mut self) -> Result<()> {
        if self.pipeline.is_some() {
            return Ok(());
        }
        let launch = self.launch_override.clone().unwrap_or_else(|| {
            launch::encode_launch(self.kind, &self.cfg, self.width, self.height, self.fps)
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
        // never block the encoder thread. The queue is one frame and
        // leaks the oldest; `block=false` alone would let it grow.
        appsrc.set_property("block", false);
        appsrc.set_property("max-buffers", 1u64);
        if appsrc.find_property("leaky-type").is_some() {
            appsrc.set_property_from_str("leaky-type", "downstream");
        }
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
        let raw = buffer.pts().map_or(0, |t| t.nseconds());
        if self.pts_shift.is_none() {
            let shift = self.first_in_pts.map_or(0, |i| encoder_pts_shift(i, raw));
            log::info!("encoder: output timestamps are shifted by {shift} ns; mapped back");
            self.pts_shift = Some(shift);
        }
        let pts = Duration::from_nanos(raw.saturating_sub(self.pts_shift.unwrap_or(0)));
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
        // `pop_filtered`, not `timed_pop_filtered(ZERO, ..)`: gstreamer-rs
        // 0.25 computes `total - elapsed` first, and a zero budget is
        // already spent by then, so it returned None without ever
        // looking at the bus.
        let mut examined = 0u32;
        while let Some(msg) = bus.pop_filtered(&[gst::MessageType::Error]) {
            examined += 1;
            if let gst::MessageView::Error(e) = msg.view() {
                return Err(Error::Platform(format!(
                    "encoder pipeline error from {:?}: {} ({:?})",
                    msg.src().map(|s| s.path_string()),
                    e.error(),
                    e.debug()
                )));
            }
        }
        gst::debug!(
            gst::CAT_DEFAULT,
            "encoder: poll_bus examined {examined} error messages"
        );
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
    fn dropped_frames(&self) -> u64 {
        self.appsrc_dropped
    }

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
        if !self.logged_first_buffer {
            self.logged_first_buffer = true;
            // E5 diagnostic for the first board run: a writable buffer
            // or a set DISCONT flag means the no-share NVMM buffer was
            // copied before it reached the encoder.
            gst::info!(
                gst::CAT_DEFAULT,
                "encoder: first buffer flags={:?} writable={}",
                buffer.flags(),
                buffer.is_writable()
            );
            log::info!(
                "encoder: first buffer flags={:?} writable={}",
                buffer.flags(),
                buffer.is_writable()
            );
        }
        // The appsrc holds its own ref from here on: release the frame's
        // now, so the encoder path pins one capture surface less (N11).
        drop(frame);
        let appsrc = self
            .appsrc
            .as_ref()
            .ok_or_else(|| Error::Platform("encoder appsrc not started".into()))?;
        // With max-buffers=1 + leaky downstream, a non-empty queue
        // means this push will drop the oldest frame. `current_level_buffers`
        // needs gstreamer-app's v1_20 feature; queued bytes is a
        // sufficient non-empty proxy here.
        if appsrc.current_level_bytes() > 0 {
            self.appsrc_dropped += 1;
        }
        if self.first_in_pts.is_none() {
            self.first_in_pts = buffer.pts().map(|t| t.nseconds());
        }
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
        self.first_in_pts = None;
        self.pts_shift = None;
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

#[cfg(test)]
mod tests {
    use super::encoder_pts_shift;

    #[test]
    fn shift_rounds_to_whole_seconds() {
        let thousand_hours = 3_600_000_000_000_000u64;
        let first_in = 454_552_445_000u64;
        assert_eq!(
            encoder_pts_shift(first_in, first_in + thousand_hours + 40_000_000),
            thousand_hours
        );
        assert_eq!(encoder_pts_shift(first_in, first_in + 66_000_000), 0);
        assert_eq!(encoder_pts_shift(5, 3), 0);
    }
}
