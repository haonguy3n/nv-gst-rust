//! Pipeline builder and threaded runner.
//!
//! The builder validates the config against `PlatformCaps` and
//! refuses memory-domain mismatches at startup — it never inserts a
//! converter silently. The runner uses threads, not async: a capture
//! thread feeds a bounded drop-oldest channel, the processing thread
//! runs pre → infer → post → overlay → encode → sink per frame.
//! Only `HwFrame` handles move between stages, never pixel bytes.

#![forbid(unsafe_code)]

pub mod channel;
pub mod metrics;

pub use metrics::PipelineMetrics;

use edge_core::{
    Config, Detections, Encoder, Error, FramePoll, FrameSource, GpuStream, HwFrame,
    InferenceEngine, Overlay, PixelFormat, Platform, PostProcessor, PreConfig, Preprocessor,
    Result, Sink,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Pipeline {
    input_spec: Option<edge_core::TensorSpec>,
    source: Box<dyn FrameSource>,
    preprocessor: Box<dyn Preprocessor>,
    engine: Box<dyn InferenceEngine>,
    postprocessor: Box<dyn PostProcessor>,
    overlay: Box<dyn Overlay>,
    encoder: Option<Box<dyn Encoder>>,
    sink: Box<dyn Sink>,
    config: Config,
    metadata_only: bool,
}

impl Pipeline {
    pub fn build(
        platform: &dyn Platform,
        config: Config,
        postprocessor: Box<dyn PostProcessor>,
        sink: Box<dyn Sink>,
    ) -> Result<Self> {
        config.validate()?;
        let caps = platform.caps();

        if config.source.width > caps.max_width || config.source.height > caps.max_height {
            return Err(Error::invalid(
                "source.width/height",
                format!(
                    "exceeds {} maximum {}x{}",
                    platform.name(),
                    caps.max_width,
                    caps.max_height
                ),
            ));
        }
        if !caps.precisions.is_empty() && !caps.precisions.contains(&config.model.precision) {
            return Err(Error::invalid(
                "model.precision",
                format!(
                    "{} not supported by platform `{}`",
                    config.model.precision,
                    platform.name()
                ),
            ));
        }
        if !caps.devices.is_empty() && !caps.devices.contains(&config.model.device) {
            return Err(Error::invalid(
                "model.device",
                format!(
                    "`{}` not available on platform `{}` (has: {})",
                    config.model.device,
                    platform.name(),
                    caps.devices.join(", ")
                ),
            ));
        }
        if !caps.formats.contains(&PixelFormat::Nv12) {
            return Err(Error::DomainMismatch {
                stage: "source",
                expected: "NV12".into(),
                got: format!("{:?}", caps.formats),
            });
        }

        let source = platform.source(&config.source)?;
        if source.memory() != caps.frame_memory {
            return Err(Error::DomainMismatch {
                stage: "source",
                expected: caps.frame_memory.to_string(),
                got: source.memory().to_string(),
            });
        }

        let pre_cfg = PreConfig {
            src_width: config.source.width,
            src_height: config.source.height,
            dst_width: config.model.input_width,
            dst_height: config.model.input_height,
            layout: source.layout(),
            color: config.model.input_color,
            colorspace: config.model.input_colorspace,
            normalize: config.model.input_normalize,
            pad_value: config.model.pad_value,
        };
        let preprocessor = platform.preprocessor(&pre_cfg)?;
        let mut engine = platform.engine(&config.model)?;
        let input_spec = engine.input_bindings().first().map(|t| t.spec.clone());
        let overlay = platform.overlay()?;

        let encoder = if config.encoder.enabled {
            if config.encoder.hardware {
                if !caps.has_hw_encoder {
                    return Err(Error::invalid(
                        "encoder.enabled",
                        format!(
                            "platform `{}` has no hardware encoder; \
                             set stream.metadata_only = true, or opt in to CPU encode \
                             with zero_copy = \"relaxed\" and encoder.hardware = false",
                            platform.name()
                        ),
                    ));
                }
                if !caps.codecs.contains(&config.encoder.codec) {
                    return Err(Error::invalid(
                        "encoder.codec",
                        format!("not supported by platform `{}`", platform.name()),
                    ));
                }
            } else {
                // The explicit opt-in (design revision decision 3):
                // relaxed mode only, and the platform must actually
                // offer a CPU path. Config validation already
                // rejects strict + hardware=false; re-check here for
                // programmatically built configs.
                if config.zero_copy == edge_core::ZeroCopyMode::Strict {
                    return Err(Error::invalid(
                        "encoder.hardware",
                        "zero_copy = \"strict\" refuses a CPU encoder; \
                         set zero_copy = \"relaxed\" to opt in",
                    ));
                }
                if !caps.cpu_encode {
                    return Err(Error::invalid(
                        "encoder.enabled",
                        format!(
                            "platform `{}` offers no CPU encoder either; \
                             set stream.metadata_only = true",
                            platform.name()
                        ),
                    ));
                }
            }
            match platform.encoder(&config.encoder)? {
                Some(enc) => Some(enc),
                None => {
                    return Err(Error::invalid(
                        "encoder.enabled",
                        format!(
                            "platform `{}` returned no encoder; \
                             set stream.metadata_only = true",
                            platform.name()
                        ),
                    ))
                }
            }
        } else {
            None
        };
        let metadata_only = encoder.is_none();

        Ok(Self {
            input_spec,
            source,
            preprocessor,
            engine,
            postprocessor,
            overlay,
            encoder,
            sink,
            config,
            metadata_only,
        })
    }

    pub fn metadata_only(&self) -> bool {
        self.metadata_only
    }

    /// The per-frame buffer budget, printed at startup so a hidden
    /// extra buffer is visible in the log, not just in a trace.
    pub fn buffer_budget(&self) -> String {
        let s = &self.config.source;
        let m = &self.config.model;
        let capture = PixelFormat::Nv12.frame_bytes(s.width, s.height);
        let (dtype, input) = match &self.input_spec {
            Some(spec) => (format!("{:?}", spec.dtype), spec.byte_len()),
            None => (
                "FP16 (assumed)".to_string(),
                (m.input_width as usize) * (m.input_height as usize) * 3 * 2,
            ),
        };
        format!(
            "buffer budget:\n  capture NV12 {}x{} {}: {} bytes x pool {}\n  \
             input tensor {} 3x{}x{}: {} bytes (engine binding, written once)\n  \
             output tensors: host-visible, read in place\n  \
             mode: {}",
            s.width,
            s.height,
            self.source.layout(),
            capture,
            s.pool_size,
            dtype,
            m.input_width,
            m.input_height,
            input,
            if self.metadata_only {
                "metadata only (no HW encoder)"
            } else {
                "video + metadata"
            }
        )
    }

    /// Run until the source ends or `max_frames` frames are processed.
    pub fn run(&mut self, max_frames: Option<u64>) -> Result<PipelineMetrics> {
        let mut metrics = PipelineMetrics::default();
        let stream = GpuStream::new(0);
        let limit = max_frames.or(self.config.source.frames);

        let (tx, rx, dropped) = channel::channel::<HwFrame>(self.config.pipeline.queue_depth);
        // Shared stop flag: the only way the capture thread learns
        // the consumer is gone (a stage error must never deadlock
        // against an unbounded source — REVIEW P0-1).
        let stop = Arc::new(AtomicBool::new(false));
        let capture_stop = Arc::clone(&stop);

        // Temporarily move the source into the capture thread.
        let mut source = std::mem::replace(&mut self.source, NoopSource::boxed());
        let capture = std::thread::spawn(move || -> Result<(Box<dyn FrameSource>, u64, u64)> {
            let mut captured = 0u64;
            let mut timeouts = 0u64;
            source.start()?;
            loop {
                if capture_stop.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(limit) = limit {
                    if captured >= limit {
                        break;
                    }
                }
                match source.next_frame(Duration::from_millis(100))? {
                    FramePoll::Frame(frame) => {
                        captured += 1;
                        // Drop-oldest: an evicted frame returns its
                        // buffer to the capture pool here.
                        drop(tx.send(frame));
                    }
                    // A momentary shortage is not the end of
                    // anything: count it and keep capturing.
                    FramePoll::Timeout => timeouts += 1,
                    FramePoll::EndOfStream => break,
                }
            }
            source.stop()?;
            Ok((source, captured, timeouts))
        });

        let mut detections = Detections::with_capacity(64);
        let mut last_err: Option<Error> = None;

        loop {
            let Some(frame) = rx.recv_timeout(Duration::from_millis(200)) else {
                if rx.is_closed_and_empty() || capture.is_finished() {
                    break;
                }
                continue;
            };
            if let Err(e) = self.process_one(frame, &stream, &mut detections, &mut metrics) {
                last_err = Some(e);
                break;
            }
        }

        // Tell capture to stop and drop the receiver *before*
        // joining, so an unbounded source cannot block shutdown.
        stop.store(true, Ordering::Relaxed);
        drop(rx);
        let (source, captured, timeouts) = capture
            .join()
            .map_err(|_| Error::Platform("capture thread panicked".into()))??;
        self.source = source;
        metrics.frames_captured = captured;
        metrics.capture_timeouts = timeouts;
        metrics.frames_dropped += dropped.load(Ordering::Relaxed);

        match last_err {
            Some(e) => Err(e),
            None => Ok(metrics),
        }
    }

    fn process_one(
        &mut self,
        frame: HwFrame,
        stream: &GpuStream,
        detections: &mut Detections,
        metrics: &mut PipelineMetrics,
    ) -> Result<()> {
        let t = Instant::now();
        let engine = &mut *self.engine;
        let input = engine
            .input_bindings()
            .first_mut()
            .ok_or_else(|| Error::Platform("engine exposes no input binding".into()))?;
        self.preprocessor.run(&frame, input, stream)?;
        metrics.pre_total += t.elapsed();

        let t = Instant::now();
        self.engine.infer(stream)?;
        metrics.infer_total += t.elapsed();

        let t = Instant::now();
        self.postprocessor
            .decode(self.engine.output_bindings(), frame.meta(), detections)?;
        metrics.post_total += t.elapsed();
        metrics.detections += detections.len() as u64;

        let t = Instant::now();
        self.overlay.draw(&frame, detections, stream)?;
        metrics.overlay_total += t.elapsed();

        let t = Instant::now();
        match &mut self.encoder {
            Some(encoder) => {
                let sink = &mut *self.sink;
                let mut sink_err: Option<Error> = None;
                // The encoder holds the HwFrame until the hardware
                // releases it; enforce the same-fd contract in
                // release builds too, not just debug ones.
                if frame.meta().dmabuf_fd != frame.buf.dmabuf_fd() {
                    return Err(Error::Platform(format!(
                        "encoder fd contract violated: meta says fd {}, buffer is fd {}",
                        frame.meta().dmabuf_fd,
                        frame.buf.dmabuf_fd()
                    )));
                }
                encoder.encode(frame, &mut |pkt| {
                    if let Err(e) = sink.push_video(pkt) {
                        sink_err = Some(e);
                    }
                })?;
                if let Some(e) = sink_err {
                    return Err(e);
                }
            }
            None => drop(frame), // returns the buffer to the capture pool
        }
        metrics.encode_total += t.elapsed();

        self.sink.push_meta(detections)?;
        metrics.frames_processed += 1;
        Ok(())
    }
}

/// Placeholder swapped in while the real source lives in the capture thread.
struct NoopSource;

impl NoopSource {
    fn boxed() -> Box<dyn FrameSource> {
        Box::new(Self)
    }
}

impl FrameSource for NoopSource {
    fn start(&mut self) -> Result<()> {
        Ok(())
    }
    fn next_frame(&mut self, _timeout: Duration) -> Result<FramePoll> {
        Ok(FramePoll::EndOfStream)
    }
    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
    fn memory(&self) -> edge_core::FrameMemory {
        edge_core::FrameMemory::DmaBuf
    }
}
