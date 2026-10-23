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
    Config, Detections, Encoder, Error, FrameSource, GpuStream, HwFrame, InferenceEngine, Layout,
    Overlay, PixelFormat, Platform, PostProcessor, PreConfig, Preprocessor, Result, Sink,
};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

pub struct Pipeline {
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
            layout: Layout::PitchLinear,
        };
        let preprocessor = platform.preprocessor(&pre_cfg)?;
        let engine = platform.engine(&config.model)?;
        let overlay = platform.overlay()?;

        let encoder = if config.encoder.enabled {
            if !caps.has_hw_encoder {
                return Err(Error::invalid(
                    "encoder.enabled",
                    format!(
                        "platform `{}` has no hardware encoder; \
                         set stream.metadata_only = true (frames never go to a CPU encoder)",
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
        let input = (m.input_width as usize) * (m.input_height as usize) * 3 * 2; // FP16
        format!(
            "buffer budget:\n  capture NV12 {}x{} pitch-linear: {} bytes x pool {}\n  \
             input tensor FP16 3x{}x{}: {} bytes (engine binding, written once)\n  \
             output tensors: host-visible, read in place\n  \
             mode: {}",
            s.width,
            s.height,
            capture,
            s.pool_size,
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

        // Temporarily move the source into the capture thread.
        let mut source = std::mem::replace(&mut self.source, NoopSource::boxed());
        let capture = std::thread::spawn(move || -> Result<(Box<dyn FrameSource>, u64)> {
            let mut captured = 0u64;
            source.start()?;
            loop {
                if let Some(limit) = limit {
                    if captured >= limit {
                        break;
                    }
                }
                match source.next_frame(Duration::from_millis(100))? {
                    Some(frame) => {
                        captured += 1;
                        // Drop-oldest: an evicted frame returns its
                        // buffer to the capture pool here.
                        drop(tx.send(frame));
                    }
                    None => break, // end of stream / exhausted source
                }
            }
            source.stop()?;
            Ok((source, captured))
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

        let (source, captured) = capture
            .join()
            .map_err(|_| Error::Platform("capture thread panicked".into()))??;
        self.source = source;
        metrics.frames_captured = captured;
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
            .decode(self.engine.output_bindings(), &frame.meta, detections)?;
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
                // releases it; assert the fd contract here.
                debug_assert_eq!(frame.meta.dmabuf_fd, frame.buf.dmabuf_fd());
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
    fn next_frame(&mut self, _timeout: Duration) -> Result<Option<HwFrame>> {
        Ok(None)
    }
    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
    fn memory(&self) -> edge_core::FrameMemory {
        edge_core::FrameMemory::DmaBuf
    }
}
