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

pub use metrics::{LiveMetrics, PipelineMetrics};

use edge_core::{
    Config, Detections, Encoder, EngineState, EngineWarmup, Error, FramePoll, FrameSource,
    GpuStream, HwFrame, InferenceEngine, ModelConfig, Overlay, PixelFormat, Platform,
    PostProcessor, PreConfig, Preprocessor, Result, Sink, TensorSpec,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Pipeline {
    input_spec: Option<edge_core::TensorSpec>,
    source: Box<dyn FrameSource>,
    preprocessor: Box<dyn Preprocessor>,
    engine: EngineSlot,
    postprocessor: Box<dyn PostProcessor>,
    overlay: Box<dyn Overlay>,
    encoder: Option<Box<dyn Encoder>>,
    sink: Box<dyn Sink>,
    config: Config,
    metadata_only: bool,
    /// External stop request (signals, supervisors, tests).
    stop: Arc<AtomicBool>,
    /// Counters observable while a run is in progress.
    live_metrics: Arc<LiveMetrics>,
    /// The camera's ordered GPU work queue, from the platform.
    stream: GpuStream,
}

/// The engine stage: ready, or still warming (plan build). While
/// warming, frames skip pre/infer/post and carry empty detections.
enum EngineSlot {
    Ready(Box<dyn InferenceEngine>),
    Warming(Box<dyn EngineWarmup>),
}

/// The engine's declared input must match the config the
/// preprocessor was built for, or the mismatch would surface
/// only at the first infer (or never, on a dynamic-shape model).
fn check_input_spec(spec: Option<&TensorSpec>, model: &ModelConfig) -> Result<()> {
    let Some(spec) = spec else { return Ok(()) };
    let expected_shape = vec![
        1,
        3,
        model.input_height as usize,
        model.input_width as usize,
    ];
    if !spec.shape.is_empty() && spec.shape != expected_shape {
        return Err(Error::DomainMismatch {
            stage: "engine",
            expected: format!("input shape {expected_shape:?}"),
            got: format!("{:?}", spec.shape),
        });
    }
    // `model.precision` is a builder choice for layer execution, not
    // the dtype of the I/O bindings: a TensorRT plan built with FP16
    // (or INT8) still exposes FP32 input/output tensors, which is what
    // the mapped-host design reads and writes. The engine's own spec is
    // authoritative; only reject a non-float input that no
    // preprocessor in this workspace can write.
    if !matches!(spec.dtype, edge_core::DType::F32 | edge_core::DType::F16) {
        return Err(Error::DomainMismatch {
            stage: "engine",
            expected: "float input binding (F32 or F16)".into(),
            got: format!("{:?}", spec.dtype),
        });
    }
    Ok(())
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
        if let Some(device) = &config.model.device {
            if !caps.devices.is_empty() && !caps.devices.contains(device) {
                return Err(Error::invalid(
                    "model.device",
                    format!(
                        "`{device}` not available on platform `{}` (has: {})",
                        platform.name(),
                        caps.devices.join(", ")
                    ),
                ));
            }
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
        // The engine may still be warming (plan build); its input
        // spec is checked now when ready, or at the moment it
        // becomes ready. Video flows either way.
        let (engine, input_spec) = match platform.engine(&config.model)? {
            EngineState::Ready(mut engine) => {
                let spec = engine.input_bindings().first().map(|t| t.spec.clone());
                check_input_spec(spec.as_ref(), &config.model)?;
                (EngineSlot::Ready(engine), spec)
            }
            EngineState::Warming(warmup) => (EngineSlot::Warming(warmup), None),
        };
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
        let stream = platform.stream(config.source.camera_id)?;

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
            stop: Arc::new(AtomicBool::new(false)),
            live_metrics: Arc::new(LiveMetrics::default()),
            stream,
        })
    }

    pub fn metadata_only(&self) -> bool {
        self.metadata_only
    }

    /// Counters shared with the running pipeline; snapshot them
    /// from any thread while `run` is in progress.
    pub fn live_metrics(&self) -> Arc<LiveMetrics> {
        Arc::clone(&self.live_metrics)
    }

    /// Handle for stopping `run()` from another thread (signal
    /// handlers, supervisors). Setting it ends the run cleanly:
    /// capture stops, the source is restored, and `run` returns
    /// the metrics gathered so far with `Ok(())`.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    /// Clear a stop request so the pipeline can run again. The
    /// only way the external flag is reset; `run` never does it.
    pub fn clear_stop(&self) {
        self.stop.store(false, Ordering::Relaxed);
    }

    /// The per-frame buffer budget, printed at startup so a hidden
    /// extra buffer is visible in the log, not just in a trace.
    /// Plane lines use the tight geometry implied by the config until
    /// a backend can report its real surface pitch (the NVMM branch
    /// fills this from `NvBufSurface` params on device).
    pub fn buffer_budget(&self) -> String {
        let s = &self.config.source;
        let m = &self.config.model;
        let planes = PixelFormat::Nv12.plane_layout(s.width, s.height);
        let capture = planes.byte_len();
        let (dtype, input) = match &self.input_spec {
            Some(spec) => (format!("{:?}", spec.dtype), spec.byte_len()),
            None => (
                "unknown (engine warming)".to_string(),
                // Mapped-host bindings are F32 in the T5 design; this
                // is only the startup estimate until the engine reports
                // its real spec.
                (m.input_width as usize) * (m.input_height as usize) * 3 * 4,
            ),
        };
        let mut plane_lines = String::new();
        for (i, p) in planes.planes().enumerate() {
            plane_lines.push_str(&format!(
                "\n    plane {i}: offset {} pitch {} row {} rows {}",
                p.offset, p.pitch, p.row_bytes, p.rows
            ));
        }
        format!(
            "buffer budget:\n  capture NV12 {}x{} {}: {} bytes x pool {}{plane_lines}\n  \
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

    /// Run until the source ends, `max_frames` frames are
    /// captured, the stop handle is set, or a stage fails. A stop
    /// requested before the run starts is honoured (the run ends
    /// immediately); call `clear_stop` to run again afterwards.
    ///
    /// Always returns the metrics gathered so far — including on
    /// the error path, where the first (processing) error wins —
    /// and always restores the source, so the pipeline can run
    /// again. The one exception is a capture-thread panic, where
    /// the source is lost with the thread.
    pub fn run(&mut self, max_frames: Option<u64>) -> (PipelineMetrics, Result<()>) {
        // The external stop flag is never reset here: a stop that
        // arrives before or between runs must be honoured, not
        // silently overwritten. Teardown uses its own per-run
        // flag instead.
        let shutdown = Arc::new(AtomicBool::new(false));
        self.live_metrics.reset();
        let mut metrics = PipelineMetrics {
            latency: self.live_metrics.latency_handle(),
            ..Default::default()
        };
        let limit = max_frames.or(self.config.source.frames);
        let max_idle = Duration::from_millis(self.config.pipeline.max_idle_ms);

        let (tx, rx, dropped) = channel::channel::<HwFrame>(self.config.pipeline.queue_depth);
        self.live_metrics.attach_dropped(Arc::clone(&dropped));
        let queue_depth = rx.depth_handle();
        self.live_metrics.attach_depth(Arc::clone(&queue_depth));
        let capture_stop = Arc::clone(&self.stop);
        let capture_shutdown = Arc::clone(&shutdown);
        let capture_live = Arc::clone(&self.live_metrics);

        // Temporarily move the source into the capture thread.
        // It travels through a slot so a failed spawn can hand it
        // back instead of dropping it with the unspawned closure.
        let source = std::mem::replace(&mut self.source, NoopSource::boxed());
        let slot = Arc::new(std::sync::Mutex::new(Some(source)));
        let thread_slot = Arc::clone(&slot);
        let capture = std::thread::Builder::new()
            .name(format!("capture-cam{}", self.config.source.camera_id))
            .spawn(move || {
                let mut source = thread_slot
                    .lock()
                    .expect("source slot poisoned")
                    .take()
                    .expect("source slot empty in capture thread");
                let (counts, result) = capture_loop(
                    &mut *source,
                    &tx,
                    &capture_stop,
                    &capture_shutdown,
                    &capture_live,
                    limit,
                    max_idle,
                );
                // stop() runs on every path; the loop's error wins.
                let result = result.and_then(|()| source.stop());
                (source, counts, result)
            });
        let capture = match capture {
            Ok(handle) => handle,
            Err(e) => {
                if let Some(source) = slot.lock().expect("source slot poisoned").take() {
                    self.source = source;
                }
                return (
                    metrics,
                    Err(Error::Platform(format!("spawning capture thread: {e}"))),
                );
            }
        };

        let mut detections = Detections::with_capacity(64);
        let mut last_err: Option<Error> = None;

        loop {
            if self.stop.load(Ordering::Relaxed) && rx.is_empty() {
                break;
            }
            let Some(frame) = rx.recv_timeout(Duration::from_millis(200)) else {
                // Closed-and-empty is the only safe exit: the
                // sender drops with the capture thread (even on
                // panic), while is_finished() could race a frame
                // sent just before the thread exited.
                if rx.is_closed_and_empty() {
                    break;
                }
                continue;
            };
            if let Err(e) = self.process_one(frame, &mut detections, &mut metrics) {
                last_err = Some(e);
                break;
            }
            self.live_metrics
                .sync_processing(&metrics, dropped.load(Ordering::Relaxed));
        }

        // Tell capture to stop and drop the receiver *before*
        // joining, so an unbounded source cannot block shutdown.
        shutdown.store(true, Ordering::Relaxed);
        drop(rx);
        // The receiver is gone: nothing is queued any more. Keep the
        // peak for the report, reset the current gauge.
        queue_depth.store(0, Ordering::Relaxed);
        let capture_result = match capture.join() {
            Ok((source, (captured, timeouts), result)) => {
                // Restore the source before inspecting any result:
                // a failed run must not leave a NoopSource behind.
                self.source = source;
                // Counts survive a capture error: partial metrics
                // are the evidence for what the source did.
                metrics.frames_captured = captured;
                metrics.capture_timeouts = timeouts;
                result
            }
            Err(_) => Err(Error::Platform("capture thread panicked".into())),
        };
        metrics.frames_dropped += dropped.load(Ordering::Relaxed);
        self.live_metrics
            .set_capture(metrics.frames_captured, metrics.capture_timeouts);
        self.live_metrics
            .sync_processing(&metrics, metrics.frames_dropped);
        let final_snapshot = self.live_metrics.snapshot();
        metrics.queue_depth = final_snapshot.queue_depth;
        metrics.queue_depth_max = final_snapshot.queue_depth_max;

        // Flush sinks on every path; an earlier error still wins.
        let finish_result = self.sink.finish();
        let result = match last_err {
            Some(e) => Err(e),
            None => capture_result.and(finish_result),
        };
        (metrics, result)
    }

    /// Advance a warming engine, if any. On transition the new
    /// engine's input spec gets the same check the builder
    /// applies to an engine that was ready up front.
    fn poll_engine(&mut self) -> Result<()> {
        let EngineSlot::Warming(warmup) = &mut self.engine else {
            return Ok(());
        };
        let Some(mut engine) = warmup.poll_ready()? else {
            return Ok(());
        };
        let spec = engine.input_bindings().first().map(|t| t.spec.clone());
        check_input_spec(spec.as_ref(), &self.config.model)?;
        self.input_spec = spec;
        self.engine = EngineSlot::Ready(engine);
        Ok(())
    }

    fn process_one(
        &mut self,
        frame: HwFrame,
        detections: &mut Detections,
        metrics: &mut PipelineMetrics,
    ) -> Result<()> {
        // Taken before the encoder consumes the frame: the age
        // covers queueing plus every stage and the sink call.
        let received_at = frame.received_at();
        self.poll_engine()?;
        match &mut self.engine {
            EngineSlot::Ready(engine) => {
                let t = Instant::now();
                let input = engine
                    .input_bindings()
                    .first_mut()
                    .ok_or_else(|| Error::Platform("engine exposes no input binding".into()))?;
                self.preprocessor.run(&frame, input, &self.stream)?;
                metrics.pre_total += t.elapsed();

                let t = Instant::now();
                engine.infer(&self.stream)?;
                metrics.infer_total += t.elapsed();

                let t = Instant::now();
                self.postprocessor
                    .decode(engine.output_bindings(), frame.meta(), detections)?;
                metrics.post_total += t.elapsed();
                metrics.detections += detections.len() as u64;
            }
            // Still warming: video and metadata flow with empty
            // detections rather than waiting for the engine.
            EngineSlot::Warming(_) => detections.clear(frame.meta()),
        }

        let t = Instant::now();
        self.overlay.draw(&frame, detections, &self.stream)?;
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
        metrics.latency.record(received_at.elapsed());
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
    fn layout(&self) -> edge_core::Layout {
        edge_core::Layout::PitchLinear
    }
}

/// The capture thread's loop: pull frames into the channel until
/// the source ends, the limit is reached, stop is requested, or
/// no frame has arrived within `max_idle` (0 disables the limit).
/// Returns ((captured, timeouts), result): the counts are always
/// returned, even when the loop ends in error, so partial metrics
/// survive. `source.stop()` is the caller's job, on every path.
fn capture_loop(
    source: &mut dyn FrameSource,
    tx: &channel::Sender<HwFrame>,
    stop: &AtomicBool,
    shutdown: &AtomicBool,
    live: &LiveMetrics,
    limit: Option<u64>,
    max_idle: Duration,
) -> ((u64, u64), Result<()>) {
    let mut captured = 0u64;
    let mut timeouts = 0u64;
    if let Err(e) = source.start() {
        return ((captured, timeouts), Err(e));
    }
    let mut last_frame = Instant::now();
    // Set when polls starve on the buffer pool: then the consumer
    // (not the camera) is what stopped, and the stall report
    // must say so.
    let mut pool_starved = false;
    loop {
        // External stop (supervisor/signal) or runner teardown.
        if stop.load(Ordering::Relaxed) || shutdown.load(Ordering::Relaxed) {
            break;
        }
        if let Some(limit) = limit {
            if captured >= limit {
                break;
            }
        }
        let poll = match source.next_frame(Duration::from_millis(100)) {
            Ok(poll) => poll,
            Err(e) => return ((captured, timeouts), Err(e)),
        };
        match poll {
            FramePoll::Frame(frame) => {
                captured += 1;
                pool_starved = false;
                live.set_capture(captured, timeouts);
                last_frame = Instant::now();
                // Drop-oldest: an evicted frame returns its
                // buffer to the capture pool here.
                drop(tx.send(frame));
            }
            // A momentary shortage is not the end of anything:
            // keep capturing — unless no frame has been handed
            // out past the idle limit, which a supervisor must
            // hear about, attributed to the side that stopped.
            // Normal pacing (NoFrame) is not counted as a timeout;
            // pool starvation is.
            FramePoll::Timeout(reason) => {
                if reason == edge_core::TimeoutReason::PoolExhausted {
                    timeouts += 1;
                    pool_starved = true;
                    live.set_capture(captured, timeouts);
                }
                if !max_idle.is_zero() && last_frame.elapsed() > max_idle {
                    let what = if pool_starved {
                        format!(
                            "processing stalled: capture pool exhausted, no frame \
                             handed out for {:?} (max_idle {:?})",
                            last_frame.elapsed(),
                            max_idle
                        )
                    } else {
                        format!(
                            "source stalled: no frame for {:?} (max_idle {:?})",
                            last_frame.elapsed(),
                            max_idle
                        )
                    };
                    return ((captured, timeouts), Err(Error::Platform(what)));
                }
            }
            FramePoll::EndOfStream => break,
        }
    }
    ((captured, timeouts), Ok(()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::TimeoutReason;
    use std::sync::atomic::AtomicBool;

    /// A source that never produces a frame, timing out for one
    /// fixed reason.
    struct SilentSource(TimeoutReason);
    impl FrameSource for SilentSource {
        fn start(&mut self) -> Result<()> {
            Ok(())
        }
        fn next_frame(&mut self, _t: Duration) -> Result<FramePoll> {
            Ok(FramePoll::Timeout(self.0))
        }
        fn stop(&mut self) -> Result<()> {
            Ok(())
        }
        fn memory(&self) -> edge_core::FrameMemory {
            edge_core::FrameMemory::DmaBuf
        }
        fn layout(&self) -> edge_core::Layout {
            edge_core::Layout::PitchLinear
        }
    }

    fn stall_error(reason: TimeoutReason) -> ((u64, u64), String) {
        let (tx, _rx, _dropped) = channel::channel::<HwFrame>(2);
        let stop = AtomicBool::new(false);
        let shutdown = AtomicBool::new(false);
        let live = LiveMetrics::default();
        let mut source = SilentSource(reason);
        let (counts, result) = capture_loop(
            &mut source,
            &tx,
            &stop,
            &shutdown,
            &live,
            None,
            Duration::from_millis(50),
        );
        (
            counts,
            result.expect_err("idle limit must trip").to_string(),
        )
    }

    /// R2-2: a quiet sensor is blamed on the source, and its
    /// pacing timeouts are not counted (R2-5).
    #[test]
    fn silent_sensor_reports_source_stalled() {
        let ((captured, timeouts), err) = stall_error(TimeoutReason::NoFrame);
        assert_eq!((captured, timeouts), (0, 0));
        assert!(err.contains("source stalled"), "{err}");
    }

    /// R2-2: a starved pool is blamed on processing, and counted.
    #[test]
    fn starved_pool_reports_processing_stalled() {
        let ((captured, timeouts), err) = stall_error(TimeoutReason::PoolExhausted);
        assert_eq!(captured, 0);
        assert!(timeouts > 0);
        assert!(err.contains("processing stalled"), "{err}");
    }
}
