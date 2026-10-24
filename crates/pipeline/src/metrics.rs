use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Fixed-size ring of end-to-end ages (~9 minutes at 30 fps).
/// Preallocated: recording a frame never allocates, and a process
/// that runs for days keeps a bounded, recent window instead of
/// growing until it is OOM-killed.
pub const LATENCY_WINDOW: usize = 16_384;

#[derive(Debug)]
struct LatencyRing {
    samples: Box<[u32]>,
    len: usize,
    next: usize,
}

impl LatencyRing {
    fn new() -> Self {
        Self {
            samples: vec![0u32; LATENCY_WINDOW].into_boxed_slice(),
            len: 0,
            next: 0,
        }
    }

    fn record(&mut self, us: u32) {
        if self.len < LATENCY_WINDOW {
            self.samples[self.len] = us;
            self.len += 1;
        } else {
            self.samples[self.next] = us;
            self.next = (self.next + 1) % LATENCY_WINDOW;
        }
    }

    fn clear(&mut self) {
        self.len = 0;
        self.next = 0;
    }
}

/// Bounded, allocation-free reservoir of end-to-end frame ages
/// (source hand-off to sink accept, in microseconds). Shared with
/// [`LiveMetrics`] so a supervisor can read percentiles during a run
/// without waiting for `run()` to return.
#[derive(Debug, Clone)]
pub struct LatencySamples {
    ring: Arc<Mutex<LatencyRing>>,
}

impl Default for LatencySamples {
    fn default() -> Self {
        Self {
            ring: Arc::new(Mutex::new(LatencyRing::new())),
        }
    }
}

impl LatencySamples {
    pub fn record(&self, age: Duration) {
        let us = age.as_micros().min(u128::from(u32::MAX)) as u32;
        if let Ok(mut ring) = self.ring.lock() {
            ring.record(us);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut ring) = self.ring.lock() {
            ring.clear();
        }
    }

    pub fn len(&self) -> usize {
        self.ring.lock().map(|ring| ring.len).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sorted copy, allocated only when a percentile is rendered.
    fn sorted(&self) -> Vec<u32> {
        let mut samples = match self.ring.lock() {
            Ok(ring) => ring.samples[..ring.len].to_vec(),
            Err(_) => Vec::new(),
        };
        samples.sort_unstable();
        samples
    }

    /// Nearest-rank percentile (`q` in `0.0..=1.0`), computed on a
    /// sorted copy so the hot path stays a single push.
    pub fn percentile(&self, q: f32) -> Option<Duration> {
        if !(0.0..=1.0).contains(&q) {
            return None;
        }
        let samples = self.sorted();
        if samples.is_empty() {
            return None;
        }
        let rank = (q * samples.len() as f32).ceil() as usize;
        let index = rank.saturating_sub(1).min(samples.len() - 1);
        Some(Duration::from_micros(u64::from(samples[index])))
    }

    pub fn p50(&self) -> Option<Duration> {
        self.percentile(0.50)
    }

    pub fn p95(&self) -> Option<Duration> {
        self.percentile(0.95)
    }

    pub fn p99(&self) -> Option<Duration> {
        self.percentile(0.99)
    }

    pub fn max(&self) -> Option<Duration> {
        let max = self
            .ring
            .lock()
            .map(|ring| ring.samples[..ring.len].iter().copied().max())
            .unwrap_or(None)?;
        Some(Duration::from_micros(u64::from(max)))
    }
}

/// Counters and per-stage timings, measured from day one.
#[derive(Debug, Default, Clone)]
pub struct PipelineMetrics {
    pub frames_captured: u64,
    /// Frames evicted by drop-oldest backpressure.
    pub frames_dropped: u64,
    /// Frames dropped by the leaky processing → encoder hand-off
    /// (encoder slower than the sensor).
    pub encode_dropped: u64,
    /// Capture polls starved by pool exhaustion (all buffers
    /// held downstream). Normal sensor pacing between frames is
    /// not counted. The run continues; this is the counter that
    /// makes a starving pool visible instead of silent.
    pub capture_timeouts: u64,
    pub frames_processed: u64,
    /// Frames that ran pre/infer/post (the engine was ready). The stage
    /// averages divide by this, not by `frames_processed`.
    pub frames_inferred: u64,
    pub detections: u64,
    pub pre_total: Duration,
    pub infer_total: Duration,
    /// GPU time of `infer` from CUDA events, summed over the
    /// `frames_gpu_timed` frames whose engine reported it.
    pub gpu_total: Duration,
    pub frames_gpu_timed: u64,
    pub post_total: Duration,
    pub overlay_total: Duration,
    pub encode_push_total: Duration,
    /// End-to-end age samples: source hand-off to sink accept.
    pub latency: LatencySamples,
    /// Frames queued between capture and processing right now.
    pub queue_depth: u64,
    /// Peak queue depth observed during the run.
    pub queue_depth_max: u64,
}

impl PipelineMetrics {
    fn avg_over(frames: u64, total: Duration) -> Duration {
        if frames == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos((total.as_nanos() / u128::from(frames)) as u64)
        }
    }

    /// Average of a stage that only runs on inferred frames.
    fn avg(&self, total: Duration) -> Duration {
        Self::avg_over(self.frames_inferred, total)
    }

    /// Average GPU time per timed frame; `None` when the engine reports none.
    fn avg_gpu(&self) -> Option<Duration> {
        (self.frames_gpu_timed > 0).then(|| Self::avg_over(self.frames_gpu_timed, self.gpu_total))
    }

    /// Average of a stage that runs on every processed frame.
    fn avg_processed(&self, total: Duration) -> Duration {
        Self::avg_over(self.frames_processed, total)
    }

    /// Prometheus text exposition, ready to serve from a `/metrics`
    /// endpoint in the sinks/HTTP layer.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();
        let counter = |out: &mut String, name: &str, help: &str, value: u64| {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {value}\n"
            ));
        };
        let gauge = |out: &mut String, name: &str, help: &str, value: String| {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {value}\n"
            ));
        };
        counter(
            &mut out,
            "edge_frames_captured_total",
            "Frames handed out by the source",
            self.frames_captured,
        );
        counter(
            &mut out,
            "edge_frames_dropped_total",
            "Frames captured then evicted by drop-oldest backpressure",
            self.frames_dropped,
        );
        counter(
            &mut out,
            "edge_capture_timeouts_total",
            "Capture polls starved by pool exhaustion",
            self.capture_timeouts,
        );
        counter(
            &mut out,
            "edge_encode_dropped_total",
            "Frames dropped by the leaky processing-to-encoder hand-off",
            self.encode_dropped,
        );
        counter(
            &mut out,
            "edge_frames_processed_total",
            "Frames that completed every stage",
            self.frames_processed,
        );
        counter(
            &mut out,
            "edge_frames_inferred_total",
            "Frames that ran the detector (engine ready)",
            self.frames_inferred,
        );
        counter(
            &mut out,
            "edge_detections_total",
            "Detections emitted",
            self.detections,
        );
        gauge(
            &mut out,
            "edge_queue_depth",
            "Frames queued between capture and processing",
            self.queue_depth.to_string(),
        );
        gauge(
            &mut out,
            "edge_queue_depth_max",
            "Peak queue depth during the run",
            self.queue_depth_max.to_string(),
        );
        out.push_str(
            "# HELP edge_latency_seconds End-to-end frame age (source hand-off to sink accept)\n\
             # TYPE edge_latency_seconds gauge\n",
        );
        for (quantile, value) in [
            ("0.5", self.latency.p50()),
            ("0.95", self.latency.p95()),
            ("0.99", self.latency.p99()),
        ] {
            if let Some(duration) = value {
                out.push_str(&format!(
                    "edge_latency_seconds{{quantile=\"{quantile}\"}} {:.6}\n",
                    duration.as_secs_f64()
                ));
            }
        }
        if let Some(max) = self.latency.max() {
            out.push_str(&format!(
                "edge_latency_max_seconds {:.6}\n",
                max.as_secs_f64()
            ));
        }
        out.push_str(
            "# HELP edge_stage_seconds_total Cumulative stage time\n\
             # TYPE edge_stage_seconds_total counter\n",
        );
        for (stage, total) in [
            ("pre", self.pre_total),
            ("infer", self.infer_total),
            ("post", self.post_total),
            ("overlay", self.overlay_total),
            ("encode_push", self.encode_push_total),
        ] {
            out.push_str(&format!(
                "edge_stage_seconds_total{{stage=\"{stage}\"}} {:.6}\n",
                total.as_secs_f64()
            ));
        }
        counter(
            &mut out,
            "edge_infer_gpu_frames_total",
            "Inferred frames with a GPU time from CUDA events",
            self.frames_gpu_timed,
        );
        out.push_str(&format!(
            "# HELP edge_infer_gpu_seconds_total Cumulative GPU time of inference (CUDA events)\n\
             # TYPE edge_infer_gpu_seconds_total counter\n\
             edge_infer_gpu_seconds_total {:.6}\n",
            self.gpu_total.as_secs_f64()
        ));
        out
    }

    pub fn summary(&self) -> String {
        let p = |value: Option<Duration>| match value {
            Some(d) => format!("{d:?}"),
            None => "n/a".to_string(),
        };
        format!(
            "captured={} dropped={} enc_dropped={} timeouts={} processed={} inferred={} detections={} \
             qmax={} p50={} p95={} p99={} \
             avg pre={:?} infer={:?}{} post={:?} overlay={:?} encode_push={:?}",
            self.frames_captured,
            self.frames_dropped,
            self.encode_dropped,
            self.capture_timeouts,
            self.frames_processed,
            self.frames_inferred,
            self.detections,
            self.queue_depth_max,
            p(self.latency.p50()),
            p(self.latency.p95()),
            p(self.latency.p99()),
            self.avg(self.pre_total),
            self.avg(self.infer_total),
            self.avg_gpu().map_or_else(String::new, |d| format!(" gpu={d:?}")),
            self.avg(self.post_total),
            self.avg(self.overlay_total),
            self.avg_processed(self.encode_push_total),
        )
    }
}

/// Live view of the counters, updated by the capture and
/// processing threads as the run progresses. Handed out by
/// `Pipeline::live_metrics` so a supervisor (or a `/metrics`
/// endpoint) can observe a long-running pipeline instead of
/// waiting for `run()` to return.
///
/// Counters reset at the start of every `run()`; a Prometheus
/// scraper sees that as a counter reset (the usual process-
/// restart semantics), which rate functions handle.
#[derive(Debug, Default)]
pub struct LiveMetrics {
    frames_captured: AtomicU64,
    frames_dropped: AtomicU64,
    encode_dropped: AtomicU64,
    /// The channel's own drop counter for the current run, read
    /// directly so drops stay visible even while processing is
    /// stalled (when nothing calls `sync_processing`).
    dropped_source: Mutex<Option<Arc<AtomicU64>>>,
    /// The channel's own depth counter, for the same reason.
    depth_source: Mutex<Option<Arc<AtomicU64>>>,
    capture_timeouts: AtomicU64,
    frames_processed: AtomicU64,
    frames_inferred: AtomicU64,
    detections: AtomicU64,
    queue_depth: AtomicU64,
    queue_depth_max: AtomicU64,
    pre_ns: AtomicU64,
    infer_ns: AtomicU64,
    gpu_ns: AtomicU64,
    frames_gpu_timed: AtomicU64,
    post_ns: AtomicU64,
    overlay_ns: AtomicU64,
    encode_push_ns: AtomicU64,
    latency: LatencySamples,
}

impl LiveMetrics {
    fn attached_depth(&self) -> u64 {
        self.depth_source
            .lock()
            .expect("live metrics mutex poisoned")
            .as_ref()
            .map_or_else(
                || self.queue_depth.load(Ordering::Relaxed),
                |source| source.load(Ordering::Relaxed),
            )
    }

    /// Point-in-time copy of every counter and percentile input.
    pub fn snapshot(&self) -> PipelineMetrics {
        PipelineMetrics {
            frames_captured: self.frames_captured.load(Ordering::Relaxed),
            frames_dropped: self
                .dropped_source
                .lock()
                .expect("live metrics mutex poisoned")
                .as_ref()
                .map_or_else(
                    || self.frames_dropped.load(Ordering::Relaxed),
                    |source| source.load(Ordering::Relaxed),
                ),
            encode_dropped: self.encode_dropped.load(Ordering::Relaxed),
            capture_timeouts: self.capture_timeouts.load(Ordering::Relaxed),
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            frames_inferred: self.frames_inferred.load(Ordering::Relaxed),
            detections: self.detections.load(Ordering::Relaxed),
            pre_total: Duration::from_nanos(self.pre_ns.load(Ordering::Relaxed)),
            infer_total: Duration::from_nanos(self.infer_ns.load(Ordering::Relaxed)),
            gpu_total: Duration::from_nanos(self.gpu_ns.load(Ordering::Relaxed)),
            frames_gpu_timed: self.frames_gpu_timed.load(Ordering::Relaxed),
            post_total: Duration::from_nanos(self.post_ns.load(Ordering::Relaxed)),
            overlay_total: Duration::from_nanos(self.overlay_ns.load(Ordering::Relaxed)),
            encode_push_total: Duration::from_nanos(self.encode_push_ns.load(Ordering::Relaxed)),
            latency: self.latency.clone(),
            queue_depth: self.attached_depth(),
            queue_depth_max: self.queue_depth_max.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn reset(&self) {
        for counter in [
            &self.frames_captured,
            &self.frames_dropped,
            &self.encode_dropped,
            &self.capture_timeouts,
            &self.frames_processed,
            &self.frames_inferred,
            &self.detections,
            &self.queue_depth,
            &self.queue_depth_max,
            &self.pre_ns,
            &self.infer_ns,
            &self.gpu_ns,
            &self.frames_gpu_timed,
            &self.post_ns,
            &self.overlay_ns,
            &self.encode_push_ns,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        self.latency.clear();
    }

    /// Read drops straight from the current run's channel counter.
    pub(crate) fn attach_dropped(&self, source: Arc<AtomicU64>) {
        *self
            .dropped_source
            .lock()
            .expect("live metrics mutex poisoned") = Some(source);
    }

    /// Read queue depth straight from the current run's channel.
    pub(crate) fn attach_depth(&self, source: Arc<AtomicU64>) {
        *self
            .depth_source
            .lock()
            .expect("live metrics mutex poisoned") = Some(source);
    }

    /// The shared latency reservoir, so the returned
    /// [`PipelineMetrics`] and live snapshots see the same samples.
    pub(crate) fn latency_handle(&self) -> LatencySamples {
        self.latency.clone()
    }

    pub(crate) fn set_capture(&self, captured: u64, timeouts: u64) {
        self.frames_captured.store(captured, Ordering::Relaxed);
        self.capture_timeouts.store(timeouts, Ordering::Relaxed);
    }

    /// Publish the processing thread's counters after each frame,
    /// including the queue depth the channel reports right now.
    pub(crate) fn sync_processing(&self, metrics: &PipelineMetrics, dropped: u64) {
        self.frames_processed
            .store(metrics.frames_processed, Ordering::Relaxed);
        self.frames_inferred
            .store(metrics.frames_inferred, Ordering::Relaxed);
        self.detections.store(metrics.detections, Ordering::Relaxed);
        self.frames_dropped.store(dropped, Ordering::Relaxed);
        self.encode_dropped
            .store(metrics.encode_dropped, Ordering::Relaxed);
        let depth = self.attached_depth();
        self.queue_depth.store(depth, Ordering::Relaxed);
        self.queue_depth_max.fetch_max(depth, Ordering::Relaxed);
        self.pre_ns
            .store(metrics.pre_total.as_nanos() as u64, Ordering::Relaxed);
        self.infer_ns
            .store(metrics.infer_total.as_nanos() as u64, Ordering::Relaxed);
        self.gpu_ns
            .store(metrics.gpu_total.as_nanos() as u64, Ordering::Relaxed);
        self.frames_gpu_timed
            .store(metrics.frames_gpu_timed, Ordering::Relaxed);
        self.post_ns
            .store(metrics.post_total.as_nanos() as u64, Ordering::Relaxed);
        self.overlay_ns
            .store(metrics.overlay_total.as_nanos() as u64, Ordering::Relaxed);
        self.encode_push_ns.store(
            metrics.encode_push_total.as_nanos() as u64,
            Ordering::Relaxed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R2-6: drops read from the channel counter are visible in
    /// snapshots without any processing-side sync.
    #[test]
    fn live_drops_are_visible_without_processing_sync() {
        let live = LiveMetrics::default();
        let dropped = Arc::new(AtomicU64::new(0));
        live.attach_dropped(Arc::clone(&dropped));
        dropped.store(5, Ordering::Relaxed);
        assert_eq!(live.snapshot().frames_dropped, 5);
    }

    #[test]
    fn live_queue_depth_is_visible_without_processing_sync() {
        let live = LiveMetrics::default();
        let depth = Arc::new(AtomicU64::new(0));
        live.attach_depth(Arc::clone(&depth));
        depth.store(3, Ordering::Relaxed);
        assert_eq!(live.snapshot().queue_depth, 3);
    }

    #[test]
    fn latency_percentiles_are_nearest_rank() {
        let samples = LatencySamples::default();
        assert_eq!(samples.p50(), None);
        for ms in 1..=100 {
            samples.record(Duration::from_millis(ms));
        }
        assert_eq!(samples.p50(), Some(Duration::from_millis(50)));
        assert_eq!(samples.p95(), Some(Duration::from_millis(95)));
        assert_eq!(samples.p99(), Some(Duration::from_millis(99)));
        assert_eq!(samples.max(), Some(Duration::from_millis(100)));
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn prometheus_output_has_each_metric_once() {
        let samples = LatencySamples::default();
        samples.record(Duration::from_millis(20));
        samples.record(Duration::from_millis(30));
        let m = PipelineMetrics {
            frames_captured: 30,
            frames_dropped: 2,
            frames_processed: 28,
            frames_inferred: 3,
            detections: 28,
            infer_total: Duration::from_millis(1500),
            latency: samples,
            queue_depth: 1,
            queue_depth_max: 4,
            ..Default::default()
        };
        let text = m.render_prometheus();
        assert!(text.contains("edge_frames_captured_total 30\n"));
        assert!(text.contains("edge_frames_dropped_total 2\n"));
        assert!(text.contains("edge_frames_inferred_total 3\n"));
        assert!(text.contains("edge_queue_depth 1\n"));
        assert!(text.contains("edge_queue_depth_max 4\n"));
        assert!(text.contains("edge_latency_seconds{quantile=\"0.5\"} 0.020000\n"));
        assert!(
            text.contains("edge_stage_seconds_total{stage=\"infer\"} 1.500000\n"),
            "{text}"
        );
        assert_eq!(
            text.matches("# HELP edge_stage_seconds_total").count(),
            1,
            "HELP must appear once per metric:\n{text}"
        );
    }

    #[test]
    fn gpu_time_shows_only_when_reported() {
        let mut m = PipelineMetrics {
            frames_inferred: 4,
            infer_total: Duration::from_millis(20),
            ..Default::default()
        };
        assert!(!m.summary().contains("gpu="), "{}", m.summary());
        m.gpu_total = Duration::from_millis(8);
        m.frames_gpu_timed = 4;
        assert!(m.summary().contains("infer=5ms gpu=2ms"), "{}", m.summary());
        let text = m.render_prometheus();
        assert!(text.contains("edge_infer_gpu_frames_total 4\n"), "{text}");
        assert!(
            text.contains("edge_infer_gpu_seconds_total 0.008000\n"),
            "{text}"
        );
    }

    /// Frames processed while the engine warmed must not dilute the averages.
    #[test]
    fn stage_averages_divide_by_inferred_frames() {
        let m = PipelineMetrics {
            frames_processed: 1000,
            frames_inferred: 10,
            infer_total: Duration::from_millis(50),
            encode_push_total: Duration::from_millis(1000),
            ..Default::default()
        };
        let line = m.summary();
        assert!(line.contains("inferred=10"), "{line}");
        assert!(line.contains("infer=5ms"), "{line}");
        assert!(line.contains("encode_push=1ms"), "{line}");
        assert_eq!(
            PipelineMetrics::default().avg(Duration::from_secs(1)),
            Duration::ZERO
        );
    }
}
