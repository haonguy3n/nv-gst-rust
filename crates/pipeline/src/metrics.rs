use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Counters and per-stage timings, measured from day one.
/// Exported as logs now; a Prometheus exporter reads the same struct.
#[derive(Debug, Default, Clone)]
pub struct PipelineMetrics {
    pub frames_captured: u64,
    /// Frames evicted by drop-oldest backpressure.
    pub frames_dropped: u64,
    /// Capture polls starved by pool exhaustion (all buffers
    /// held downstream). Normal sensor pacing between frames is
    /// not counted. The run continues; this is the counter that
    /// makes a starving pool visible instead of silent.
    pub capture_timeouts: u64,
    pub frames_processed: u64,
    pub detections: u64,
    pub pre_total: Duration,
    pub infer_total: Duration,
    pub post_total: Duration,
    pub overlay_total: Duration,
    pub encode_total: Duration,
}

impl PipelineMetrics {
    fn avg(&self, total: Duration) -> Duration {
        if self.frames_processed == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos((total.as_nanos() / u128::from(self.frames_processed)) as u64)
        }
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
            "edge_frames_processed_total",
            "Frames that completed every stage",
            self.frames_processed,
        );
        counter(
            &mut out,
            "edge_detections_total",
            "Detections emitted",
            self.detections,
        );
        out.push_str(
            "# HELP edge_stage_seconds_total Cumulative stage time\n\
             # TYPE edge_stage_seconds_total counter\n",
        );
        for (stage, total) in [
            ("pre", self.pre_total),
            ("infer", self.infer_total),
            ("post", self.post_total),
            ("overlay", self.overlay_total),
            ("encode", self.encode_total),
        ] {
            out.push_str(&format!(
                "edge_stage_seconds_total{{stage=\"{stage}\"}} {:.6}\n",
                total.as_secs_f64()
            ));
        }
        out
    }

    pub fn summary(&self) -> String {
        format!(
            "captured={} dropped={} timeouts={} processed={} detections={} \
             avg pre={:?} infer={:?} post={:?} overlay={:?} encode={:?}",
            self.frames_captured,
            self.frames_dropped,
            self.capture_timeouts,
            self.frames_processed,
            self.detections,
            self.avg(self.pre_total),
            self.avg(self.infer_total),
            self.avg(self.post_total),
            self.avg(self.overlay_total),
            self.avg(self.encode_total),
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
    /// The channel's own drop counter for the current run, read
    /// directly so drops stay visible even while processing is
    /// stalled (when nothing calls `sync_processing`).
    dropped_source: Mutex<Option<Arc<AtomicU64>>>,
    capture_timeouts: AtomicU64,
    frames_processed: AtomicU64,
    detections: AtomicU64,
    pre_ns: AtomicU64,
    infer_ns: AtomicU64,
    post_ns: AtomicU64,
    overlay_ns: AtomicU64,
    encode_ns: AtomicU64,
}

impl LiveMetrics {
    /// Point-in-time copy of every counter.
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
            capture_timeouts: self.capture_timeouts.load(Ordering::Relaxed),
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            detections: self.detections.load(Ordering::Relaxed),
            pre_total: Duration::from_nanos(self.pre_ns.load(Ordering::Relaxed)),
            infer_total: Duration::from_nanos(self.infer_ns.load(Ordering::Relaxed)),
            post_total: Duration::from_nanos(self.post_ns.load(Ordering::Relaxed)),
            overlay_total: Duration::from_nanos(self.overlay_ns.load(Ordering::Relaxed)),
            encode_total: Duration::from_nanos(self.encode_ns.load(Ordering::Relaxed)),
        }
    }

    pub(crate) fn reset(&self) {
        for counter in [
            &self.frames_captured,
            &self.frames_dropped,
            &self.capture_timeouts,
            &self.frames_processed,
            &self.detections,
            &self.pre_ns,
            &self.infer_ns,
            &self.post_ns,
            &self.overlay_ns,
            &self.encode_ns,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }

    /// Read drops straight from the current run's channel counter.
    pub(crate) fn attach_dropped(&self, source: Arc<AtomicU64>) {
        *self
            .dropped_source
            .lock()
            .expect("live metrics mutex poisoned") = Some(source);
    }

    pub(crate) fn set_capture(&self, captured: u64, timeouts: u64) {
        self.frames_captured.store(captured, Ordering::Relaxed);
        self.capture_timeouts.store(timeouts, Ordering::Relaxed);
    }

    /// Publish the processing thread's counters (and the channel
    /// drop count it can see) after each frame.
    pub(crate) fn sync_processing(&self, metrics: &PipelineMetrics, dropped: u64) {
        self.frames_processed
            .store(metrics.frames_processed, Ordering::Relaxed);
        self.detections.store(metrics.detections, Ordering::Relaxed);
        self.frames_dropped.store(dropped, Ordering::Relaxed);
        self.pre_ns
            .store(metrics.pre_total.as_nanos() as u64, Ordering::Relaxed);
        self.infer_ns
            .store(metrics.infer_total.as_nanos() as u64, Ordering::Relaxed);
        self.post_ns
            .store(metrics.post_total.as_nanos() as u64, Ordering::Relaxed);
        self.overlay_ns
            .store(metrics.overlay_total.as_nanos() as u64, Ordering::Relaxed);
        self.encode_ns
            .store(metrics.encode_total.as_nanos() as u64, Ordering::Relaxed);
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
    fn prometheus_output_has_each_counter_once() {
        let m = PipelineMetrics {
            frames_captured: 30,
            frames_dropped: 2,
            frames_processed: 28,
            detections: 28,
            infer_total: Duration::from_millis(1500),
            ..Default::default()
        };
        let text = m.render_prometheus();
        assert!(text.contains("edge_frames_captured_total 30\n"));
        assert!(text.contains("edge_frames_dropped_total 2\n"));
        assert!(text.contains("edge_stage_seconds_total{stage=\"infer\"} 1.500000\n"));
        assert_eq!(
            text.matches("# HELP edge_stage_seconds_total").count(),
            1,
            "HELP must appear once per metric:\n{text}"
        );
    }
}
