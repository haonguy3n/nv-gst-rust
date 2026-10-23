use std::time::Duration;

/// Counters and per-stage timings, measured from day one.
/// Exported as logs now; a Prometheus exporter reads the same struct.
#[derive(Debug, Default, Clone)]
pub struct PipelineMetrics {
    pub frames_captured: u64,
    /// Frames evicted by drop-oldest backpressure or lost to an
    /// exhausted capture pool.
    pub frames_dropped: u64,
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
            total / u32::try_from(self.frames_processed).unwrap_or(u32::MAX)
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "captured={} dropped={} processed={} detections={} \
             avg pre={:?} infer={:?} post={:?} overlay={:?} encode={:?}",
            self.frames_captured,
            self.frames_dropped,
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
