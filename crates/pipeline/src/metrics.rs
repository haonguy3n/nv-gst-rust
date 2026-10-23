use std::time::Duration;

/// Counters and per-stage timings, measured from day one.
/// Exported as logs now; a Prometheus exporter reads the same struct.
#[derive(Debug, Default, Clone)]
pub struct PipelineMetrics {
    pub frames_captured: u64,
    /// Frames evicted by drop-oldest backpressure.
    pub frames_dropped: u64,
    /// Capture polls that timed out (e.g. pool temporarily
    /// exhausted). The run continues; this is the counter that
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
            "Capture polls that timed out",
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

#[cfg(test)]
mod tests {
    use super::*;

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
