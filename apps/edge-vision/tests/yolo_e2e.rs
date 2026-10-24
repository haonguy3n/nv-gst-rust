//! End-to-end detector test: ONNX fixture -> sim preprocessor ->
//! ONNX Runtime -> YOLOv8 decoder + NMS -> sink.
//!
//! This is the committed replacement for the manual YOLOX runs the
//! early reviews flagged (P2-16): the fixture emits a constant
//! `[1, 7, 4]` head with one strong class-1 box, one overlapping
//! duplicate (NMS must suppress it), one below-threshold candidate
//! and one empty candidate. Nothing here depends on hardware, but
//! it exercises every CPU stage through the real `Pipeline`.
#![cfg(feature = "onnx")]

use edge_core::{Config, Detection, Detections, EncodedPacket, Result, Sink};
use edge_pipeline::Pipeline;
use platform_sim::SimPlatform;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct CollectSink {
    detections: Arc<Mutex<Vec<Detection>>>,
    frames: Arc<Mutex<u64>>,
}

impl Sink for CollectSink {
    fn push_video(&mut self, _pkt: EncodedPacket) -> Result<()> {
        panic!("sim has no encoder; video must never be pushed");
    }

    fn push_meta(&mut self, det: &Detections) -> Result<()> {
        *self.frames.lock().unwrap() += 1;
        self.detections
            .lock()
            .unwrap()
            .extend_from_slice(&det.items);
        Ok(())
    }
}

fn config() -> Config {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/platform-sim/tests/fixtures/yolo-const.onnx");
    Config::from_toml_str(&format!(
        r#"
platform = "sim"
[source]
kind = "synthetic"
width = 64
height = 64
fps = 30
paced = false
pool_size = 8
frames = 4
[model]
path = "{}"
decoder = "yolo"
input_width = 64
input_height = 64
num_classes = 3
score_threshold = 0.25
[stream]
metadata_only = true
[pipeline]
queue_depth = 4
"#,
        fixture.display()
    ))
    .expect("test config")
}

#[test]
fn yolo_fixture_runs_end_to_end_and_nms_suppresses_the_duplicate() {
    let sink = CollectSink::default();
    let seen = Arc::clone(&sink.detections);
    let frames = Arc::clone(&sink.frames);
    let mut pipeline = Pipeline::build(
        &SimPlatform,
        config(),
        Box::new(
            edge_postproc::YoloPostProcessor::new(64, 64, 3, 0.25)
                .with_iou_threshold(0.45)
                .with_top_k(100),
        ),
        Box::new(sink),
    )
    .expect("build pipeline");

    let (metrics, result) = pipeline.run(None);
    result.expect("run pipeline");
    assert_eq!(metrics.frames_captured, 4, "{}", metrics.summary());
    assert!(metrics.frames_processed > 0, "{}", metrics.summary());
    assert_eq!(
        *frames.lock().unwrap(),
        metrics.frames_processed,
        "every processed frame reaches the sink"
    );
    assert_eq!(
        metrics.detections,
        metrics.frames_processed,
        "exactly one detection per frame after NMS: {}",
        metrics.summary()
    );

    let detections = seen.lock().unwrap();
    assert_eq!(detections.len() as u64, metrics.frames_processed);
    for d in detections.iter() {
        assert_eq!(d.class_id, 1, "{d:?}");
        assert!((d.score - 0.9).abs() < 1e-5, "{d:?}");
        // Model and frame are both 64x64: no scale, no padding.
        assert!((d.x - 0.375).abs() < 1e-5, "{d:?}");
        assert!((d.y - 0.375).abs() < 1e-5, "{d:?}");
        assert!((d.width - 0.25).abs() < 1e-5, "{d:?}");
        assert!((d.height - 0.25).abs() < 1e-5, "{d:?}");
    }
}
