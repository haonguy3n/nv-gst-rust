use edge_core::{Config, Detections, EncodedPacket, Result, Sink};
use edge_pipeline::Pipeline;
use platform_sim::{SimPlatform, SimPostProcessor};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct CountingSink {
    meta_frames: Arc<Mutex<Vec<u64>>>,
}

impl Sink for CountingSink {
    fn push_video(&mut self, _pkt: EncodedPacket) -> Result<()> {
        panic!("sim has no encoder; video must never be pushed");
    }
    fn push_meta(&mut self, det: &Detections) -> Result<()> {
        self.meta_frames.lock().unwrap().push(det.frame_seq);
        Ok(())
    }
}

fn test_config(frames: u64) -> Config {
    Config::from_toml_str(&format!(
        r#"
platform = "sim"
[source]
kind = "synthetic"
width = 640
height = 480
fps = 30
pool_size = 6
frames = {frames}
[model]
input_width = 320
input_height = 320
score_threshold = 0.25
[stream]
metadata_only = true
"#
    ))
    .expect("test config")
}

#[test]
fn sim_pipeline_processes_every_frame_in_order() {
    let platform = SimPlatform;
    let cfg = test_config(10);
    let sink = CountingSink::default();
    let seen = Arc::clone(&sink.meta_frames);
    let mut pipeline = Pipeline::build(
        &platform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(sink),
    )
    .expect("build pipeline");

    assert!(pipeline.metadata_only());
    let metrics = pipeline.run(None).expect("run pipeline");

    // Channel may drop under backpressure, but processed frames must
    // remain a strictly increasing subsequence of captured frames.
    assert_eq!(metrics.frames_captured, 10);
    assert_eq!(
        metrics.frames_processed + metrics.frames_dropped,
        metrics.frames_captured,
        "every captured frame is processed or counted as dropped: {}",
        metrics.summary()
    );
    assert!(metrics.frames_processed > 0);
    // Exactly one above-threshold detection per processed frame.
    assert_eq!(metrics.detections, metrics.frames_processed);

    let seqs = seen.lock().unwrap();
    assert_eq!(seqs.len() as u64, metrics.frames_processed);
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "seq order: {seqs:?}");
}

#[test]
fn builder_refuses_encoder_on_platform_without_hw_encoder() {
    let platform = SimPlatform;
    let mut cfg = test_config(1);
    cfg.encoder.enabled = true;
    cfg.stream.metadata_only = false;
    let result = Pipeline::build(
        &platform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(CountingSink::default()),
    );
    match result {
        Ok(_) => panic!("build must refuse a hardware encoder on sim"),
        Err(e) => assert!(
            e.to_string().contains("no hardware encoder"),
            "unexpected error: {e}"
        ),
    }
}
