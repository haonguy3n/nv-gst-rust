//! Throughput bench: how fast can this machine push frames through
//! every pipeline stage with the simulator backend?
//!
//!     cargo run --release -p edge-pipeline --example bench [FRAMES]
//!
//! The bench disables the sim source's sensor pacing
//! (`paced = false`) and sizes the pool and queue so backpressure
//! never engages. What remains is the per-stage cost the pipeline
//! itself adds.

use edge_core::Config;
use edge_pipeline::Pipeline;
use edge_sinks::NullSink;
use platform_sim::{SimPlatform, SimPostProcessor};
use std::time::Instant;

fn main() {
    let frames: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2000);

    let cfg = Config::from_toml_str(&format!(
        r#"
platform = "sim"
[source]
kind = "synthetic"
width = 640
height = 480
fps = 240
paced = false
pool_size = 512
frames = {frames}
[model]
input_width = 320
input_height = 320
[stream]
metadata_only = true
[pipeline]
queue_depth = 256
"#
    ))
    .expect("bench config");

    let mut pipeline = Pipeline::build(
        &SimPlatform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(NullSink),
    )
    .expect("build pipeline");

    let started = Instant::now();
    let metrics = pipeline.run(None).expect("run pipeline");
    let elapsed = started.elapsed();

    let fps = metrics.frames_processed as f64 / elapsed.as_secs_f64();
    println!(
        "processed {} frames in {:?} ({:.0} fps, {:.1} us/frame)",
        metrics.frames_processed,
        elapsed,
        fps,
        elapsed.as_micros() as f64 / metrics.frames_processed.max(1) as f64
    );
    println!("{}", metrics.summary());
    assert_eq!(
        metrics.frames_processed + metrics.frames_dropped,
        metrics.frames_captured,
        "every captured frame must be accounted for"
    );
}
