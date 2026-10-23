use edge_core::{Config, Detections, EncodedPacket, Error, Result, Sink};
use edge_pipeline::Pipeline;
use platform_sim::{DelayedSimPlatform, SimPlatform, SimPostProcessor};
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
pool_size = 8
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
    let (metrics, result) = pipeline.run(None);
    result.expect("run pipeline");

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

struct FailingSink;
impl Sink for FailingSink {
    fn push_video(&mut self, _pkt: EncodedPacket) -> Result<()> {
        Ok(())
    }
    fn push_meta(&mut self, _det: &Detections) -> Result<()> {
        Err(Error::Platform("boom".into()))
    }
}

/// Regression for REVIEW P0-1: a stage error against an unbounded
/// source must return (with the error), never deadlock in join().
#[test]
fn stage_error_with_unbounded_source_returns_instead_of_deadlocking() {
    let mut cfg = test_config(0);
    cfg.source.frames = None;
    cfg.source.paced = false;
    let mut pipeline = Pipeline::build(
        &SimPlatform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(FailingSink),
    )
    .expect("build pipeline");
    let started = std::time::Instant::now();
    let (metrics, result) = pipeline.run(None);
    assert!(result.is_err(), "sink error must surface");
    // Partial metrics survive the error path (REVIEW item 4).
    assert!(metrics.frames_captured > 0, "{}", metrics.summary());
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "run() must return promptly after a stage error"
    );
}

struct SlowSink;
impl Sink for SlowSink {
    fn push_video(&mut self, _pkt: EncodedPacket) -> Result<()> {
        Ok(())
    }
    fn push_meta(&mut self, _det: &Detections) -> Result<()> {
        std::thread::sleep(std::time::Duration::from_millis(50));
        Ok(())
    }
}

/// Regression for REVIEW P0-2: the smallest legal pool with a slow
/// sink must account for every captured frame (processed or
/// dropped) and capture timeouts must be visible, never read as
/// end-of-stream that truncates the run silently.
#[test]
fn smallest_legal_pool_with_slow_sink_accounts_for_every_frame() {
    let mut cfg = test_config(30);
    cfg.source.pool_size = cfg.pipeline.queue_depth + 3; // smallest legal
    cfg.source.paced = false;
    let mut pipeline = Pipeline::build(
        &SimPlatform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(SlowSink),
    )
    .expect("build pipeline");
    let (metrics, result) = pipeline.run(None);
    result.expect("run pipeline");
    assert_eq!(metrics.frames_captured, 30, "{}", metrics.summary());
    assert_eq!(
        metrics.frames_processed + metrics.frames_dropped,
        metrics.frames_captured,
        "{}",
        metrics.summary()
    );
    assert!(metrics.frames_processed > 0);
}

/// Item 5: the stop handle ends an unbounded run promptly and
/// cleanly (Ok, with the frames processed so far counted).
#[test]
fn stop_handle_ends_unbounded_run() {
    let mut cfg = test_config(0);
    cfg.source.frames = None;
    let mut pipeline = Pipeline::build(
        &SimPlatform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(CountingSink::default()),
    )
    .expect("build pipeline");
    let stop = pipeline.stop_handle();
    let runner = std::thread::spawn(move || {
        let out = pipeline.run(None);
        (pipeline, out)
    });
    std::thread::sleep(std::time::Duration::from_millis(150));
    let stopped_at = std::time::Instant::now();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let (_pipeline, (metrics, result)) = runner.join().expect("runner thread panicked");
    assert!(
        stopped_at.elapsed() < std::time::Duration::from_secs(2),
        "run must observe the stop promptly"
    );
    result.expect("a requested stop is a clean Ok");
    assert!(metrics.frames_processed > 0, "{}", metrics.summary());
}

/// Item 6: a source that never produces a frame fails the run
/// at the idle limit instead of counting timeouts forever. The
/// sim source with an exhausted pool times out forever only if
/// frames are leaked; use fps pacing with a long frame gap
/// instead: 1 fps paced source, 300 ms idle limit — the second
/// frame is 1 s out, so the run must fail first.
#[test]
fn stalled_source_fails_at_idle_limit() {
    let mut cfg = test_config(0);
    cfg.source.frames = None;
    cfg.source.fps = 1;
    cfg.source.paced = true;
    cfg.pipeline.max_idle_ms = 300;
    // The paced sim sleeps inside next_frame until the frame is
    // due rather than timing out, so bound the whole test by
    // stopping from another thread if it ever regresses.
    let mut pipeline = Pipeline::build(
        &SimPlatform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(CountingSink::default()),
    )
    .expect("build pipeline");
    let stop = pipeline.stop_handle();
    // Watchdog only as a hang guard; deliberately not joined.
    let _watchdog = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(10));
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let started = std::time::Instant::now();
    let (metrics, result) = pipeline.run(None);
    let elapsed = started.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "idle limit must trip long before the watchdog (took {elapsed:?})"
    );
    assert!(result.is_err(), "stalled source must fail the run");
    // Partial capture counts survive the capture error.
    assert!(
        metrics.capture_timeouts > 0,
        "timeouts counted before the stall must be reported: {}",
        metrics.summary()
    );
}

/// Item 4/5: the shutdown path sets the internal stop flag; that
/// must not poison a second run on the same pipeline.
#[test]
fn pipeline_can_run_twice() {
    let mut cfg = test_config(0);
    cfg.source.frames = None;
    cfg.source.paced = false;
    let mut pipeline = Pipeline::build(
        &SimPlatform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(CountingSink::default()),
    )
    .expect("build pipeline");
    for run in 0..2 {
        let (metrics, result) = pipeline.run(Some(3));
        result.expect("run");
        assert_eq!(
            metrics.frames_captured,
            3,
            "run {run}: {}",
            metrics.summary()
        );
        assert!(
            metrics.frames_processed > 0,
            "run {run}: {}",
            metrics.summary()
        );
    }
}

/// Process item 1: counters are observable from another thread
/// while the run is still in progress, and the final snapshot
/// agrees with the returned metrics.
#[test]
fn live_metrics_are_visible_during_run() {
    let mut cfg = test_config(0);
    cfg.source.frames = None;
    cfg.source.paced = true;
    let mut pipeline = Pipeline::build(
        &SimPlatform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(CountingSink::default()),
    )
    .expect("build pipeline");
    let live = pipeline.live_metrics();
    let stop = pipeline.stop_handle();
    let runner = std::thread::spawn(move || pipeline.run(None));

    // Poll until the running pipeline reports progress.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut snap = live.snapshot();
    while snap.frames_processed == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "no live progress observed: {}",
            snap.summary()
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
        snap = live.snapshot();
    }
    assert!(snap.frames_captured >= snap.frames_processed);

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let (metrics, result) = runner.join().expect("runner thread");
    result.expect("run");
    let final_snap = live.snapshot();
    assert_eq!(final_snap.frames_captured, metrics.frames_captured);
    assert_eq!(final_snap.frames_processed, metrics.frames_processed);
    assert_eq!(final_snap.detections, metrics.detections);
}

/// C2 (Jetson inference plan): while the engine warms up (plan
/// build), frames flow with empty detections; once it is ready,
/// the same run starts producing detections — video and metadata
/// both flow from the first frame.
#[test]
fn warming_engine_flows_empty_then_detects() {
    #[derive(Default)]
    struct LenSink {
        seen: Arc<Mutex<Vec<usize>>>,
    }
    impl Sink for LenSink {
        fn push_video(&mut self, _pkt: EncodedPacket) -> Result<()> {
            Ok(())
        }
        fn push_meta(&mut self, det: &Detections) -> Result<()> {
            self.seen.lock().unwrap().push(det.items.len());
            Ok(())
        }
    }

    // 60 frames at 30 fps paced = ~2 s of run; engine ready at 0.6 s.
    let cfg = test_config(60);
    let platform = DelayedSimPlatform {
        ready_after: std::time::Duration::from_millis(600),
    };
    let sink = LenSink::default();
    let seen = Arc::clone(&sink.seen);
    let mut pipeline = Pipeline::build(
        &platform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(sink),
    )
    .expect("build pipeline");
    let (metrics, result) = pipeline.run(None);
    result.expect("run pipeline");

    assert_eq!(
        metrics.frames_processed + metrics.frames_dropped,
        metrics.frames_captured,
        "every captured frame accounted for: {}",
        metrics.summary()
    );
    let lens = seen.lock().unwrap();
    assert!(!lens.is_empty(), "metadata must flow from frame one");
    assert_eq!(
        lens[0], 0,
        "first frames carry empty detections while warming"
    );
    assert!(
        lens.iter().any(|&n| n > 0),
        "detections must start once the engine is ready: {lens:?}"
    );
    assert!(
        *lens.last().unwrap() > 0,
        "final frames must have detections: {lens:?}"
    );
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
