//! Encoding runs on its own thread (N7): a slow CPU encoder must not
//! stretch the inference loop, and packets must still reach the sink
//! in order. Uses a fake encoder over the sim backend so the test runs
//! anywhere.

use edge_core::{
    Config, EncodedPacket, Encoder, EncoderConfig, EngineState, Error, FrameSource, GpuStream,
    HwFrame, ModelConfig, Overlay, Platform, PlatformCaps, PreConfig, Preprocessor, Result,
    SourceConfig,
};
use edge_pipeline::Pipeline;
use edge_sinks::{CountingSink, Counts};
use platform_sim::{SimPlatform, SimPostProcessor};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Default)]
struct EncoderEvents {
    thread: Option<String>,
    frames: u64,
    finish_packets: u64,
}

struct FakeEncoder {
    slow: Duration,
    fail_after: Option<u64>,
    events: Arc<Mutex<EncoderEvents>>,
}

impl Encoder for FakeEncoder {
    fn encode(&mut self, frame: HwFrame, out: &mut dyn FnMut(EncodedPacket)) -> Result<()> {
        let mut events = self.events.lock().unwrap();
        events.thread = Some(
            std::thread::current()
                .name()
                .unwrap_or("<unnamed>")
                .to_string(),
        );
        events.frames += 1;
        let frames = events.frames;
        drop(events);
        if self.fail_after.is_some_and(|limit| frames > limit) {
            return Err(Error::Platform("encoder boom".into()));
        }
        std::thread::sleep(self.slow);
        out(EncodedPacket::new(
            frame.meta().pts,
            frames == 1,
            vec![frames as u8; 4],
        ));
        Ok(())
    }

    fn finish(&mut self, out: &mut dyn FnMut(EncodedPacket)) -> Result<()> {
        self.events.lock().unwrap().finish_packets += 1;
        out(EncodedPacket::new(Duration::ZERO, true, vec![0xff; 2]));
        Ok(())
    }
}

struct EncoderPlatform {
    slow: Duration,
    fail_after: Option<u64>,
    events: Arc<Mutex<EncoderEvents>>,
}

impl Platform for EncoderPlatform {
    fn name(&self) -> &'static str {
        "sim-encoder"
    }
    fn caps(&self) -> PlatformCaps {
        PlatformCaps {
            cpu_encode: true,
            ..SimPlatform.caps()
        }
    }
    fn source(&self, cfg: &SourceConfig) -> Result<Box<dyn FrameSource>> {
        SimPlatform.source(cfg)
    }
    fn preprocessor(&self, cfg: &PreConfig) -> Result<Box<dyn Preprocessor>> {
        SimPlatform.preprocessor(cfg)
    }
    fn engine(&self, cfg: &ModelConfig) -> Result<EngineState> {
        SimPlatform.engine(cfg)
    }
    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        SimPlatform.overlay()
    }
    fn stream(&self, camera_id: u8) -> Result<GpuStream> {
        SimPlatform.stream(camera_id)
    }
    fn encoder(
        &self,
        cfg: &EncoderConfig,
        _width: u32,
        _height: u32,
    ) -> Result<Option<Box<dyn Encoder>>> {
        if cfg.hardware {
            return Ok(None);
        }
        Ok(Some(Box::new(FakeEncoder {
            slow: self.slow,
            fail_after: self.fail_after,
            events: Arc::clone(&self.events),
        })))
    }
}

fn config(frames: u64) -> Config {
    Config::from_toml_str(&format!(
        r#"
platform = "sim"
zero_copy = "relaxed"
[source]
kind = "synthetic"
width = 320
height = 240
fps = 30
paced = false
pool_size = 8
frames = {frames}
[model]
input_width = 64
input_height = 64
[encoder]
enabled = true
hardware = false
codec = "H264"
[stream]
metadata_only = false
[pipeline]
queue_depth = 4
"#
    ))
    .expect("config")
}

fn run(
    platform: &EncoderPlatform,
    frames: u64,
) -> (edge_pipeline::PipelineMetrics, Result<()>, Counts) {
    let counts = Arc::new(Mutex::new(Counts::default()));
    let mut pipeline = Pipeline::build(
        platform,
        config(frames),
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(CountingSink::new(Arc::clone(&counts))),
    )
    .expect("build pipeline");
    let (metrics, result) = pipeline.run(None);
    let counts = *counts.lock().unwrap();
    (metrics, result, counts)
}

#[test]
fn encoder_runs_on_its_own_thread_and_packets_reach_the_sink() {
    let events = Arc::new(Mutex::new(EncoderEvents::default()));
    let platform = EncoderPlatform {
        slow: Duration::from_millis(20),
        fail_after: None,
        events: Arc::clone(&events),
    };
    let (metrics, result, counts) = run(&platform, 30);
    result.expect("run");
    let events = events.lock().unwrap();
    assert_eq!(
        events.thread.as_deref(),
        Some("encode-cam0"),
        "encode must run on the dedicated encoder thread"
    );
    assert!(events.frames > 0);
    assert_eq!(events.finish_packets, 1, "finish must be called once");
    assert!(counts.video_packets > 0, "no packet reached the sink");
    assert_eq!(
        counts.meta_frames, metrics.frames_processed,
        "every processed frame must still emit metadata"
    );
    assert_eq!(
        metrics.frames_processed + metrics.frames_dropped,
        metrics.frames_captured,
        "{}",
        metrics.summary()
    );
    assert!(
        metrics.encode_dropped > 0,
        "a slow encoder with a one-deep hand-off must drop frames: {}",
        metrics.summary()
    );
}

#[test]
fn encoder_error_surfaces_and_does_not_deadlock() {
    let events = Arc::new(Mutex::new(EncoderEvents::default()));
    let platform = EncoderPlatform {
        slow: Duration::from_millis(1),
        fail_after: Some(1),
        events: Arc::clone(&events),
    };
    let started = std::time::Instant::now();
    let (metrics, result, _counts) = run(&platform, 30);
    let err = result.expect_err("encoder failure must surface");
    assert!(err.to_string().contains("encoder boom"), "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "an encoder error must not deadlock teardown"
    );
    assert_eq!(
        events.lock().unwrap().finish_packets,
        0,
        "no flush after error"
    );
    assert!(metrics.frames_processed > 0, "{}", metrics.summary());
}

#[test]
fn encoder_error_stops_an_unbounded_run() {
    let events = Arc::new(Mutex::new(EncoderEvents::default()));
    let platform = EncoderPlatform {
        slow: Duration::from_millis(1),
        fail_after: Some(1),
        events,
    };
    let mut cfg = config(0);
    cfg.source.frames = None;
    cfg.source.paced = false;
    let counts = Arc::new(Mutex::new(Counts::default()));
    let mut pipeline = Pipeline::build(
        &platform,
        cfg,
        Box::new(SimPostProcessor::new(0.25)),
        Box::new(CountingSink::new(Arc::clone(&counts))),
    )
    .expect("build pipeline");
    let started = std::time::Instant::now();
    let (_metrics, result) = pipeline.run(None);
    let err = result.expect_err("an unbounded camera must stop when the encoder dies");
    assert!(err.to_string().contains("encoder boom"), "{err}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "encoder failure must not leave an unbounded run going"
    );
}
