//! P0-3: the encoder decision matrix. A fake platform with
//! switchable caps pins down which (zero_copy, hardware, caps)
//! combinations build, refuse, or take the CPU opt-in.

use edge_core::{
    Codec, Config, Detections, EncodedPacket, Encoder, EncoderConfig, FrameMemory, FrameMeta,
    FramePoll, FrameSource, GpuStream, HwFrame, InferenceEngine, ModelConfig, Overlay, PixelFormat,
    Platform, PlatformCaps, PostProcessor, PreConfig, Preprocessor, Result, Sink, SourceConfig,
    Tensor,
};
use edge_pipeline::Pipeline;
use std::time::Duration;

struct FakePlatform {
    hw: bool,
    cpu: bool,
}

struct EosSource;
impl FrameSource for EosSource {
    fn start(&mut self) -> Result<()> {
        Ok(())
    }
    fn next_frame(&mut self, _t: Duration) -> Result<FramePoll> {
        Ok(FramePoll::EndOfStream)
    }
    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
    fn memory(&self) -> FrameMemory {
        FrameMemory::Nvmm
    }
}

struct NoopPre;
impl Preprocessor for NoopPre {
    fn run(&mut self, _f: &HwFrame, _d: &mut Tensor, _s: &GpuStream) -> Result<()> {
        Ok(())
    }
}

struct NoopEngine {
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
}
impl InferenceEngine for NoopEngine {
    fn input_bindings(&mut self) -> &mut [Tensor] {
        &mut self.inputs
    }
    fn output_bindings(&self) -> &[Tensor] {
        &self.outputs
    }
    fn infer(&mut self, _s: &GpuStream) -> Result<()> {
        Ok(())
    }
}

struct NoopPost;
impl PostProcessor for NoopPost {
    fn decode(&mut self, _o: &[Tensor], m: &FrameMeta, out: &mut Detections) -> Result<()> {
        out.clear(m.seq);
        Ok(())
    }
}

struct NoopOverlay;
impl Overlay for NoopOverlay {
    fn draw(&mut self, _f: &HwFrame, _d: &Detections, _s: &GpuStream) -> Result<()> {
        Ok(())
    }
}

struct NoopEncoder;
impl Encoder for NoopEncoder {
    fn encode(&mut self, _f: HwFrame, _o: &mut dyn FnMut(EncodedPacket)) -> Result<()> {
        Ok(())
    }
}

struct NoopSink;
impl Sink for NoopSink {
    fn push_video(&mut self, _p: EncodedPacket) -> Result<()> {
        Ok(())
    }
}

impl Platform for FakePlatform {
    fn name(&self) -> &'static str {
        "fake"
    }
    fn caps(&self) -> PlatformCaps {
        PlatformCaps {
            frame_memory: FrameMemory::Nvmm,
            formats: vec![PixelFormat::Nv12],
            max_width: 3840,
            max_height: 2160,
            codecs: if self.hw {
                vec![Codec::H264, Codec::H265]
            } else {
                Vec::new()
            },
            precisions: Vec::new(),
            devices: Vec::new(),
            has_hw_encoder: self.hw,
            cpu_encode: self.cpu,
            accelerators: Vec::new(),
        }
    }
    fn source(&self, _c: &SourceConfig) -> Result<Box<dyn FrameSource>> {
        Ok(Box::new(EosSource))
    }
    fn preprocessor(&self, _c: &PreConfig) -> Result<Box<dyn Preprocessor>> {
        Ok(Box::new(NoopPre))
    }
    fn engine(&self, _c: &ModelConfig) -> Result<Box<dyn InferenceEngine>> {
        Ok(Box::new(NoopEngine {
            inputs: Vec::new(),
            outputs: Vec::new(),
        }))
    }
    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        Ok(Box::new(NoopOverlay))
    }
    fn encoder(&self, cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>> {
        let available = if cfg.hardware { self.hw } else { self.cpu };
        Ok(available.then(|| Box::new(NoopEncoder) as Box<dyn Encoder>))
    }
}

fn config(zero_copy: &str, hardware: bool) -> Config {
    Config::from_toml_str(&format!(
        r#"
platform = "fake"
zero_copy = "{zero_copy}"
[source]
kind = "argus"
width = 1440
height = 1080
fps = 30
pool_size = 8
[model]
input_width = 416
input_height = 416
[encoder]
enabled = true
hardware = {hardware}
[stream]
metadata_only = false
"#
    ))
    .expect("config")
}

fn build(platform: &FakePlatform, cfg: Config) -> Result<Pipeline> {
    Pipeline::build(platform, cfg, Box::new(NoopPost), Box::new(NoopSink))
}

#[test]
fn strict_nano_refuses_cpu_encode() {
    let nano = FakePlatform {
        hw: false,
        cpu: true,
    };
    // Config validation rejects strict + hardware=false first.
    assert!(Config::from_toml_str(
        "platform=\"x\"\nzero_copy=\"strict\"\n[source]\nkind=\"a\"\nwidth=64\nheight=64\nfps=30\n[model]\ninput_width=32\ninput_height=32\n[encoder]\nenabled=true\nhardware=false\n"
    )
    .is_err());
    let _ = &nano;
}

#[test]
fn relaxed_nano_cpu_opt_in_builds() {
    let nano = FakePlatform {
        hw: false,
        cpu: true,
    };
    let p = build(&nano, config("relaxed", false)).expect("relaxed CPU opt-in must build");
    assert!(!p.metadata_only());
}

#[test]
fn relaxed_without_cpu_path_refuses() {
    let bare = FakePlatform {
        hw: false,
        cpu: false,
    };
    match build(&bare, config("relaxed", false)) {
        Ok(_) => panic!("must refuse when the platform has no CPU encoder"),
        Err(e) => assert!(e.to_string().contains("no CPU encoder"), "{e}"),
    }
}

#[test]
fn nx_hardware_path_builds_and_nano_hardware_request_refuses() {
    let nx = FakePlatform {
        hw: true,
        cpu: true,
    };
    assert!(build(&nx, config("strict", true)).is_ok());
    let nano = FakePlatform {
        hw: false,
        cpu: true,
    };
    match build(&nano, config("relaxed", true)) {
        Ok(_) => panic!("hardware=true on a Nano must refuse"),
        Err(e) => assert!(e.to_string().contains("no hardware encoder"), "{e}"),
    }
}
