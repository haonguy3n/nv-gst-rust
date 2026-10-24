//! P0-3: the encoder decision matrix. A fake platform with
//! switchable caps pins down which (zero_copy, hardware, caps)
//! combinations build, refuse, or take the CPU opt-in.

use edge_core::{
    Codec, Config, DType, Detections, EncodedPacket, Encoder, EncoderConfig, EngineState,
    FrameMemory, FrameMeta, FramePoll, FrameSource, GpuStream, HwFrame, InferenceEngine, Layout,
    ModelConfig, Overlay, PixelFormat, Platform, PlatformCaps, PostProcessor, PreConfig,
    Preprocessor, Result, Sink, SourceConfig, Tensor, TensorBuffer, TensorSpec,
};
use edge_pipeline::Pipeline;
use std::time::Duration;

struct FakePlatform {
    hw: bool,
    cpu: bool,
    /// Declare an engine input that disagrees with the config.
    bad_input: bool,
    /// Declare a normal F32 engine input (TensorRT FP16 plans still
    /// expose F32 I/O bindings).
    f32_input: bool,
    /// Declare an F16 input binding, which no preprocessor can write yet.
    f16_input: bool,
    /// Omit every output binding.
    empty_output: bool,
}

struct StubBuffer;
impl TensorBuffer for StubBuffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
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
    fn layout(&self) -> Layout {
        Layout::PitchLinear
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
        out.clear(m);
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
        }
    }
    fn source(&self, _c: &SourceConfig) -> Result<Box<dyn FrameSource>> {
        Ok(Box::new(EosSource))
    }
    fn preprocessor(&self, _c: &PreConfig) -> Result<Box<dyn Preprocessor>> {
        Ok(Box::new(NoopPre))
    }
    fn engine(&self, _c: &ModelConfig) -> Result<EngineState> {
        let inputs = if self.bad_input {
            vec![Tensor::new(
                TensorSpec::new("images", vec![1, 3, 999, 999], DType::F32),
                Box::new(StubBuffer),
            )]
        } else if self.f32_input {
            vec![Tensor::new(
                TensorSpec::new("images", vec![1, 3, 416, 416], DType::F32),
                Box::new(StubBuffer),
            )]
        } else if self.f16_input {
            vec![Tensor::new(
                TensorSpec::new("images", vec![1, 3, 416, 416], DType::F16),
                Box::new(StubBuffer),
            )]
        } else {
            vec![Tensor::new(
                TensorSpec::new("images", vec![1, 3, 416, 416], DType::F32),
                Box::new(StubBuffer),
            )]
        };
        let outputs = if self.empty_output {
            Vec::new()
        } else {
            vec![Tensor::new(
                TensorSpec::new("detections", vec![1, 1, 1], DType::F32),
                Box::new(StubBuffer),
            )]
        };
        Ok(EngineState::Ready(Box::new(NoopEngine { inputs, outputs })))
    }
    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        Ok(Box::new(NoopOverlay))
    }
    fn stream(&self, _camera_id: u8) -> Result<GpuStream> {
        Ok(GpuStream::noop())
    }

    fn encoder(
        &self,
        cfg: &EncoderConfig,
        _width: u32,
        _height: u32,
    ) -> Result<Option<Box<dyn Encoder>>> {
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
        bad_input: false,
        f32_input: false,
        f16_input: false,
        empty_output: false,
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
        bad_input: false,
        f32_input: false,
        f16_input: false,
        empty_output: false,
    };
    let p = build(&nano, config("relaxed", false)).expect("relaxed CPU opt-in must build");
    assert!(!p.metadata_only());
}

#[test]
fn relaxed_without_cpu_path_refuses() {
    let bare = FakePlatform {
        hw: false,
        cpu: false,
        bad_input: false,
        f32_input: false,
        f16_input: false,
        empty_output: false,
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
        bad_input: false,
        f32_input: false,
        f16_input: false,
        empty_output: false,
    };
    assert!(build(&nx, config("strict", true)).is_ok());
    let nano = FakePlatform {
        hw: false,
        cpu: true,
        bad_input: false,
        f32_input: false,
        f16_input: false,
        empty_output: false,
    };
    match build(&nano, config("relaxed", true)) {
        Ok(_) => panic!("hardware=true on a Nano must refuse"),
        Err(e) => assert!(e.to_string().contains("no hardware encoder"), "{e}"),
    }
}

/// Item 13: an engine whose declared input disagrees with the
/// config is refused at build time.
#[test]
fn builder_refuses_engine_input_shape_mismatch() {
    let platform = FakePlatform {
        hw: true,
        cpu: false,
        bad_input: true,
        f32_input: false,
        f16_input: false,
        empty_output: false,
    };
    match build(&platform, config("strict", true)) {
        Ok(_) => panic!("mismatched engine input must not build"),
        Err(e) => assert!(
            e.to_string().contains("engine"),
            "error must name the stage: {e}"
        ),
    }
}

/// N6: `model.precision` chooses layer execution, not the I/O dtype.
/// A plan built with FP16 still exposes F32 input/output bindings, so
/// the builder must not demand `F16` for an FP16 config (it used to).
#[test]
fn fp16_config_accepts_an_f32_engine_input_binding() {
    let platform = FakePlatform {
        hw: true,
        cpu: false,
        bad_input: false,
        f32_input: true,
        f16_input: false,
        empty_output: false,
    };
    let cfg = Config::from_toml_str(
        r#"
platform = "fake"
zero_copy = "strict"
[source]
kind = "argus"
width = 1440
height = 1080
fps = 30
pool_size = 8
[model]
input_width = 416
input_height = 416
precision = "fp16"
[encoder]
enabled = true
hardware = true
[stream]
metadata_only = false
"#,
    )
    .expect("config");
    assert!(
        build(&platform, cfg).is_ok(),
        "FP16 execution with F32 bindings must build"
    );
}

/// FP16 input bindings are not writable by any preprocessor yet, so
/// the builder must refuse them instead of accepting a config that
/// will fail at the first frame.
#[test]
fn f16_engine_input_is_rejected_until_a_preprocessor_writes_it() {
    let platform = FakePlatform {
        hw: true,
        cpu: false,
        bad_input: false,
        f32_input: false,
        f16_input: true,
        empty_output: false,
    };
    match build(&platform, config("strict", true)) {
        Ok(_) => panic!("F16 input must be refused"),
        Err(e) => assert!(e.to_string().contains("F32"), "{e}"),
    }
}

#[test]
fn builder_refuses_an_engine_without_outputs() {
    let platform = FakePlatform {
        hw: true,
        cpu: false,
        bad_input: false,
        f32_input: false,
        f16_input: false,
        empty_output: true,
    };
    match build(&platform, config("strict", true)) {
        Ok(_) => panic!("an engine with no outputs must not build"),
        Err(e) => assert!(e.to_string().contains("output binding"), "{e}"),
    }
}

/// N11/E6: with an explicit Argus pool, the builder refuses a live
/// in-flight budget that would starve the sensor.
#[test]
fn builder_refuses_an_in_flight_budget_larger_than_the_argus_pool() {
    let platform = FakePlatform {
        hw: true,
        cpu: false,
        bad_input: false,
        f32_input: false,
        f16_input: false,
        empty_output: false,
    };
    let mut cfg = config("strict", true);
    cfg.pipeline.queue_depth = 4;
    cfg.source.argus_pool = Some(7); // needs 10
    match build(&platform, cfg) {
        Ok(_) => panic!("too-small Argus pool must be refused"),
        Err(e) => assert!(e.to_string().contains("in-flight"), "{e}"),
    }
}
