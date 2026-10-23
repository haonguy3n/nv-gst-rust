use crate::config::{Codec, EncoderConfig, ModelConfig, SourceConfig};
use crate::detection::Detections;
use crate::error::Result;
use crate::frame::{FrameMemory, FrameMeta, HwFrame, Layout, PixelFormat};
use crate::tensor::Tensor;
use std::time::Duration;

/// Pre-processor configuration, derived from the model input by the builder.
#[derive(Debug, Clone)]
pub struct PreConfig {
    pub src_width: u32,
    pub src_height: u32,
    pub dst_width: u32,
    pub dst_height: u32,
    pub layout: Layout,
}

/// Opaque handle to one ordered GPU work queue. One stream per camera:
/// pre-process, inference and overlay queue on it in order.
#[derive(Debug, Clone, Copy)]
pub struct GpuStream {
    id: u32,
}

impl GpuStream {
    pub fn new(id: u32) -> Self {
        Self { id }
    }
    pub fn id(&self) -> u32 {
        self.id
    }
}

/// A compressed packet handed out by the encoder through a callback,
/// so the encoder never allocates a `Vec` per packet in product
/// backends (the bytes wrap the V4L2 capture-plane mmap there).
#[derive(Debug)]
pub struct EncodedPacket {
    pub pts: Duration,
    pub keyframe: bool,
    pub data: Vec<u8>,
}

pub trait FrameSource: Send {
    fn start(&mut self) -> Result<()>;
    /// Hands out a frame from the source's hardware buffer pool;
    /// never allocates in steady state, never copies pixels.
    /// `Ok(None)` means timeout or end of stream.
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<HwFrame>>;
    fn stop(&mut self) -> Result<()>;
    /// Memory domain this source produces. Checked at build time.
    fn memory(&self) -> FrameMemory;
}

pub trait Preprocessor: Send {
    /// Reads the frame in place on the GPU, writes straight into the
    /// engine's input binding.
    fn run(&mut self, frame: &HwFrame, dst: &mut Tensor, stream: &GpuStream) -> Result<()>;
}

pub trait InferenceEngine: Send {
    fn input_bindings(&mut self) -> &mut [Tensor];
    /// Host-visible small tensors, read in place by post-processing.
    fn output_bindings(&self) -> &[Tensor];
    fn infer(&mut self, stream: &GpuStream) -> Result<()>;
}

/// Platform-neutral; reads only the small output tensors.
pub trait PostProcessor: Send {
    fn decode(&self, outputs: &[Tensor], meta: &FrameMeta, out: &mut Detections) -> Result<()>;
}

pub trait Overlay: Send {
    /// Draws boxes into the frame in place (CUDA on Jetson).
    fn draw(&mut self, frame: &HwFrame, det: &Detections, stream: &GpuStream) -> Result<()>;
}

pub trait Encoder: Send {
    /// Imports the frame's dmabuf into the HW encoder; owns the frame
    /// until the hardware releases it, then dropping it returns the
    /// buffer to the capture pool.
    fn encode(&mut self, frame: HwFrame, out: &mut dyn FnMut(EncodedPacket)) -> Result<()>;
}

/// RTSP, WebRTC, file, MQTT metadata.
pub trait Sink: Send {
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()>;
    fn push_meta(&mut self, det: &Detections) -> Result<()> {
        let _ = det;
        Ok(())
    }
}

/// What a backend can do. The builder validates config against this
/// instead of the core ever naming a vendor.
#[derive(Debug, Clone)]
pub struct PlatformCaps {
    pub frame_memory: FrameMemory,
    pub formats: Vec<PixelFormat>,
    pub max_width: u32,
    pub max_height: u32,
    pub codecs: Vec<Codec>,
    pub has_hw_encoder: bool,
    /// e.g. `["gpu", "dla0", "dla1"]` on Orin NX, `["cpu-sim"]` in sim.
    pub accelerators: Vec<String>,
}

/// One per platform crate; the binary picks it from config.
pub trait Platform {
    fn name(&self) -> &'static str;
    fn caps(&self) -> PlatformCaps;
    fn source(&self, cfg: &SourceConfig) -> Result<Box<dyn FrameSource>>;
    fn preprocessor(&self, cfg: &PreConfig) -> Result<Box<dyn Preprocessor>>;
    fn engine(&self, cfg: &ModelConfig) -> Result<Box<dyn InferenceEngine>>;
    fn overlay(&self) -> Result<Box<dyn Overlay>>;
    /// `None` when the SoC has no HW encoder (Orin Nano): the pipeline
    /// then streams metadata only.
    fn encoder(&self, cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>>;
}
