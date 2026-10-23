use crate::config::{Codec, EncoderConfig, ModelConfig, Precision, SourceConfig};
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
    pub color: crate::config::InputColor,
    pub colorspace: crate::config::InputColorspace,
    pub normalize: crate::config::InputNormalize,
    /// Letterbox padding value in byte domain.
    pub pad_value: u8,
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

/// A compressed packet handed out by the encoder through a
/// callback. The payload is a shared handle, so fan-out to N
/// sinks clones a refcount, never the bytes; hardware backends
/// will back it with the encoder's output buffer (kept alive by
/// the packet and returned on drop) instead of a heap copy.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    pub pts: Duration,
    pub keyframe: bool,
    pub data: std::sync::Arc<[u8]>,
}

impl EncodedPacket {
    pub fn new(pts: Duration, keyframe: bool, data: impl Into<std::sync::Arc<[u8]>>) -> Self {
        Self {
            pts,
            keyframe,
            data: data.into(),
        }
    }
}

/// Result of polling a source for a frame. Timeout and end of
/// stream are different events and must never be conflated: a
/// pipeline continues on `Timeout` (and counts it) and stops only
/// on `EndOfStream`.
pub enum FramePoll {
    Frame(HwFrame),
    /// No frame was available within the timeout (e.g. the buffer
    /// pool is momentarily exhausted). Counted separately as a
    /// capture timeout; `frames_dropped` is reserved for frames
    /// that were captured and then evicted by backpressure.
    Timeout,
    /// The source will never produce another frame.
    EndOfStream,
}

pub trait FrameSource: Send {
    fn start(&mut self) -> Result<()>;
    /// Hands out a frame from the source's hardware buffer pool;
    /// never allocates in steady state, never copies pixels.
    fn next_frame(&mut self, timeout: Duration) -> Result<FramePoll>;
    fn stop(&mut self) -> Result<()>;
    /// Memory domain this source produces. Checked at build time.
    fn memory(&self) -> FrameMemory;
    /// Layout of the frames this source produces (nvarguscamerasrc
    /// NVMM is block-linear; most file/sim sources pitch-linear).
    /// The builder feeds this to the preprocessor instead of
    /// assuming a layout, and equals `FrameMeta::layout`.
    fn layout(&self) -> Layout {
        Layout::PitchLinear
    }
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
/// `&mut self` so decoders can reuse scratch buffers across
/// frames instead of allocating per frame.
pub trait PostProcessor: Send {
    fn decode(&mut self, outputs: &[Tensor], meta: &FrameMeta, out: &mut Detections) -> Result<()>;
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
    pub precisions: Vec<Precision>,
    /// Accelerator names `model.device` may select ("gpu", "dla0",
    /// "cpu", ...). Empty means the backend does not distinguish.
    pub devices: Vec<String>,
    pub has_hw_encoder: bool,
    /// A CPU-encode path exists (VIC copy out + x264/x265 on
    /// Jetson). Only reachable under `zero_copy = "relaxed"` with
    /// `encoder.hardware = false`; strict mode never uses it.
    pub cpu_encode: bool,
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
