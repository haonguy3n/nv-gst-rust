use crate::config::{Codec, EncoderConfig, ModelConfig, Precision, SourceConfig};
use crate::detection::Detections;
use crate::error::Result;
use crate::frame::{FrameMemory, FrameMeta, HwFrame, Layout, PixelFormat};
use crate::tensor::Tensor;
use std::any::Any;
use std::sync::Arc;
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

/// Opaque handle to one ordered GPU work queue. One stream per
/// camera: pre-process, inference and overlay queue on it in order.
///
/// Created by the platform (`Platform::stream`), never by the
/// runner: on Jetson it wraps the camera's `cudaStream_t` (inside
/// an `Arc<dyn Any>` the owning backend downcasts), and in the
/// simulator it is a no-op marker.
#[derive(Clone, Default)]
pub struct GpuStream {
    inner: Option<Arc<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for GpuStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuStream")
            .field("backend", &self.inner.as_ref().map(|_| "<opaque>"))
            .finish()
    }
}

impl GpuStream {
    /// No GPU work queue (simulator, tests).
    pub fn noop() -> Self {
        Self { inner: None }
    }

    /// Wrap a backend stream handle (e.g. the Jetson `CudaStream`).
    pub fn from_backend<T: Any + Send + Sync>(handle: Arc<T>) -> Self {
        Self {
            inner: Some(handle as Arc<dyn Any + Send + Sync>),
        }
    }

    /// The backend handle, for the backend that created it.
    pub fn backend<T: Any>(&self) -> Option<&T> {
        self.inner.as_ref().and_then(|h| h.downcast_ref::<T>())
    }

    pub fn is_noop(&self) -> bool {
        self.inner.is_none()
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
    fn layout(&self) -> Layout;
    /// Buffers currently back in the pool, when the backend can
    /// report it. Conformance uses this to prove no frame leak
    /// across a start/stop cycle; `None` means "not reportable".
    fn pool_available(&self) -> Option<usize> {
        None
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

/// An engine still being prepared (TensorRT plan build on first
/// boot can take minutes). The runner polls this once per frame;
/// until it yields the engine, frames flow with empty detections
/// so video never waits for inference (round 3, L3).
pub trait EngineWarmup: Send {
    /// `Ok(Some(engine))` when ready, `Ok(None)` while still
    /// warming, `Err` when preparation failed for good.
    fn poll_ready(&mut self) -> Result<Option<Box<dyn InferenceEngine>>>;
}

/// What `Platform::engine` hands the builder.
pub enum EngineState {
    Ready(Box<dyn InferenceEngine>),
    Warming(Box<dyn EngineWarmup>),
}

impl EngineState {
    /// Block until the engine is ready (tests, conformance,
    /// one-shot tools). The runner does not use this: it polls
    /// per frame so video starts immediately.
    pub fn wait_ready(self, timeout: Duration) -> Result<Box<dyn InferenceEngine>> {
        match self {
            Self::Ready(engine) => Ok(engine),
            Self::Warming(mut warmup) => {
                let deadline = std::time::Instant::now() + timeout;
                loop {
                    if let Some(engine) = warmup.poll_ready()? {
                        return Ok(engine);
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err(crate::error::Error::Platform(format!(
                            "engine still warming after {timeout:?}"
                        )));
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
    }
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
    /// Flush buffered output. The runner calls this on every
    /// exit path so flush errors surface instead of dying in Drop.
    fn finish(&mut self) -> Result<()> {
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
}

/// One per platform crate; the binary picks it from config.
pub trait Platform {
    fn name(&self) -> &'static str;
    fn caps(&self) -> PlatformCaps;
    fn source(&self, cfg: &SourceConfig) -> Result<Box<dyn FrameSource>>;
    fn preprocessor(&self, cfg: &PreConfig) -> Result<Box<dyn Preprocessor>>;
    /// Ready now, or warming in the background (plan build). The
    /// runner handles both; see `EngineWarmup`.
    fn engine(&self, cfg: &ModelConfig) -> Result<EngineState>;
    fn overlay(&self) -> Result<Box<dyn Overlay>>;
    /// `None` when the SoC has no HW encoder (Orin Nano): the pipeline
    /// then streams metadata only.
    fn encoder(&self, cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>>;
    /// The ordered GPU work queue for one camera. Backends with a
    /// real device (Jetson CUDA) override this; the default is a
    /// no-op stream for CPU-only backends.
    fn stream(&self, _camera_id: u8) -> Result<GpuStream> {
        Ok(GpuStream::noop())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeCudaStream(u64);

    #[test]
    fn gpu_stream_wraps_and_downcasts_backend_handle() {
        let stream = GpuStream::from_backend(Arc::new(FakeCudaStream(7)));
        assert!(!stream.is_noop());
        assert_eq!(stream.backend::<FakeCudaStream>().map(|s| s.0), Some(7));
        assert!(stream.backend::<u32>().is_none());

        let noop = GpuStream::noop();
        assert!(noop.is_noop());
        assert!(noop.backend::<FakeCudaStream>().is_none());
    }
}
