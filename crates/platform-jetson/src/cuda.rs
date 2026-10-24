//! CUDA objects as the core's hooks expect them. Role: Adapter from
//! `trt_sys::{Stream, MappedBuffer}` to `GpuStream` and `TensorBuffer`.
//!
//! One `CudaStream` per camera (D7): pre-process, inference and overlay
//! queue on it in order. A tensor buffer is mapped host memory (D3): the
//! CPU sees `&[f32]`, the GPU a device address bound once at engine load.

use crate::mapped::F32Memory;
use edge_core::{Error, GpuStream, Result};
use std::sync::Arc;
use std::time::Duration;

pub(crate) fn platform_err(e: trt_sys::Error) -> Error {
    Error::Platform(e.to_string())
}

/// The camera's non-blocking CUDA stream, inside a `GpuStream`.
pub struct CudaStream {
    inner: trt_sys::Stream,
}

impl CudaStream {
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: trt_sys::Stream::new().map_err(platform_err)?,
        })
    }

    /// The wrapped stream, for the stages that queue work on it.
    pub fn inner(&self) -> &trt_sys::Stream {
        &self.inner
    }

    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize().map_err(platform_err)
    }
}

/// Two CUDA events around the GPU work of one inference (T10). Created once
/// with the engine; each frame only records them again.
pub struct GpuTimer {
    start: trt_sys::Event,
    end: trt_sys::Event,
}

impl GpuTimer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            start: trt_sys::Event::new().map_err(platform_err)?,
            end: trt_sys::Event::new().map_err(platform_err)?,
        })
    }

    pub fn begin(&self, stream: &trt_sys::Stream) -> Result<()> {
        self.start.record(stream).map_err(platform_err)
    }

    pub fn finish(&self, stream: &trt_sys::Stream) -> Result<()> {
        self.end.record(stream).map_err(platform_err)
    }

    /// Between the two records; call after the stream synchronized.
    pub fn elapsed(&self) -> Result<Duration> {
        let ms = self
            .end
            .elapsed_ms_since(&self.start)
            .map_err(platform_err)?;
        Ok(Duration::from_secs_f64(f64::from(ms.max(0.0)) / 1000.0))
    }
}

/// A new stream wrapped for `Platform::stream`.
pub fn new_gpu_stream() -> Result<GpuStream> {
    Ok(GpuStream::from_backend(Arc::new(CudaStream::new()?)))
}

pub use crate::mapped::MappedTensor;

impl F32Memory for trt_sys::MappedBuffer {
    fn as_slice(&self) -> &[f32] {
        trt_sys::MappedBuffer::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        trt_sys::MappedBuffer::as_mut_slice(self)
    }
}
