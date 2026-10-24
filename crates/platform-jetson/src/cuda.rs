//! CUDA objects as the core's hooks expect them. Role: Adapter from
//! `trt_sys::{Stream, MappedBuffer}` to `GpuStream` and `TensorBuffer`.
//!
//! One `CudaStream` per camera (D7): pre-process, inference and overlay
//! queue on it in order. A tensor buffer is mapped host memory (D3): the
//! CPU sees `&[f32]`, the GPU a device address bound once at engine load.

use edge_core::{Error, GpuStream, Result, TensorBuffer};
use std::any::Any;
use std::sync::Arc;

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

/// A new stream wrapped for `Platform::stream`.
pub fn new_gpu_stream() -> Result<GpuStream> {
    Ok(GpuStream::from_backend(Arc::new(CudaStream::new()?)))
}

/// Mapped host memory as a tensor's backing store.
pub struct MappedTensor {
    buffer: trt_sys::MappedBuffer,
}

impl MappedTensor {
    pub fn new(buffer: trt_sys::MappedBuffer) -> Self {
        Self { buffer }
    }
}

impl TensorBuffer for MappedTensor {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_f32_slice(&self) -> Option<&[f32]> {
        Some(self.buffer.as_slice())
    }

    fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        Some(self.buffer.as_mut_slice())
    }
}
