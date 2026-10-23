#[cfg(feature = "cpu-debug")]
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::os::fd::RawFd;
use std::sync::Arc;
use std::time::Duration;

/// Where a frame's pixels live. There is deliberately no `Host` variant:
/// a frame can never be in CPU memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameMemory {
    /// Jetson NvBufSurface (NVMM).
    Nvmm,
    /// NXP / Ambarella / generic V4L2 DMA-BUF.
    DmaBuf,
}

impl std::fmt::Display for FrameMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvmm => f.write_str("NVMM"),
            Self::DmaBuf => f.write_str("DMA-BUF"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layout {
    PitchLinear,
    BlockLinear,
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PitchLinear => f.write_str("pitch-linear"),
            Self::BlockLinear => f.write_str("block-linear"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    /// The v1 format: ISP, CUDA and the encoder all read NV12 natively.
    Nv12,
}

impl PixelFormat {
    /// Bytes for one frame at `width`x`height` (both must be even for NV12).
    pub fn frame_bytes(self, width: u32, height: u32) -> usize {
        match self {
            Self::Nv12 => (width as usize * height as usize * 3) / 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameMeta {
    pub seq: u64,
    /// Sensor timestamp, relative to stream start.
    pub pts: Duration,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub layout: Layout,
    pub camera_id: u8,
    /// Capture dmabuf fd. The encoder asserts it encodes this same fd.
    pub dmabuf_fd: RawFd,
}

/// CPU view of frame pixels. Only reachable behind the `cpu-debug`
/// feature, which release builds never enable.
#[cfg(feature = "cpu-debug")]
pub struct CpuMapping<'a> {
    data: &'a [u8],
}

#[cfg(feature = "cpu-debug")]
impl<'a> CpuMapping<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
    pub fn as_slice(&self) -> &[u8] {
        self.data
    }
}

/// Handle to a hardware frame buffer: exposes ids, never bytes.
/// Dropping the last reference returns it to its preallocated pool.
pub trait HwBuffer: Send + Sync + 'static {
    fn memory(&self) -> FrameMemory;
    /// Every backend must provide a dmabuf fd: it is the lingua franca
    /// for cross-API import (CUDA, V4L2 encoder, ...).
    fn dmabuf_fd(&self) -> RawFd;
    /// Downcast inside its own backend only.
    fn as_any(&self) -> &dyn Any;

    #[cfg(feature = "cpu-debug")]
    fn map_cpu(&self) -> Result<CpuMapping<'_>> {
        Err(crate::error::Error::Unsupported(
            "map_cpu not implemented by this backend".into(),
        ))
    }
}

/// A frame moving through the pipeline. Only this handle ever crosses
/// a stage boundary; pixel bytes never do.
#[derive(Clone)]
pub struct HwFrame {
    pub meta: FrameMeta,
    pub buf: Arc<dyn HwBuffer>,
}

impl HwFrame {
    pub fn new(meta: FrameMeta, buf: Arc<dyn HwBuffer>) -> Self {
        Self { meta, buf }
    }
    pub fn memory(&self) -> FrameMemory {
        self.buf.memory()
    }
}
