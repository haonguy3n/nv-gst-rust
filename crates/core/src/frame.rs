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
    /// A backend-defined domain (e.g. Ambarella IAV/DSP shared
    /// memory). Naming it here must never require a core release:
    /// a new platform brings its own tag, and equality is exact.
    Other(&'static str),
}

impl std::fmt::Display for FrameMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvmm => f.write_str("NVMM"),
            Self::DmaBuf => f.write_str("DMA-BUF"),
            Self::Other(name) => f.write_str(name),
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

/// Geometry of one image plane inside a frame's single dmabuf
/// allocation. Offsets and pitches are **bytes**, never inferred
/// from the visible width: a padded NVMM surface is a different
/// allocation from a tight NV12 frame, and the layout in
/// `FrameMeta` is authoritative (review P1-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaneDesc {
    /// Byte offset from the start of the dmabuf.
    pub offset: usize,
    /// Row stride in bytes (>= `row_bytes`).
    pub pitch: usize,
    /// Meaningful bytes per row: NV12 luma and interleaved UV rows
    /// are both `width` bytes; sub-sampled planes are described in
    /// bytes, not samples.
    pub row_bytes: usize,
    /// Visible rows in this plane.
    pub rows: u32,
}

/// Up to three planes (NV12 uses two; RGB-family formats would use
/// three). A plane-aware descriptor is what lets a backend expose
/// block-linear/padded memory honestly instead of pretending every
/// `map_cpu()` is a tight row-major image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PlaneLayout {
    planes: [Option<PlaneDesc>; 3],
}

impl PlaneLayout {
    /// Tight NV12: full-size Y plane, then one interleaved UV
    /// plane with half the rows and the same byte pitch.
    pub fn nv12(width: u32, height: u32) -> Self {
        let w = width as usize;
        let h = height as usize;
        Self {
            planes: [
                Some(PlaneDesc {
                    offset: 0,
                    pitch: w,
                    row_bytes: w,
                    rows: height,
                }),
                Some(PlaneDesc {
                    offset: w * h,
                    pitch: w,
                    row_bytes: w,
                    rows: (height / 2).max(1),
                }),
                None,
            ],
        }
    }

    /// Build from an explicit plane list (a backend reading
    /// `NvBufSurface` plane params uses this).
    pub fn from_planes(planes: &[PlaneDesc]) -> Self {
        let mut out = [None; 3];
        for (slot, plane) in out.iter_mut().zip(planes.iter().take(3)) {
            *slot = Some(*plane);
        }
        Self { planes: out }
    }

    pub fn planes(&self) -> impl Iterator<Item = &PlaneDesc> {
        self.planes.iter().flatten()
    }

    pub fn len(&self) -> usize {
        self.planes.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes spanned by the allocation: the furthest plane end.
    pub fn byte_len(&self) -> usize {
        self.planes()
            .map(|p| p.offset + p.pitch * p.rows as usize)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    /// The v1 format: ISP, CUDA and the encoder all read NV12 natively.
    Nv12,
}

impl PixelFormat {
    /// Plane geometry for one frame at `width`x`height`. Tight
    /// buffers only; a backend with padded/block-linear memory
    /// overrides the descriptor from its surface parameters.
    pub fn plane_layout(self, width: u32, height: u32) -> PlaneLayout {
        match self {
            Self::Nv12 => PlaneLayout::nv12(width, height),
        }
    }

    /// Bytes for one tight frame at `width`x`height` (both must be
    /// even for NV12).
    pub fn frame_bytes(self, width: u32, height: u32) -> usize {
        self.plane_layout(width, height).byte_len()
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
    /// Authoritative plane geometry for `dmabuf_fd`: every consumer
    /// reads offsets/pitches from here rather than assuming the
    /// frame is tight.
    pub planes: PlaneLayout,
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
    /// for cross-API import (CUDA, V4L2 encoder, ...). Contract: the
    /// fd names ONE contiguous allocation; planes and their pitches
    /// are described by `FrameMeta::planes` as byte offsets into it.
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
    // Private: the fd in here is part of the same-fd contract, so
    // no stage may rewrite it after the source hands the frame out.
    meta: FrameMeta,
    pub buf: Arc<dyn HwBuffer>,
    /// Monotonic time the source produced the frame. Used for the
    /// end-to-end age metric (queue + stages), not for joining to
    /// video (that uses `meta.pts`).
    received_at: std::time::Instant,
}

impl HwFrame {
    pub fn new(meta: FrameMeta, buf: Arc<dyn HwBuffer>) -> Self {
        Self {
            meta,
            buf,
            received_at: std::time::Instant::now(),
        }
    }

    pub fn meta(&self) -> &FrameMeta {
        &self.meta
    }
    pub fn memory(&self) -> FrameMemory {
        self.buf.memory()
    }
    /// When the source handed this frame out. The runner measures
    /// `received_at().elapsed()` after the sink accepts the frame,
    /// so queueing and every stage are included in the latency.
    pub fn received_at(&self) -> std::time::Instant {
        self.received_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_plane_layout_is_tight_and_byte_accurate() {
        let layout = PixelFormat::Nv12.plane_layout(640, 480);
        let planes: Vec<_> = layout.planes().copied().collect();
        assert_eq!(planes.len(), 2);
        assert_eq!(
            (planes[0].offset, planes[0].pitch, planes[0].rows),
            (0, 640, 480)
        );
        assert_eq!(
            (planes[1].offset, planes[1].pitch, planes[1].rows),
            (640 * 480, 640, 240)
        );
        assert_eq!(layout.byte_len(), 640 * 480 * 3 / 2);
        assert_eq!(PixelFormat::Nv12.frame_bytes(640, 480), layout.byte_len());
    }

    #[test]
    fn explicit_plane_layout_keeps_padded_pitch() {
        // A padded NVMM-style surface: pitch 704 for a 640-wide plane.
        let layout = PlaneLayout::from_planes(&[
            PlaneDesc {
                offset: 0,
                pitch: 704,
                row_bytes: 640,
                rows: 480,
            },
            PlaneDesc {
                offset: 704 * 480,
                pitch: 704,
                row_bytes: 640,
                rows: 240,
            },
        ]);
        assert_eq!(layout.planes().next().unwrap().pitch, 704);
        assert_eq!(layout.byte_len(), 704 * 480 + 704 * 240);
    }
}
