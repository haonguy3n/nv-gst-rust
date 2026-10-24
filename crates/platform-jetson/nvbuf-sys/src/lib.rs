//! Read-only view of NVIDIA's `NvBufSurface`.
//!
//! This crate holds the only `unsafe` needed to read the surface that
//! `nvarguscamerasrc` puts into an NVMM `GstBuffer`: `gst_buffer_map`
//! returns an `NvBufSurface*` in the mapped bytes, not pixels. The
//! definitions are hand-written `#[repr(C)]` mirrors of the r39.2
//! headers (the field order and padding are checked by tests); they do
//! not link against `libnvbufsurface`.
//!
//! The `surface`, `ffi` externs and `OwnedSurface` half (create, map,
//! VIC transform) calls `libnvbufsurface`/`libnvbufsurftransform`; it
//! exists only with the `link` feature (device builds). The pure
//! half (layouts, constants, `planar`) always builds and is tested
//! on the PC.
//!
//! Reading a camera frame's surface below this line: nothing reads
//! plane pixels of the capture frame. It extracts the dmabuf fd, layout,
//! and plane pitch/offset/psize, which is what `FrameMeta::planes`
//! needs. Block-linear bytes are left for VIC; the CPU must never map
//! or copy them.

#![deny(unsafe_op_in_unsafe_fn)]

mod error;
mod ffi;
pub mod planar;
#[cfg(feature = "link")]
mod surface;

pub use error::NvbufError;
pub use ffi::{ColorFormat, Filter, Rect};
#[cfg(feature = "link")]
pub use surface::{transform_letterbox, EglImage, MappedPlane, OwnedSurface, SurfaceRef};

use std::ffi::c_void;
use std::fmt;

/// `NVBUF_MAX_PLANES` in `nvbufsurface.h`.
pub const NVBUF_MAX_PLANES: usize = 4;

/// `STRUCTURE_PADDING` in `nvbufsurface.h`.
const STRUCTURE_PADDING: usize = 4;

/// `NvBufSurfaceLayout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceLayout {
    Pitch,
    BlockLinear,
}

/// One plane's geometry, in bytes (as the driver reports it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneInfo {
    pub offset: usize,
    pub pitch: usize,
    pub row_bytes: usize,
    pub rows: u32,
    pub size: usize,
}

/// The fields the pipeline needs from one `NvBufSurface`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceInfo {
    pub width: u32,
    pub height: u32,
    pub layout: SurfaceLayout,
    /// dmabuf fd from `surfaceList[0].bufferDesc`.
    pub fd: i32,
    pub planes: [Option<PlaneInfo>; NVBUF_MAX_PLANES],
}

impl SurfaceInfo {
    pub fn planes(&self) -> impl Iterator<Item = &PlaneInfo> {
        self.planes.iter().flatten()
    }

    pub fn plane_count(&self) -> usize {
        self.planes.iter().flatten().count()
    }

    /// Bytes spanned by the planes, from the driver's psize/pitch.
    pub fn byte_len(&self) -> usize {
        self.planes()
            .map(|p| p.offset + p.size.max(p.pitch * p.rows as usize))
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurfaceError {
    /// The mapped size is not exactly one `NvBufSurface`.
    WrongSize { got: usize, need: usize },
    /// The surface has no batches or a null `surfaceList`.
    Empty,
    /// `surfaceList[0]` is not usable.
    BadParams,
    /// Unsupported plane count/layout.
    Unsupported(String),
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongSize { got, need } => write!(
                f,
                "mapped NVMM buffer is {got} bytes, expected exactly {need} for NvBufSurface"
            ),
            Self::Empty => f.write_str("NvBufSurface has no batches"),
            Self::BadParams => f.write_str("NvBufSurface surfaceList[0] is null"),
            Self::Unsupported(what) => write!(f, "unsupported NvBufSurface: {what}"),
        }
    }
}

impl std::error::Error for SurfaceError {}

// The C structs below must stay in sync with nvbufsurface.h r39.2.
// They are only read, never constructed for the driver.

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvBufSurfacePlaneParams {
    num_planes: u32,
    width: [u32; NVBUF_MAX_PLANES],
    height: [u32; NVBUF_MAX_PLANES],
    pitch: [u32; NVBUF_MAX_PLANES],
    offset: [u32; NVBUF_MAX_PLANES],
    psize: [u32; NVBUF_MAX_PLANES],
    bytes_per_pix: [u32; NVBUF_MAX_PLANES],
    _reserved: [*mut c_void; STRUCTURE_PADDING * NVBUF_MAX_PLANES],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvBufSurfaceMappedAddr {
    addr: [*mut c_void; NVBUF_MAX_PLANES],
    egl_image: *mut c_void,
    nvmm_ptr: *mut c_void,
    cuda_ptr: *mut c_void,
    _reserved: [*mut c_void; STRUCTURE_PADDING],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvBufSurfaceParams {
    width: u32,
    height: u32,
    pitch: u32,
    color_format: i32,
    layout: i32,
    buffer_desc: u64,
    data_size: u32,
    data_ptr: *mut c_void,
    plane_params: NvBufSurfacePlaneParams,
    mapped_addr: NvBufSurfaceMappedAddr,
    paramex: *mut c_void,
    cuda_buffer: *mut c_void,
    _reserved: [*mut c_void; STRUCTURE_PADDING],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct NvBufSurface {
    gpu_id: u32,
    batch_size: u32,
    num_filled: u32,
    is_contiguous: bool,
    mem_type: i32,
    surface_list: *mut NvBufSurfaceParams,
    is_imported_buf: bool,
    _reserved: [*mut c_void; STRUCTURE_PADDING],
}

/// Read one `NvBufSurface` from the bytes returned by
/// `gst_buffer_map(READ)` on an NVMM buffer.
///
/// `mapped` must be the GstMemory map whose `data` is an
/// `NvBufSurface*` and whose `size` is at least
/// `size_of::<NvBufSurface>()`.
pub fn surface_from_bytes(mapped: &[u8]) -> Result<SurfaceInfo, SurfaceError> {
    let need = std::mem::size_of::<NvBufSurface>();
    // Exactly the struct, as the gst-nvmm-cpp reference checks: a
    // tight NV12 map is much larger and must never be cast.
    if mapped.len() != need {
        return Err(SurfaceError::WrongSize {
            got: mapped.len(),
            need,
        });
    }
    // SAFETY: `mapped` was produced by GStreamer's NVMM map and is at
    // least `size_of::<NvBufSurface>()`; the C structs are `repr(C)`
    // mirrors checked by tests. We only read plain fields.
    let surface = unsafe { &*(mapped.as_ptr().cast::<NvBufSurface>()) };
    if surface.batch_size == 0 || surface.num_filled == 0 || surface.surface_list.is_null() {
        return Err(SurfaceError::Empty);
    }
    // SAFETY: the driver's surfaceList points to at least batchSize
    // NvBufSurfaceParams; batch 0 is the frame we hand out.
    let params = unsafe { &*surface.surface_list };
    if params.plane_params.num_planes == 0
        || params.plane_params.num_planes as usize > NVBUF_MAX_PLANES
    {
        return Err(SurfaceError::Unsupported(format!(
            "plane count {}",
            params.plane_params.num_planes
        )));
    }
    let layout = match params.layout {
        0 => SurfaceLayout::Pitch,
        1 => SurfaceLayout::BlockLinear,
        other => return Err(SurfaceError::Unsupported(format!("layout enum {other}"))),
    };
    let mut planes = [None; NVBUF_MAX_PLANES];
    for (i, slot) in planes
        .iter_mut()
        .enumerate()
        .take(params.plane_params.num_planes as usize)
    {
        let pitch = params.plane_params.pitch[i] as usize;
        let width = params.plane_params.width[i] as usize;
        let bpp = params.plane_params.bytes_per_pix[i].max(1) as usize;
        let row_bytes = (width * bpp).min(pitch.max(1));
        *slot = Some(PlaneInfo {
            offset: params.plane_params.offset[i] as usize,
            pitch,
            row_bytes,
            rows: params.plane_params.height[i],
            size: params.plane_params.psize[i] as usize,
        });
    }
    Ok(SurfaceInfo {
        width: params.width,
        height: params.height,
        layout,
        fd: params.buffer_desc as i32,
        planes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    struct Fixture {
        surface: NvBufSurface,
        params: NvBufSurfaceParams,
    }

    fn fixture(padded: bool) -> Box<Fixture> {
        let mut fixture: Box<Fixture> = Box::new(unsafe { std::mem::zeroed() });
        let params_ptr = &mut fixture.params as *mut NvBufSurfaceParams;
        fixture.surface = NvBufSurface {
            gpu_id: 0,
            batch_size: 1,
            num_filled: 1,
            is_contiguous: true,
            mem_type: 0,
            surface_list: params_ptr,
            is_imported_buf: false,
            _reserved: [std::ptr::null_mut(); 4],
        };
        fixture.params.width = 1440;
        fixture.params.height = 1080;
        fixture.params.layout = 1; // block-linear
        fixture.params.buffer_desc = 42;
        fixture.params.plane_params.num_planes = 2;
        fixture.params.plane_params.width[0] = 1440;
        fixture.params.plane_params.height[0] = 1080;
        fixture.params.plane_params.pitch[0] = if padded { 1472 } else { 1440 };
        fixture.params.plane_params.bytes_per_pix[0] = 1;
        fixture.params.plane_params.psize[0] = if padded { 1_736_704 } else { 1440 * 1080 };
        fixture.params.plane_params.width[1] = 720;
        fixture.params.plane_params.height[1] = 540;
        fixture.params.plane_params.pitch[1] = if padded { 1472 } else { 1440 };
        fixture.params.plane_params.bytes_per_pix[1] = 2;
        fixture.params.plane_params.offset[1] = fixture.params.plane_params.psize[0];
        fixture.params.plane_params.psize[1] = fixture.params.plane_params.pitch[1] * 540;
        fixture
    }

    fn surface_bytes(fixture: &Fixture) -> &[u8] {
        // SAFETY: the slice covers exactly the `surface` field, and
        // the fixture stays alive for the borrow.
        unsafe {
            std::slice::from_raw_parts(
                &fixture.surface as *const NvBufSurface as *const u8,
                std::mem::size_of::<NvBufSurface>(),
            )
        }
    }

    #[test]
    fn reads_padded_block_linear_planes() {
        let fixture = fixture(true);
        let info = surface_from_bytes(surface_bytes(&fixture)).expect("parse");
        assert_eq!(info.width, 1440);
        assert_eq!(info.height, 1080);
        assert_eq!(info.layout, SurfaceLayout::BlockLinear);
        assert_eq!(info.fd, 42);
        assert_eq!(info.plane_count(), 2);
        let planes: Vec<_> = info.planes().copied().collect();
        assert_eq!(planes[0].pitch, 1472);
        assert_eq!(planes[0].row_bytes, 1440);
        assert_eq!(planes[1].row_bytes, 1440);
        assert_eq!(planes[1].offset, 1_736_704);
    }

    #[test]
    fn rejects_a_tight_nv12_map_as_a_surface() {
        let bytes = vec![0u8; 1440 * 1080 * 3 / 2];
        assert!(matches!(
            surface_from_bytes(&bytes),
            Err(SurfaceError::WrongSize { .. })
        ));
    }

    #[test]
    fn rejects_a_too_small_map() {
        let bytes = vec![0u8; 16];
        assert!(matches!(
            surface_from_bytes(&bytes),
            Err(SurfaceError::WrongSize { .. })
        ));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn struct_sizes_match_the_64_bit_headers() {
        assert_eq!(std::mem::size_of::<NvBufSurface>(), 72);
        assert_eq!(std::mem::size_of::<NvBufSurfaceParams>(), 416);
        assert_eq!(std::mem::size_of::<NvBufSurfacePlaneParams>(), 232);
    }
}
