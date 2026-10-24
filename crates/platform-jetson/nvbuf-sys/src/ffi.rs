//! `#[repr(C)]` mirrors of the create/transform structs of
//! `nvbufsurface.h` / `nvbufsurftransform.h` (r39.2) and the extern
//! declarations. The externs exist only with the `link` feature.

// Without `link` the C calls that use most of this are not compiled.
#![cfg_attr(not(feature = "link"), allow(dead_code))]

#[cfg(feature = "link")]
use crate::NvBufSurface;
use std::ffi::c_void;

const STRUCTURE_PADDING: usize = 4;

/// `NvBufSurfaceColorFormat` values this crate creates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorFormat {
    Gray8,
    Rgba,
}

impl ColorFormat {
    pub(crate) fn raw(self) -> i32 {
        match self {
            Self::Gray8 => NVBUF_COLOR_FORMAT_GRAY8,
            Self::Rgba => NVBUF_COLOR_FORMAT_RGBA,
        }
    }

    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Rgba => 4,
        }
    }
}

/// `NvBufSurfTransform_Inter` values worth choosing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Bilinear,
}

impl Filter {
    pub(crate) fn raw(self) -> i32 {
        match self {
            Self::Nearest => NVBUFSURF_TRANSFORM_INTER_NEAREST,
            Self::Bilinear => NVBUFSURF_TRANSFORM_INTER_BILINEAR,
        }
    }
}

/// A rectangle in pixels (`NvBufSurfTransformRect` without the padding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn new(left: u32, top: u32, width: u32, height: u32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }

    pub(crate) fn to_c(self) -> NvBufSurfTransformRect {
        NvBufSurfTransformRect {
            top: self.top,
            left: self.left,
            width: self.width,
            height: self.height,
            _reserved: [std::ptr::null_mut(); STRUCTURE_PADDING],
        }
    }
}

// Enum values copied from the headers.
pub(crate) const NVBUF_COLOR_FORMAT_GRAY8: i32 = 1;
pub(crate) const NVBUF_COLOR_FORMAT_RGBA: i32 = 19;
pub(crate) const NVBUF_LAYOUT_PITCH: i32 = 0;
#[allow(dead_code)]
pub(crate) const NVBUF_LAYOUT_BLOCK_LINEAR: i32 = 1;
pub(crate) const NVBUF_MEM_DEFAULT: i32 = 0;
pub(crate) const NVBUF_MAP_READ: i32 = 0;
pub(crate) const NVBUFSURF_TRANSFORM_COMPUTE_VIC: i32 = 2;
pub(crate) const NVBUFSURF_TRANSFORM_INTER_NEAREST: i32 = 0;
pub(crate) const NVBUFSURF_TRANSFORM_INTER_BILINEAR: i32 = 1;
pub(crate) const NVBUFSURF_TRANSFORM_CROP_SRC: u32 = 1;
pub(crate) const NVBUFSURF_TRANSFORM_CROP_DST: u32 = 1 << 1;
pub(crate) const NVBUFSURF_TRANSFORM_FILTER: u32 = 1 << 2;
pub(crate) const NVBUFSURF_TRANSFORM_ERROR_SUCCESS: i32 = 0;

/// `NvBufSurfaceCreateParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct NvBufSurfaceCreateParams {
    pub gpu_id: u32,
    pub width: u32,
    pub height: u32,
    pub size: u32,
    pub is_contiguous: bool,
    pub color_format: i32,
    pub layout: i32,
    pub mem_type: i32,
    pub _reserved: [*mut c_void; STRUCTURE_PADDING],
}

/// `NvBufSurfTransformRect` (note: `top` comes before `left`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct NvBufSurfTransformRect {
    pub top: u32,
    pub left: u32,
    pub width: u32,
    pub height: u32,
    pub _reserved: [*mut c_void; STRUCTURE_PADDING],
}

/// `NvBufSurfTransformConfigParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct NvBufSurfTransformConfigParams {
    pub compute_mode: i32,
    pub gpu_id: i32,
    pub cuda_stream: *mut c_void,
    pub _reserved: [*mut c_void; STRUCTURE_PADDING],
}

/// `NvBufSurfTransformParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct NvBufSurfTransformParams {
    pub transform_flag: u32,
    pub transform_flip: i32,
    pub transform_filter: i32,
    pub src_rect: *mut NvBufSurfTransformRect,
    pub dst_rect: *mut NvBufSurfTransformRect,
    pub _reserved: [*mut c_void; STRUCTURE_PADDING],
}

impl NvBufSurfaceCreateParams {
    pub(crate) fn pitch_rgba_like(width: u32, height: u32, color_format: i32) -> Self {
        Self {
            gpu_id: 0,
            width,
            height,
            size: 0,
            is_contiguous: false,
            color_format,
            layout: NVBUF_LAYOUT_PITCH,
            mem_type: NVBUF_MEM_DEFAULT,
            _reserved: [std::ptr::null_mut(); STRUCTURE_PADDING],
        }
    }
}

#[cfg(feature = "link")]
#[link(name = "nvbufsurface")]
extern "C" {
    pub(crate) fn NvBufSurfaceCreate(
        surf: *mut *mut NvBufSurface,
        batch_size: u32,
        params: *mut NvBufSurfaceCreateParams,
    ) -> i32;
    pub(crate) fn NvBufSurfaceDestroy(surf: *mut NvBufSurface) -> i32;
    pub(crate) fn NvBufSurfaceMap(
        surf: *mut NvBufSurface,
        index: i32,
        plane: i32,
        map_type: i32,
    ) -> i32;
    pub(crate) fn NvBufSurfaceUnMap(surf: *mut NvBufSurface, index: i32, plane: i32) -> i32;
    pub(crate) fn NvBufSurfaceSyncForCpu(surf: *mut NvBufSurface, index: i32, plane: i32) -> i32;
    pub(crate) fn NvBufSurfaceSyncForDevice(surf: *mut NvBufSurface, index: i32, plane: i32)
        -> i32;
    pub(crate) fn NvBufSurfaceMemSet(
        surf: *mut NvBufSurface,
        index: i32,
        plane: i32,
        value: u8,
    ) -> i32;
}

#[cfg(feature = "link")]
#[link(name = "nvbufsurftransform")]
extern "C" {
    pub(crate) fn NvBufSurfTransformSetSessionParams(
        config: *mut NvBufSurfTransformConfigParams,
    ) -> i32;
    pub(crate) fn NvBufSurfTransform(
        src: *mut NvBufSurface,
        dst: *mut NvBufSurface,
        params: *mut NvBufSurfTransformParams,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // Numbers measured by compiling the real r39.2 headers with a C
    // compiler (offsetof/sizeof, LP64). See the crate README note in
    // the task report: `gcc -I<copy of the two headers> sz.c`.
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn create_and_transform_structs_match_the_64_bit_headers() {
        assert_eq!(size_of::<NvBufSurfaceCreateParams>(), 64);
        assert_eq!(offset_of!(NvBufSurfaceCreateParams, width), 4);
        assert_eq!(offset_of!(NvBufSurfaceCreateParams, height), 8);
        assert_eq!(offset_of!(NvBufSurfaceCreateParams, size), 12);
        assert_eq!(offset_of!(NvBufSurfaceCreateParams, is_contiguous), 16);
        assert_eq!(offset_of!(NvBufSurfaceCreateParams, color_format), 20);
        assert_eq!(offset_of!(NvBufSurfaceCreateParams, layout), 24);
        assert_eq!(offset_of!(NvBufSurfaceCreateParams, mem_type), 28);

        assert_eq!(size_of::<NvBufSurfTransformRect>(), 48);
        assert_eq!(offset_of!(NvBufSurfTransformRect, left), 4);
        assert_eq!(offset_of!(NvBufSurfTransformRect, width), 8);
        assert_eq!(offset_of!(NvBufSurfTransformRect, height), 12);

        assert_eq!(size_of::<NvBufSurfTransformConfigParams>(), 48);
        assert_eq!(offset_of!(NvBufSurfTransformConfigParams, gpu_id), 4);
        assert_eq!(offset_of!(NvBufSurfTransformConfigParams, cuda_stream), 8);

        assert_eq!(size_of::<NvBufSurfTransformParams>(), 64);
        assert_eq!(offset_of!(NvBufSurfTransformParams, transform_flip), 4);
        assert_eq!(offset_of!(NvBufSurfTransformParams, transform_filter), 8);
        assert_eq!(offset_of!(NvBufSurfTransformParams, src_rect), 16);
        assert_eq!(offset_of!(NvBufSurfTransformParams, dst_rect), 24);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn mapped_address_offsets_match_the_headers() {
        use crate::{NvBufSurfaceMappedAddr, NvBufSurfaceParams, NvBufSurfacePlaneParams};
        assert_eq!(size_of::<NvBufSurfaceMappedAddr>(), 88);
        assert_eq!(offset_of!(NvBufSurfaceParams, plane_params), 48);
        assert_eq!(offset_of!(NvBufSurfaceParams, mapped_addr), 280);
        assert_eq!(offset_of!(NvBufSurfacePlaneParams, psize), 68);
        assert_eq!(offset_of!(NvBufSurfacePlaneParams, bytes_per_pix), 84);
    }

    #[test]
    fn header_enum_values() {
        assert_eq!(NVBUF_COLOR_FORMAT_RGBA, 19);
        assert_eq!(NVBUF_COLOR_FORMAT_GRAY8, 1);
        assert_eq!(NVBUF_LAYOUT_BLOCK_LINEAR, 1);
        assert_eq!(NVBUFSURF_TRANSFORM_COMPUTE_VIC, 2);
        assert_eq!(Rect::new(1, 2, 3, 4).to_c().top, 2);
    }
}
