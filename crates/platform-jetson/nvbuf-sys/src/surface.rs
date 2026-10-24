//! Safe wrappers over `libnvbufsurface` / `libnvbufsurftransform`
//! (feature `link`). All the `unsafe` of the create/map/transform
//! path lives here.

use crate::ffi::{self, ColorFormat, Filter, Rect};
use crate::{surface_from_bytes, NvBufSurface, NvbufError, SurfaceInfo, SurfaceLayout};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, PoisonError};

fn check(func: &'static str, code: i32) -> Result<(), NvbufError> {
    if code == 0 {
        Ok(())
    } else {
        Err(NvbufError::Call { func, code })
    }
}

/// The driver surface and its optional EGL mapping. Shared by the
/// `OwnedSurface` and every `EglImage` made from it, so the last holder
/// destroys the surface and no drop order can leave a mapping dangling.
struct Inner {
    ptr: NonNull<NvBufSurface>,
    egl: Mutex<Option<NonNull<c_void>>>,
}

// SAFETY: the surface is plain driver memory with no thread affinity; the
// Rust side only reads the pointer, and the one mutable piece (the EGL
// mapping) is behind a mutex.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    fn map_egl(&self) -> Result<NonNull<c_void>, NvbufError> {
        let mut slot = self.egl.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(image) = *slot {
            return Ok(image);
        }
        // SAFETY: valid surface, batch index 0.
        check("NvBufSurfaceMapEglImage", unsafe {
            ffi::NvBufSurfaceMapEglImage(self.ptr.as_ptr(), 0)
        })?;
        // SAFETY: surfaceList has one entry after a batch-1 create.
        let params = unsafe { &*(*self.ptr.as_ptr()).surface_list };
        let image = NonNull::new(params.mapped_addr.egl_image).ok_or_else(|| {
            NvbufError::Geometry("NvBufSurfaceMapEglImage left the EGL image null".into())
        })?;
        *slot = Some(image);
        Ok(image)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let mapped = self
            .egl
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some();
        // SAFETY: we own the surface; unmap of an unmapped surface only
        // returns an error code, which is ignored here.
        unsafe {
            if mapped {
                ffi::NvBufSurfaceUnMapEglImage(self.ptr.as_ptr(), 0);
            }
            ffi::NvBufSurfaceUnMap(self.ptr.as_ptr(), 0, 0);
            ffi::NvBufSurfaceDestroy(self.ptr.as_ptr());
        }
    }
}

/// The EGL image of an `OwnedSurface`, for CUDA to register. It keeps the
/// surface and its mapping alive while it exists.
pub struct EglImage {
    inner: Arc<Inner>,
    image: NonNull<c_void>,
}

// SAFETY: an EGLImageKHR is an opaque handle with no thread affinity.
unsafe impl Send for EglImage {}
unsafe impl Sync for EglImage {}

impl EglImage {
    /// The `EGLImageKHR`, valid while `self` lives.
    pub fn as_ptr(&self) -> *mut c_void {
        self.image.as_ptr()
    }

    /// The surface's width and height in pixels, as the driver reports them.
    pub fn size(&self) -> (u32, u32) {
        // SAFETY: surfaceList has one entry after a batch-1 create.
        let params = unsafe { &*(*self.inner.ptr.as_ptr()).surface_list };
        (params.width, params.height)
    }
}

/// A surface we created (pitch-linear, `NVBUF_MEM_DEFAULT`), mapped for
/// the CPU for its whole life. Destroyed with its last `EglImage`.
pub struct OwnedSurface {
    inner: Arc<Inner>,
    width: u32,
    height: u32,
    format: ColorFormat,
}

impl OwnedSurface {
    /// Create one surface, set `numFilled = 1` (the driver leaves it 0)
    /// and map it for CPU reads.
    pub fn create(
        width: u32,
        height: u32,
        format: ColorFormat,
        layout: SurfaceLayout,
    ) -> Result<Self, NvbufError> {
        if layout != SurfaceLayout::Pitch {
            return Err(NvbufError::Geometry(
                "OwnedSurface is pitch-linear only (the CPU reads it)".into(),
            ));
        }
        let mut params =
            ffi::NvBufSurfaceCreateParams::pitch_rgba_like(width, height, format.raw());
        let mut raw: *mut NvBufSurface = ptr::null_mut();
        // SAFETY: `raw` and `params` are valid for the call.
        check("NvBufSurfaceCreate", unsafe {
            ffi::NvBufSurfaceCreate(&mut raw, 1, &mut params)
        })?;
        let ptr = NonNull::new(raw)
            .ok_or_else(|| NvbufError::Geometry("NvBufSurfaceCreate returned null".into()))?;
        let surface = Self {
            inner: Arc::new(Inner {
                ptr,
                egl: Mutex::new(None),
            }),
            width,
            height,
            format,
        };
        // SAFETY: just created, exclusively ours.
        unsafe { (*surface.inner.ptr.as_ptr()).num_filled = 1 };
        // SAFETY: valid surface; on error `surface` drops and destroys it.
        check("NvBufSurfaceMap", unsafe {
            ffi::NvBufSurfaceMap(surface.inner.ptr.as_ptr(), 0, 0, ffi::NVBUF_MAP_READ)
        })?;
        Ok(surface)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Fill every byte with `value` and make it visible to the VIC.
    /// Meant to run once (the letterbox border).
    pub fn fill(&self, value: u8) -> Result<(), NvbufError> {
        // SAFETY: valid mapped surface.
        check("NvBufSurfaceMemSet", unsafe {
            ffi::NvBufSurfaceMemSet(self.inner.ptr.as_ptr(), 0, 0, value)
        })?;
        // SAFETY: as above.
        check("NvBufSurfaceSyncForDevice", unsafe {
            ffi::NvBufSurfaceSyncForDevice(self.inner.ptr.as_ptr(), 0, 0)
        })
    }

    /// Invalidate the CPU cache (call after the VIC wrote) and borrow
    /// plane 0's pixels with the driver's pitch.
    pub fn map_read(&self) -> Result<MappedPlane<'_>, NvbufError> {
        // SAFETY: valid mapped surface.
        check("NvBufSurfaceSyncForCpu", unsafe {
            ffi::NvBufSurfaceSyncForCpu(self.inner.ptr.as_ptr(), 0, 0)
        })?;
        // SAFETY: surfaceList has one entry after a batch-1 create.
        let params = unsafe { &*(*self.inner.ptr.as_ptr()).surface_list };
        let addr = params.mapped_addr.addr[0].cast::<u8>();
        let pitch = params.plane_params.pitch[0] as usize;
        let min = pitch * self.height as usize;
        let psize = params.plane_params.psize[0] as usize;
        if addr.is_null() || pitch < self.width as usize * self.format.bytes_per_pixel() {
            return Err(NvbufError::Geometry(format!(
                "mapped plane null or pitch {pitch} too small"
            )));
        }
        let len = if psize >= min { min } else { psize };
        if len < min {
            return Err(NvbufError::Geometry(format!("psize {psize} < {min}")));
        }
        // SAFETY: the mapping spans psize >= pitch*height bytes and lives
        // until `self` drops; the borrow ties the slice to `&self`.
        let data = unsafe { std::slice::from_raw_parts(addr, len) };
        Ok(MappedPlane {
            data,
            pitch,
            width: self.width as usize,
            height: self.height as usize,
        })
    }

    /// Map the surface as an EGL image (once; later calls share it) for
    /// `cudaGraphicsEGLRegisterImage`.
    pub fn map_egl_image(&self) -> Result<EglImage, NvbufError> {
        let image = self.inner.map_egl()?;
        Ok(EglImage {
            inner: Arc::clone(&self.inner),
            image,
        })
    }

    fn as_ptr(&self) -> *mut NvBufSurface {
        self.inner.ptr.as_ptr()
    }
}

/// CPU view of plane 0 of an `OwnedSurface`.
#[derive(Debug, Clone, Copy)]
pub struct MappedPlane<'a> {
    pub data: &'a [u8],
    pub pitch: usize,
    pub width: usize,
    pub height: usize,
}

/// A frame surface borrowed from a mapped NVMM `GstBuffer`.
pub struct SurfaceRef<'a> {
    ptr: *mut NvBufSurface,
    info: SurfaceInfo,
    _map: PhantomData<&'a [u8]>,
}

impl<'a> SurfaceRef<'a> {
    /// `mapped` is the bytes of `gst_buffer_map(READ)` on an NVMM buffer
    /// (exactly one `NvBufSurface`). The surface stays valid while the
    /// map is held, which the lifetime enforces.
    pub fn from_mapped_bytes(mapped: &'a [u8]) -> Result<Self, NvbufError> {
        let info = surface_from_bytes(mapped)?;
        Ok(Self {
            // The C API takes a non-const pointer but does not write the
            // source surface.
            ptr: mapped.as_ptr().cast::<NvBufSurface>().cast_mut(),
            info,
            _map: PhantomData,
        })
    }

    pub fn info(&self) -> &SurfaceInfo {
        &self.info
    }
}

/// VIC scale + colour-convert `src_rect` of `src` into `dst_rect` of
/// `dst`. Sets the VIC session on the calling thread first (the
/// session is per thread and per call, as the reference proved).
pub fn transform_letterbox(
    src: &SurfaceRef<'_>,
    dst: &OwnedSurface,
    src_rect: Rect,
    dst_rect: Rect,
    filter: Filter,
) -> Result<(), NvbufError> {
    let mut session = ffi::NvBufSurfTransformConfigParams {
        compute_mode: ffi::NVBUFSURF_TRANSFORM_COMPUTE_VIC,
        gpu_id: 0,
        cuda_stream: ptr::null_mut(),
        _reserved: [ptr::null_mut(); 4],
    };
    // SAFETY: `session` is valid for the call.
    check("NvBufSurfTransformSetSessionParams", unsafe {
        ffi::NvBufSurfTransformSetSessionParams(&mut session)
    })?;
    let mut s = src_rect.to_c();
    let mut d = dst_rect.to_c();
    let mut params = ffi::NvBufSurfTransformParams {
        transform_flag: ffi::NVBUFSURF_TRANSFORM_FILTER
            | ffi::NVBUFSURF_TRANSFORM_CROP_SRC
            | ffi::NVBUFSURF_TRANSFORM_CROP_DST,
        transform_flip: 0,
        transform_filter: filter.raw(),
        src_rect: &mut s,
        dst_rect: &mut d,
        _reserved: [ptr::null_mut(); 4],
    };
    // SAFETY: both surfaces are live (borrow / owner), the rects and
    // params outlive the call.
    let code = unsafe { ffi::NvBufSurfTransform(src.ptr, dst.as_ptr(), &mut params) };
    if code == ffi::NVBUFSURF_TRANSFORM_ERROR_SUCCESS {
        Ok(())
    } else {
        Err(NvbufError::Call {
            func: "NvBufSurfTransform",
            code,
        })
    }
}
