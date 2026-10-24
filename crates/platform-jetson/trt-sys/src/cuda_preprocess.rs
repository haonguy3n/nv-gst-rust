//! `RgbaTensorWriter`. Role: Adapter from "a pitch-linear RGBA NvBufSurface"
//! to "a planar float tensor in mapped host memory", on the GPU. Pattern:
//! register once, write per frame: the surface's EGL image is registered with
//! CUDA when the writer is made (the surface is pooled), and each frame only
//! launches the kernel in `shim/preprocess_kernels.cu` on the engine's stream.

use crate::{Error, Result, Stream};
use nvbuf_sys::{EglImage, OwnedSurface};
use std::os::raw::c_void;
use std::ptr::NonNull;

#[repr(C)]
struct CpSource {
    _private: [u8; 0],
}

extern "C" {
    fn cp_last_error() -> *const std::os::raw::c_char;
    fn cp_source_register(egl_image: *mut c_void, width: u32, height: u32) -> *mut CpSource;
    fn cp_source_destroy(source: *mut CpSource);
    fn cp_write_tensor(
        source: *const CpSource,
        swap_rb: i32,
        scale: f32,
        tensor_host: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}

fn preprocess_error() -> Error {
    // SAFETY: the shim returns a NUL-terminated string valid until the next
    // call on this thread.
    let text = unsafe { std::ffi::CStr::from_ptr(cp_last_error()) };
    Error::new(text.to_string_lossy().into_owned())
}

/// A surface registered with CUDA. Drop order: the registration goes first
/// (in `Drop`), then `_image` releases the EGL mapping and, last, the surface.
pub struct RgbaTensorWriter {
    raw: NonNull<CpSource>,
    width: usize,
    height: usize,
    _image: EglImage,
}

// SAFETY: the registration is a plain CUDA resource, used by one caller at a
// time (`write` takes `&self` but the stream orders the launches).
unsafe impl Send for RgbaTensorWriter {}

impl RgbaTensorWriter {
    /// Maps `surface` as an EGL image and registers it with CUDA. The surface
    /// must be pitch-linear RGBA; the registration is cached for its life.
    pub fn register(surface: &OwnedSurface) -> Result<Self> {
        crate::set_device_for_thread()?;
        let image = surface
            .map_egl_image()
            .map_err(|e| Error::new(format!("mapping the surface as an EGL image: {e}")))?;
        let (width, height) = (surface.width(), surface.height());
        // SAFETY: the image is live (held by the writer) and its size is the surface's.
        let raw = NonNull::new(unsafe { cp_source_register(image.as_ptr(), width, height) })
            .ok_or_else(preprocess_error)?;
        Ok(Self {
            raw,
            width: width as usize,
            height: height as usize,
            _image: image,
        })
    }

    /// Writes `out[0..3*w*h]` as planar floats (plane 0 = B when `swap_rb`),
    /// each `byte * scale`, and returns once the stream has finished it.
    /// `out` must be mapped host memory (`MappedBuffer`); `stream` is the
    /// one the engine runs on, so the write is ordered before the inference.
    pub fn write(&self, out: &mut [f32], swap_rb: bool, scale: f32, stream: &Stream) -> Result<()> {
        let need = 3 * self.width * self.height;
        if out.len() < need {
            return Err(Error::new(format!(
                "tensor has {} elements, need {need}",
                out.len()
            )));
        }
        crate::set_device_for_thread()?;
        // SAFETY: live registration; `out` is a live, mapped allocation of
        // at least `need` floats; the stream is live. The launch finishes in
        // `synchronize` below, before `out` is borrowed again.
        let code = unsafe {
            cp_write_tensor(
                self.raw.as_ptr(),
                i32::from(swap_rb),
                scale,
                out.as_mut_ptr().cast(),
                stream.as_raw(),
            )
        };
        if code < 0 {
            return Err(preprocess_error());
        }
        stream.synchronize()
    }
}

impl Drop for RgbaTensorWriter {
    fn drop(&mut self) {
        // SAFETY: `raw` is live; the EGL image is still mapped (a field).
        unsafe { cp_source_destroy(self.raw.as_ptr()) }
    }
}
