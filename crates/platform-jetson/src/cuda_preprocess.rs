//! `CudaTensorWriter`. Role: Adapter from `trt_sys::RgbaTensorWriter` to the
//! preprocessor's planar write. Pattern: the strategy the CPU writer
//! (`nvbuf_sys::planar`) also fills: same input (the VIC's pooled RGBA
//! surface), same output (the engine's mapped input tensor), same colour
//! order and scale, but written by a kernel on the engine's CUDA stream.

use crate::cuda::{platform_err, CudaStream};
use crate::vic::VicScaler;
use edge_core::{Error, GpuStream, Result};
use nvbuf_sys::planar::PlanarSpec;

pub struct CudaTensorWriter {
    writer: trt_sys::RgbaTensorWriter,
    spec: PlanarSpec,
}

impl CudaTensorWriter {
    /// Registers the VIC's destination surface with CUDA, once. `None`, with
    /// one warning, when CUDA cannot map it: the CPU writer is used instead.
    pub fn try_new(vic: &VicScaler, spec: PlanarSpec) -> Option<Self> {
        match trt_sys::RgbaTensorWriter::register(vic.dst()) {
            Ok(writer) => {
                log::info!("jetson preprocess: CUDA tensor writer on the VIC surface");
                Some(Self { writer, spec })
            }
            Err(e) => {
                log::warn!("jetson preprocess: CUDA cannot map the VIC surface ({e}): using the CPU writer");
                None
            }
        }
    }

    /// Writes the tensor from the surface the VIC just filled; returns when
    /// the GPU is done. `out` is the engine's mapped input.
    pub fn write(&self, out: &mut [f32], stream: &GpuStream) -> Result<()> {
        let cuda = stream.backend::<CudaStream>().ok_or_else(|| {
            Error::Platform("the CUDA tensor writer needs the engine's CudaStream".into())
        })?;
        self.writer
            .write(out, self.spec.swap_rb, self.spec.scale, cuda.inner())
            .map_err(platform_err)
    }
}
