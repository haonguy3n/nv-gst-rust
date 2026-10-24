//! `JetsonPreprocessor`. Role: Preprocessor (pipeline stage). Pattern:
//! two steps behind the one trait method: the VIC letterboxes and
//! colour-converts the NVMM frame into a pooled RGBA surface
//! (`VicScaler`), then the CPU writes the planar float tensor from it
//! (`nvbuf_sys::planar`). Letterbox geometry is `edge_core::Letterbox`,
//! the same the decoders invert, so nothing is recorded per frame.
//!
//! The VIC converts NV12 with its own matrix (BT.601 limited); the
//! `colorspace` setting cannot change it, so another value only warns.

use crate::gst_source::GstFrameBuffer;
use crate::vic::VicScaler;
use edge_core::{
    normalize_factor, Error, GpuStream, HwFrame, InputColor, InputColorspace, PreConfig,
    Preprocessor, Result, Tensor,
};
use nvbuf_sys::planar::{rgba_to_planar, PlanarSpec};
use nvbuf_sys::SurfaceRef;

pub struct JetsonPreprocessor {
    vic: VicScaler,
    spec: PlanarSpec,
    dst_width: usize,
    dst_height: usize,
}

/// Colour order and normalisation of the planar write, from the config.
fn planar_spec(cfg: &PreConfig) -> PlanarSpec {
    PlanarSpec {
        swap_rb: cfg.color == InputColor::Bgr,
        scale: normalize_factor(cfg.normalize),
    }
}

impl JetsonPreprocessor {
    pub fn new(cfg: &PreConfig) -> Result<Self> {
        if cfg.colorspace != InputColorspace::Bt601Limited {
            log::warn!(
                "jetson preprocess: the VIC converts NV12 as BT.601 limited; \
                 colorspace {:?} is not applied",
                cfg.colorspace
            );
        }
        Ok(Self {
            vic: VicScaler::new(cfg.dst_width, cfg.dst_height, cfg.pad_value)?,
            spec: planar_spec(cfg),
            dst_width: cfg.dst_width as usize,
            dst_height: cfg.dst_height as usize,
        })
    }
}

impl Preprocessor for JetsonPreprocessor {
    fn run(&mut self, frame: &HwFrame, dst: &mut Tensor, _stream: &GpuStream) -> Result<()> {
        let buffer = frame
            .buf
            .as_any()
            .downcast_ref::<GstFrameBuffer>()
            .and_then(GstFrameBuffer::gst_buffer)
            .ok_or_else(|| {
                Error::Platform("jetson preprocess needs an NVMM GstBuffer frame".into())
            })?;
        let out = dst.as_f32_slice_mut().ok_or_else(|| {
            Error::Platform("jetson preprocess needs a host-visible F32 input tensor".into())
        })?;
        // An NVMM buffer's map is the NvBufSurface itself.
        let map = buffer
            .map_readable()
            .map_err(|e| Error::Platform(format!("mapping the NVMM frame: {e}")))?;
        let src = SurfaceRef::from_mapped_bytes(map.as_slice())
            .map_err(|e| Error::Platform(format!("reading the NVMM surface: {e}")))?;
        self.vic.scale(&src)?;
        let plane = self
            .vic
            .dst()
            .map_read()
            .map_err(|e| Error::Platform(format!("reading the scaled surface: {e}")))?;
        rgba_to_planar(
            plane.data,
            plane.pitch,
            self.dst_width,
            self.dst_height,
            self.spec,
            out,
        )
        .map_err(|e| Error::Platform(format!("writing the input tensor: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::{InputNormalize, Layout};

    fn cfg(color: InputColor, normalize: InputNormalize) -> PreConfig {
        PreConfig {
            src_width: 1440,
            src_height: 1080,
            dst_width: 416,
            dst_height: 320,
            layout: Layout::BlockLinear,
            color,
            colorspace: InputColorspace::Bt601Limited,
            normalize,
            pad_value: 114,
        }
    }

    #[test]
    fn spec_follows_the_config() {
        let s = planar_spec(&cfg(InputColor::Bgr, InputNormalize::Unit));
        assert!(s.swap_rb);
        assert_eq!(s.scale, 1.0 / 255.0);
        let s = planar_spec(&cfg(InputColor::Rgb, InputNormalize::None));
        assert!(!s.swap_rb);
        assert_eq!(s.scale, 1.0);
    }
}
