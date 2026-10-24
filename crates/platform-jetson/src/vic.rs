//! VIC scaler. Role: Adapter from "an NVMM capture surface" to "a
//! letterboxed RGBA surface of the network input size". Pattern: the
//! destination surface is a pooled resource created once, the border
//! filled once, and only the content rectangle rewritten per frame
//! (the reference's "pad fill once, CROP_DST letterbox").

use edge_core::{Error, Letterbox, Result};
use nvbuf_sys::{ColorFormat, Filter, OwnedSurface, Rect, SurfaceLayout, SurfaceRef};

pub struct VicScaler {
    dst: OwnedSurface,
    dst_width: u32,
    dst_height: u32,
    /// Letterbox of the last source size, so it is computed once.
    last: Option<((u32, u32), Letterbox)>,
}

fn platform(what: &str, e: nvbuf_sys::NvbufError) -> Error {
    Error::Platform(format!("vic: {what}: {e}"))
}

impl VicScaler {
    /// A `dst_width` x `dst_height` RGBA surface, border-filled with `pad`.
    pub fn new(dst_width: u32, dst_height: u32, pad: u8) -> Result<Self> {
        let dst = OwnedSurface::create(
            dst_width,
            dst_height,
            ColorFormat::Rgba,
            SurfaceLayout::Pitch,
        )
        .map_err(|e| platform("creating the RGBA surface", e))?;
        dst.fill(pad)
            .map_err(|e| platform("filling the border", e))?;
        Ok(Self {
            dst,
            dst_width,
            dst_height,
            last: None,
        })
    }

    /// The scaled surface; `OwnedSurface::map_read` after `scale`.
    pub fn dst(&self) -> &OwnedSurface {
        &self.dst
    }

    fn letterbox_for(&mut self, src_w: u32, src_h: u32) -> Letterbox {
        match self.last {
            Some((size, lb)) if size == (src_w, src_h) => lb,
            _ => {
                let lb = Letterbox::new(src_w, src_h, self.dst_width, self.dst_height);
                self.last = Some(((src_w, src_h), lb));
                lb
            }
        }
    }

    /// Whole source frame into the letterbox rectangle (bilinear: the
    /// VIC default is nearest, training resizes are bilinear). The VIC
    /// also converts NV12 to RGBA.
    pub fn scale(&mut self, src: &SurfaceRef<'_>) -> Result<Letterbox> {
        let (sw, sh) = (src.info().width, src.info().height);
        let lb = self.letterbox_for(sw, sh);
        let dst_rect = Rect::new(lb.pad_x, lb.pad_y, lb.resized_width, lb.resized_height);
        nvbuf_sys::transform_letterbox(
            src,
            &self.dst,
            Rect::new(0, 0, sw, sh),
            dst_rect,
            Filter::Bilinear,
        )
        .map_err(|e| platform("transform", e))?;
        Ok(lb)
    }
}
