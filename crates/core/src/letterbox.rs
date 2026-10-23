//! One definition of letterbox geometry and input normalisation,
//! shared by every preprocessor (sim CPU today, Jetson VIC/CUDA
//! next) and by the decoders that map boxes back to the frame.
//! Two definitions that differ by a rounding choice would make
//! the PC's golden values disagree with the device's by a pixel.

use crate::config::{InputColor, InputNormalize};

/// Uniform scale + symmetric padding from a source frame into a
/// model input: `model_px = frame_px * scale + pad`, where the
/// resized size is `round(src * scale)` and the pad is the
/// integer half of the leftover. Integer pads are what a scaler
/// (VIC destination rectangle, CPU loop bounds) actually applies,
/// so the decoders invert exactly this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub resized_width: u32,
    pub resized_height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
}

impl Letterbox {
    pub fn new(src_width: u32, src_height: u32, dst_width: u32, dst_height: u32) -> Self {
        let scale = (dst_width as f32 / src_width.max(1) as f32)
            .min(dst_height as f32 / src_height.max(1) as f32);
        let resized_width = ((src_width as f32 * scale).round() as u32).clamp(1, dst_width.max(1));
        let resized_height =
            ((src_height as f32 * scale).round() as u32).clamp(1, dst_height.max(1));
        Self {
            scale,
            resized_width,
            resized_height,
            pad_x: dst_width.saturating_sub(resized_width) / 2,
            pad_y: dst_height.saturating_sub(resized_height) / 2,
        }
    }

    pub fn pad_x_f32(&self) -> f32 {
        self.pad_x as f32
    }

    pub fn pad_y_f32(&self) -> f32 {
        self.pad_y as f32
    }
}

/// Multiplier applied to byte-domain pixels (0..=255) when writing
/// the planar input tensor.
pub fn normalize_factor(normalize: InputNormalize) -> f32 {
    match normalize {
        InputNormalize::Unit => 1.0 / 255.0,
        InputNormalize::None => 1.0,
    }
}

/// Channel order for the planar write: returns the three plane
/// values in plane order for one pixel.
pub fn order_channels(color: InputColor, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    match color {
        InputColor::Rgb => (r, g, b),
        InputColor::Bgr => (b, g, r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_frame_into_square_pads_vertically() {
        // 128x64 into 64x64: scale 0.5, resized 64x32, pad_y 16.
        let lb = Letterbox::new(128, 64, 64, 64);
        assert_eq!(lb.scale, 0.5);
        assert_eq!((lb.resized_width, lb.resized_height), (64, 32));
        assert_eq!((lb.pad_x, lb.pad_y), (0, 16));
    }

    #[test]
    fn tall_frame_into_square_pads_horizontally() {
        let lb = Letterbox::new(64, 128, 64, 64);
        assert_eq!((lb.resized_width, lb.resized_height), (32, 64));
        assert_eq!((lb.pad_x, lb.pad_y), (16, 0));
    }

    #[test]
    fn exact_fit_has_no_padding() {
        let lb = Letterbox::new(640, 640, 640, 640);
        assert_eq!(lb.scale, 1.0);
        assert_eq!((lb.pad_x, lb.pad_y), (0, 0));
    }

    #[test]
    fn upscale_and_odd_leftover_stay_in_bounds() {
        let lb = Letterbox::new(100, 100, 641, 641);
        assert!(lb.resized_width <= 641 && lb.resized_height <= 641);
        assert_eq!(lb.pad_x, (641 - lb.resized_width) / 2);
    }
}
