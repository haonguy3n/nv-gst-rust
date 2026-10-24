//! Pure pixel math: a pitch-linear RGBA image to a planar CHW float
//! tensor (colour order and normalisation chosen by the caller).

use crate::NvbufError;

/// How to write the planes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanarSpec {
    /// Plane 0 gets B and plane 2 gets R (model wants BGR).
    pub swap_rb: bool,
    /// Multiplier on the 0..=255 byte (1/255 for unit, 1 for none).
    pub scale: f32,
}

/// Bytes of one image row that carry pixels, and the check that the
/// slice covers `height` pitched rows.
fn check(rgba: &[u8], pitch: usize, width: usize, height: usize) -> Result<(), NvbufError> {
    if width == 0 || height == 0 || pitch < width * 4 {
        return Err(NvbufError::Geometry(format!(
            "pitch {pitch} cannot hold {width}x{height} RGBA"
        )));
    }
    let need = pitch * (height - 1) + width * 4;
    if rgba.len() < need {
        return Err(NvbufError::Geometry(format!(
            "RGBA slice has {} bytes, need {need}",
            rgba.len()
        )));
    }
    Ok(())
}

/// Write `out[0..w*h]`, `out[w*h..2*w*h]`, `out[2*w*h..3*w*h]` from the
/// RGBA rows (alpha dropped). `out` may be longer; the rest is untouched.
pub fn rgba_to_planar(
    rgba: &[u8],
    pitch: usize,
    width: usize,
    height: usize,
    spec: PlanarSpec,
    out: &mut [f32],
) -> Result<(), NvbufError> {
    check(rgba, pitch, width, height)?;
    let plane = width * height;
    if out.len() < 3 * plane {
        return Err(NvbufError::Geometry(format!(
            "tensor has {} elements, need {}",
            out.len(),
            3 * plane
        )));
    }
    let mut lut = [0f32; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = i as f32 * spec.scale;
    }
    let (c0, c2) = if spec.swap_rb { (2, 0) } else { (0, 2) };
    let (p0, rest) = out.split_at_mut(plane);
    let (p1, p2) = rest.split_at_mut(plane);
    for y in 0..height {
        let row = &rgba[y * pitch..y * pitch + width * 4];
        let o = y * width;
        for (x, px) in row.as_chunks::<4>().0.iter().enumerate() {
            p0[o + x] = lut[px[c0] as usize];
            p1[o + x] = lut[px[1] as usize];
            p2[o + x] = lut[px[c2] as usize];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2x2 image, pitch 12 (4 bytes of padding per row).
    fn image() -> Vec<u8> {
        vec![
            10, 20, 30, 255, 40, 50, 60, 255, 9, 9, 9, 9, // row 0 + pad
            70, 80, 90, 255, 100, 110, 120, 255, 9, 9, 9, 9, // row 1 + pad
        ]
    }

    #[test]
    fn rgb_order_with_pitch_padding() {
        let mut out = vec![-1.0; 12];
        let spec = PlanarSpec {
            swap_rb: false,
            scale: 1.0,
        };
        rgba_to_planar(&image(), 12, 2, 2, spec, &mut out).unwrap();
        assert_eq!(&out[0..4], &[10.0, 40.0, 70.0, 100.0]);
        assert_eq!(&out[4..8], &[20.0, 50.0, 80.0, 110.0]);
        assert_eq!(&out[8..12], &[30.0, 60.0, 90.0, 120.0]);
    }

    #[test]
    fn bgr_order_and_unit_scale() {
        let mut out = vec![0.0; 12];
        let spec = PlanarSpec {
            swap_rb: true,
            scale: 1.0 / 255.0,
        };
        rgba_to_planar(&image(), 12, 2, 2, spec, &mut out).unwrap();
        let k = 1.0 / 255.0;
        assert_eq!(out[0], 30.0 * k);
        assert_eq!(out[8], 10.0 * k);
        assert_eq!(out[4], 20.0 * k);
    }

    #[test]
    fn longer_tensor_tail_is_untouched() {
        let mut out = vec![-1.0; 14];
        let spec = PlanarSpec {
            swap_rb: false,
            scale: 1.0,
        };
        rgba_to_planar(&image(), 12, 2, 2, spec, &mut out).unwrap();
        assert_eq!(&out[12..], &[-1.0, -1.0]);
    }

    #[test]
    fn last_row_needs_no_trailing_pad() {
        let img = image();
        let mut out = vec![0.0; 12];
        let spec = PlanarSpec {
            swap_rb: false,
            scale: 1.0,
        };
        assert!(rgba_to_planar(&img[..20], 12, 2, 2, spec, &mut out).is_ok());
        assert!(rgba_to_planar(&img[..19], 12, 2, 2, spec, &mut out).is_err());
    }

    #[test]
    fn rejects_bad_geometry() {
        let mut out = vec![0.0; 12];
        let spec = PlanarSpec {
            swap_rb: false,
            scale: 1.0,
        };
        assert!(rgba_to_planar(&image(), 4, 2, 2, spec, &mut out).is_err());
        let mut small = vec![0.0; 11];
        assert!(rgba_to_planar(&image(), 12, 2, 2, spec, &mut small).is_err());
    }
}
