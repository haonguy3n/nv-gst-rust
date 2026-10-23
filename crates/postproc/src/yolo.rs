use crate::nms::nms;
use edge_core::{Detection, Detections, Error, FrameMeta, PostProcessor, Result, Tensor};

/// Letterbox geometry shared by the decoders: the preprocessor
/// scaled the frame by `scale` and padded it symmetrically, so
/// `model_px = frame_px * scale + pad`. Returns
/// `(scale, pad_x, pad_y)` in model pixels.
pub(crate) fn letterbox_params(
    model_width: u32,
    model_height: u32,
    frame_width: u32,
    frame_height: u32,
) -> (f32, f32, f32) {
    let (mw, mh) = (model_width as f32, model_height as f32);
    let (fw, fh) = (frame_width as f32, frame_height as f32);
    let scale = (mw / fw).min(mh / fh);
    (scale, (mw - fw * scale) / 2.0, (mh - fh * scale) / 2.0)
}

/// Decoder for the YOLOv8/v11 raw head output: one tensor of shape
/// `[1, 4 + num_classes, num_candidates]` where each candidate is
/// `cx, cy, w, h` in model-input pixels followed by per-class scores
/// (no separate objectness in v8+).
///
/// Coordinates are mapped back through the letterbox the
/// preprocessor applied (uniform scale + symmetric padding) and
/// normalised to the frame, matching `Detection`'s contract.
pub struct YoloPostProcessor {
    model_width: u32,
    model_height: u32,
    num_classes: u32,
    score_threshold: f32,
    iou_threshold: f32,
}

impl YoloPostProcessor {
    pub fn new(
        model_width: u32,
        model_height: u32,
        num_classes: u32,
        score_threshold: f32,
    ) -> Self {
        Self {
            model_width,
            model_height,
            num_classes,
            score_threshold,
            iou_threshold: 0.45,
        }
    }

    pub fn with_iou_threshold(mut self, iou_threshold: f32) -> Self {
        self.iou_threshold = iou_threshold;
        self
    }
}

impl PostProcessor for YoloPostProcessor {
    fn decode(&self, outputs: &[Tensor], meta: &FrameMeta, out: &mut Detections) -> Result<()> {
        out.clear(meta.seq);
        let tensor = outputs
            .first()
            .ok_or_else(|| Error::Platform("yolo decoder: no output tensor".into()))?;
        let data = tensor.as_f32_slice().ok_or_else(|| {
            Error::Platform("yolo decoder: output tensor is not host-visible F32".into())
        })?;

        let channels = 4 + self.num_classes as usize;
        if tensor.spec.shape.len() != 3 || tensor.spec.shape[1] != channels {
            return Err(Error::Platform(format!(
                "yolo decoder: expected output shape [1, {channels}, N], got {:?}",
                tensor.spec.shape
            )));
        }
        let candidates = tensor.spec.shape[2];
        if data.len() < channels * candidates {
            return Err(Error::Platform(
                "yolo decoder: output tensor smaller than its spec".into(),
            ));
        }
        let at = |channel: usize, i: usize| data[channel * candidates + i];

        // Undo the letterbox: model = frame * scale + pad.
        let (fw, fh) = (meta.width as f32, meta.height as f32);
        let (scale, pad_x, pad_y) =
            letterbox_params(self.model_width, self.model_height, meta.width, meta.height);

        for i in 0..candidates {
            // Best class for this candidate.
            let (mut best_class, mut best_score) = (0u32, f32::MIN);
            for c in 0..self.num_classes as usize {
                let s = at(4 + c, i);
                if s > best_score {
                    best_score = s;
                    best_class = c as u32;
                }
            }
            if best_score < self.score_threshold {
                continue;
            }
            let (cx, cy, w, h) = (at(0, i), at(1, i), at(2, i), at(3, i));
            let x = ((cx - w / 2.0 - pad_x) / scale / fw).clamp(0.0, 1.0);
            let y = ((cy - h / 2.0 - pad_y) / scale / fh).clamp(0.0, 1.0);
            let width = ((w / scale) / fw).clamp(0.0, 1.0 - x);
            let height = ((h / scale) / fh).clamp(0.0, 1.0 - y);
            if width <= 0.0 || height <= 0.0 {
                continue;
            }
            out.push(Detection {
                x,
                y,
                width,
                height,
                score: best_score,
                class_id: best_class,
            });
        }

        nms(&mut out.items, self.iou_threshold);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::{DType, Layout, PixelFormat, TensorBuffer, TensorSpec};
    use std::any::Any;
    use std::sync::Arc;
    use std::time::Duration;

    struct TestBuffer(Vec<f32>);
    impl TensorBuffer for TestBuffer {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn as_f32_slice(&self) -> Option<&[f32]> {
            Some(&self.0)
        }
    }

    fn meta(width: u32, height: u32) -> FrameMeta {
        FrameMeta {
            seq: 7,
            pts: Duration::ZERO,
            width,
            height,
            format: PixelFormat::Nv12,
            layout: Layout::PitchLinear,
            camera_id: 0,
            dmabuf_fd: -1,
        }
    }

    /// Build a [1, 4+C, N] tensor from per-candidate rows
    /// `[cx, cy, w, h, class_scores...]`.
    fn tensor(rows: &[Vec<f32>], num_classes: usize) -> Tensor {
        let channels = 4 + num_classes;
        let n = rows.len();
        let mut data = vec![0.0f32; channels * n];
        for (i, row) in rows.iter().enumerate() {
            for (ch, v) in row.iter().enumerate() {
                data[ch * n + i] = *v;
            }
        }
        Tensor::new(
            TensorSpec::new("output0", vec![1, channels, n], DType::F32),
            Arc::new(TestBuffer(data)),
        )
    }

    #[test]
    fn decodes_box_and_best_class_without_letterbox_padding() {
        // Square frame into square model: no padding, scale 1.
        let pp = YoloPostProcessor::new(64, 64, 3, 0.25);
        let t = tensor(
            &[
                vec![32.0, 32.0, 16.0, 16.0, 0.1, 0.9, 0.2], // centre box, class 1
                vec![8.0, 8.0, 8.0, 8.0, 0.1, 0.1, 0.1],     // below threshold
            ],
            3,
        );
        let mut out = Detections::default();
        pp.decode(&[t], &meta(64, 64), &mut out).unwrap();
        assert_eq!(out.frame_seq, 7);
        assert_eq!(out.len(), 1);
        let d = out.items[0];
        assert_eq!(d.class_id, 1);
        assert!((d.score - 0.9).abs() < 1e-6);
        assert!((d.x - 0.375).abs() < 1e-6, "x={}", d.x);
        assert!((d.y - 0.375).abs() < 1e-6);
        assert!((d.width - 0.25).abs() < 1e-6);
        assert!((d.height - 0.25).abs() < 1e-6);
    }

    #[test]
    fn undoes_letterbox_padding_for_wide_frame() {
        // 128x64 frame into 64x64 model: scale 0.5, pad_y = 16.
        // Centre of frame = model (32, 32); a 32x16 frame-px box is
        // 16x8 model px. Expect normalised (0.375, 0.375, 0.25, 0.25).
        let pp = YoloPostProcessor::new(64, 64, 1, 0.25);
        let t = tensor(&[vec![32.0, 32.0, 16.0, 8.0, 0.8]], 1);
        let mut out = Detections::default();
        pp.decode(&[t], &meta(128, 64), &mut out).unwrap();
        assert_eq!(out.len(), 1);
        let d = out.items[0];
        assert!((d.x - 0.375).abs() < 1e-6, "x={}", d.x);
        assert!((d.y - 0.375).abs() < 1e-6, "y={}", d.y);
        assert!((d.width - 0.25).abs() < 1e-6);
        assert!((d.height - 0.25).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_only_best_of_overlapping_duplicates() {
        let pp = YoloPostProcessor::new(64, 64, 1, 0.25);
        let t = tensor(
            &[
                vec![32.0, 32.0, 16.0, 16.0, 0.9],
                vec![33.0, 32.0, 16.0, 16.0, 0.7],
            ],
            1,
        );
        let mut out = Detections::default();
        pp.decode(&[t], &meta(64, 64), &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out.items[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn rejects_wrong_output_shape() {
        let pp = YoloPostProcessor::new(64, 64, 80, 0.25);
        let t = tensor(&[vec![0.0, 0.0, 0.0, 0.0, 0.9]], 1);
        let mut out = Detections::default();
        assert!(pp.decode(&[t], &meta(64, 64), &mut out).is_err());
    }
}
