use crate::nms::{nms, nms_agnostic};
use crate::yolo::letterbox_params;
use edge_core::{Detection, Detections, Error, FrameMeta, PostProcessor, Result, Tensor};

/// Decoder for YOLOX exports with the box decode already in the
/// graph (e.g. Megvii's official ONNX re-exported by the reference
/// products' model importer). Three outputs, selected by name when
/// present, else positionally:
///
/// - `boxes` `[1, A, 4]`: x1, y1, x2, y2 in model-input pixels
/// - `objectness` `[1, A, 1]` (some exports repeat it per row)
/// - `scores` `[1, A, C]`: per-class scores
///
/// Final score is objectness × best class score. Suppression runs
/// class-agnostic first, then per-class (both at `iou_threshold`
/// here), then top-K — the same shape as the reference decoder.
pub struct YoloxPostProcessor {
    model_width: u32,
    model_height: u32,
    num_classes: u32,
    score_threshold: f32,
    iou_threshold: f32,
    top_k: usize,
    max_nms: usize,
    scratch: Vec<bool>,
}

impl YoloxPostProcessor {
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
            top_k: 100,
            max_nms: 300,
            scratch: Vec::new(),
        }
    }
}

fn pick<'a>(outputs: &'a [Tensor], name: &str, index: usize) -> Option<&'a Tensor> {
    outputs
        .iter()
        .find(|t| t.spec.name == name)
        .or_else(|| outputs.get(index))
}

impl PostProcessor for YoloxPostProcessor {
    fn decode(&mut self, outputs: &[Tensor], meta: &FrameMeta, out: &mut Detections) -> Result<()> {
        out.clear(meta.seq);
        let missing = |what: &str| Error::Platform(format!("yolox decoder: missing {what}"));
        let boxes_t = pick(outputs, "boxes", 0).ok_or_else(|| missing("boxes output"))?;
        let obj_t = pick(outputs, "objectness", 1).ok_or_else(|| missing("objectness output"))?;
        let scores_t = pick(outputs, "scores", 2).ok_or_else(|| missing("scores output"))?;

        fn host<'a>(t: &'a Tensor, what: &str) -> Result<&'a [f32]> {
            t.as_f32_slice().ok_or_else(|| {
                Error::Platform(format!("yolox decoder: {what} is not host-visible F32"))
            })
        }
        let boxes = host(boxes_t, "boxes")?;
        let objectness = host(obj_t, "objectness")?;
        let scores = host(scores_t, "scores")?;

        let classes = self.num_classes as usize;
        if scores_t.spec.shape.len() != 3 || scores_t.spec.shape[2] != classes {
            return Err(Error::Platform(format!(
                "yolox decoder: expected scores shape [1, A, {classes}], got {:?}",
                scores_t.spec.shape
            )));
        }
        let anchors = scores_t.spec.shape[1];
        // Objectness may be [1, A, 1] or repeated K times per row;
        // element 0 of each row is the value either way.
        let obj_stride = if anchors == 0 {
            1
        } else {
            objectness.len() / anchors.max(1)
        };
        if boxes.len() < anchors * 4 || scores.len() < anchors * classes || obj_stride == 0 {
            return Err(Error::Platform(
                "yolox decoder: output tensors smaller than their specs".into(),
            ));
        }

        let (fw, fh) = (meta.width as f32, meta.height as f32);
        let (scale, pad_x, pad_y) =
            letterbox_params(self.model_width, self.model_height, meta.width, meta.height);

        for a in 0..anchors {
            let obj = objectness[a * obj_stride];
            let (mut best_class, mut best_score) = (0u32, f32::MIN);
            for c in 0..classes {
                let s = scores[a * classes + c];
                if s > best_score {
                    best_score = s;
                    best_class = c as u32;
                }
            }
            let score = obj * best_score;
            if score < self.score_threshold {
                continue;
            }
            let (x1, y1, x2, y2) = (
                boxes[a * 4],
                boxes[a * 4 + 1],
                boxes[a * 4 + 2],
                boxes[a * 4 + 3],
            );
            let x = ((x1 - pad_x) / scale / fw).clamp(0.0, 1.0);
            let y = ((y1 - pad_y) / scale / fh).clamp(0.0, 1.0);
            let width = (((x2 - pad_x) / scale / fw) - x).clamp(0.0, 1.0 - x);
            let height = (((y2 - pad_y) / scale / fh) - y).clamp(0.0, 1.0 - y);
            if width <= 0.0 || height <= 0.0 {
                continue;
            }
            out.push(Detection {
                x,
                y,
                width,
                height,
                score,
                class_id: best_class,
            });
        }

        if out.items.len() > self.max_nms {
            out.items
                .sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
            out.items.truncate(self.max_nms);
        }
        nms_agnostic(&mut out.items, self.iou_threshold, &mut self.scratch);
        nms(&mut out.items, self.iou_threshold, &mut self.scratch);
        out.items.truncate(self.top_k);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::{DType, Layout, PixelFormat, TensorBuffer, TensorSpec};
    use std::any::Any;
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

    fn tensor(name: &str, shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        Tensor::new(
            TensorSpec::new(name, shape, DType::F32),
            Box::new(TestBuffer(data)),
        )
    }

    fn meta() -> FrameMeta {
        FrameMeta {
            seq: 3,
            pts: Duration::ZERO,
            width: 416,
            height: 416,
            format: PixelFormat::Nv12,
            layout: Layout::PitchLinear,
            camera_id: 0,
            dmabuf_fd: -1,
        }
    }

    #[test]
    fn decodes_xyxy_with_objectness_product() {
        // Model and frame both 416x416: scale 1, no padding.
        let mut pp = YoloxPostProcessor::new(416, 416, 3, 0.3);
        let outputs = vec![
            tensor(
                "boxes",
                vec![1, 2, 4],
                vec![104.0, 104.0, 312.0, 312.0, 0.0, 0.0, 50.0, 50.0],
            ),
            tensor("objectness", vec![1, 2, 1], vec![0.9, 0.9]),
            tensor(
                "scores",
                vec![1, 2, 3],
                vec![
                    0.1, 0.8, 0.1, // 0.72 >= 0.3 keep, class 1
                    0.2, 0.2, 0.2,
                ], // 0.18 < 0.3 drop
            ),
        ];
        let mut out = Detections::default();
        pp.decode(&outputs, &meta(), &mut out).unwrap();
        assert_eq!(out.len(), 1);
        let d = out.items[0];
        assert_eq!(d.class_id, 1);
        assert!((d.score - 0.72).abs() < 1e-6, "score {}", d.score);
        assert!((d.x - 0.25).abs() < 1e-6 && (d.y - 0.25).abs() < 1e-6);
        assert!((d.width - 0.5).abs() < 1e-6 && (d.height - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_wrong_class_count() {
        let mut pp = YoloxPostProcessor::new(416, 416, 80, 0.3);
        let outputs = vec![
            tensor("boxes", vec![1, 1, 4], vec![0.0; 4]),
            tensor("objectness", vec![1, 1, 1], vec![0.9]),
            tensor("scores", vec![1, 1, 3], vec![0.9; 3]),
        ];
        let mut out = Detections::default();
        assert!(pp.decode(&outputs, &meta(), &mut out).is_err());
    }
}
