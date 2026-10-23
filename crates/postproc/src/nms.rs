use edge_core::Detection;

/// Intersection-over-union of two boxes in normalised top-left form.
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.width * a.height + b.width * b.height - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Per-class greedy NMS, in place. Keeps the highest-score boxes.
/// `scratch` is caller-owned working space, reused across frames
/// so steady-state NMS allocates nothing.
pub fn nms(detections: &mut Vec<Detection>, iou_threshold: f32, scratch: &mut Vec<bool>) {
    nms_inner(detections, iou_threshold, false, scratch);
}

/// Class-agnostic greedy NMS: overlapping boxes suppress each
/// other regardless of class (YOLOX exports run this first).
pub fn nms_agnostic(detections: &mut Vec<Detection>, iou_threshold: f32, scratch: &mut Vec<bool>) {
    nms_inner(detections, iou_threshold, true, scratch);
}

fn nms_inner(
    detections: &mut Vec<Detection>,
    iou_threshold: f32,
    agnostic: bool,
    keep: &mut Vec<bool>,
) {
    // Unstable sort: in place, no allocation; equal scores may
    // swap order, which NMS does not care about.
    detections.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
    keep.clear();
    keep.resize(detections.len(), true);
    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if keep[j]
                && (agnostic || detections[j].class_id == detections[i].class_id)
                && iou(&detections[i], &detections[j]) > iou_threshold
            {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, score: f32, class_id: u32) -> Detection {
        Detection {
            x,
            y,
            width: w,
            height: h,
            score,
            class_id,
        }
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let a = det(0.1, 0.1, 0.4, 0.4, 0.9, 0);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        let a = det(0.0, 0.0, 0.2, 0.2, 0.9, 0);
        let b = det(0.5, 0.5, 0.2, 0.2, 0.8, 0);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn nms_suppresses_lower_score_duplicate_of_same_class_only() {
        let mut dets = vec![
            det(0.10, 0.10, 0.30, 0.30, 0.90, 1),
            det(0.11, 0.10, 0.30, 0.30, 0.80, 1), // duplicate, suppressed
            det(0.10, 0.10, 0.30, 0.30, 0.70, 2), // same box, other class: kept
            det(0.60, 0.60, 0.20, 0.20, 0.60, 1), // disjoint: kept
        ];
        nms(&mut dets, 0.45, &mut Vec::new());
        assert_eq!(dets.len(), 3);
        assert_eq!(dets[0].score, 0.90);
        assert!(dets.iter().all(|d| d.score != 0.80));
    }
}
