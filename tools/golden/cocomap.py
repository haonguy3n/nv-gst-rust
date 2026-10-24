"""COCO-style AP in numpy (pycocotools' COCOeval for bbox, area "all", maxDets 100).

evaluate(gt, dets, cat_ids) -> {"map": AP@[.5:.95], "ap50": AP@.5, "ap75": AP@.75}
  gt:   {image_id: [(category_id, x, y, w, h, iscrowd), ...]}
  dets: [(image_id, category_id, score, x, y, w, h), ...]
Categories without ground truth are left out of the mean, as pycocotools does.
"""

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100


def _iou(dt: np.ndarray, gt: np.ndarray, crowd: np.ndarray) -> np.ndarray:
    """xywh boxes; against a crowd GT the union is the detection's area (maskUtils.iou)."""
    ix1 = np.maximum(dt[:, None, 0], gt[None, :, 0])
    iy1 = np.maximum(dt[:, None, 1], gt[None, :, 1])
    ix2 = np.minimum(dt[:, None, 0] + dt[:, None, 2], gt[None, :, 0] + gt[None, :, 2])
    iy2 = np.minimum(dt[:, None, 1] + dt[:, None, 3], gt[None, :, 1] + gt[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a_dt = (dt[:, 2] * dt[:, 3])[:, None]
    a_gt = (gt[:, 2] * gt[:, 3])[None, :]
    union = np.where(crowd[None, :], a_dt, a_dt + a_gt - inter)
    return inter / np.maximum(union, 1e-12)


def _match_image(dt, dt_scores, gt, crowd):
    """evaluateImg for one (image, category): (dt_matched [T,D], dt_ignored [T,D], dt_scores, n_gt)."""
    order = np.argsort(-dt_scores, kind="stable")[:MAX_DETS]
    dt, dt_scores = dt[order], dt_scores[order]
    # Non-crowd ground truths first, as pycocotools sorts by the ignore flag.
    gorder = np.argsort(crowd, kind="stable")
    gt, crowd = gt[gorder], crowd[gorder]
    n_t, n_d = len(IOU_THRESHOLDS), len(dt)
    matched = np.zeros((n_t, n_d), dtype=bool)
    ignored = np.zeros((n_t, n_d), dtype=bool)
    if len(gt) and n_d:
        ious = _iou(dt, gt, crowd.astype(bool))
        for ti, t in enumerate(IOU_THRESHOLDS):
            taken = np.zeros(len(gt), dtype=bool)
            for di in range(n_d):
                best, m = min(t, 1 - 1e-10), -1
                for gi in range(len(gt)):
                    if taken[gi] and not crowd[gi]:
                        continue
                    if m > -1 and not crowd[m] and crowd[gi]:
                        break
                    if ious[di, gi] < best:
                        continue
                    best, m = ious[di, gi], gi
                if m == -1:
                    continue
                ignored[ti, di] = bool(crowd[m])
                matched[ti, di] = True
                taken[m] = True
    return matched, ignored, dt_scores, int((crowd == 0).sum())


def _average_precision(matched, ignored, scores, n_gt):
    """accumulate() for one category: AP per IoU threshold (101-point interpolation)."""
    order = np.argsort(-scores, kind="stable")
    ap = np.zeros(len(IOU_THRESHOLDS))
    for ti in range(len(IOU_THRESHOLDS)):
        m, ig = matched[ti][order], ignored[ti][order]
        keep = ~ig
        tp = np.cumsum(m[keep] & True).astype(float)
        fp = np.cumsum(~m[keep]).astype(float)
        if len(tp) == 0:
            continue
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, np.spacing(1))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        idx = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
        q = np.zeros(len(RECALL_THRESHOLDS))
        valid = idx < len(precision)
        q[valid] = precision[idx[valid]]
        ap[ti] = q.mean()
    return ap


def evaluate(gt: dict, dets: list, cat_ids: list) -> dict:
    gt_by = {}
    for image_id, rows in gt.items():
        for cat, x, y, w, h, crowd in rows:
            gt_by.setdefault((image_id, cat), []).append((x, y, w, h, crowd))
    dt_by = {}
    for image_id, cat, score, x, y, w, h in dets:
        dt_by.setdefault((image_id, cat), []).append((x, y, w, h, score))
    per_cat = []
    for cat in cat_ids:
        parts, n_gt = [], 0
        for image_id in gt:
            g = np.array(gt_by.get((image_id, cat), []), dtype=float).reshape(-1, 5)
            d = np.array(dt_by.get((image_id, cat), []), dtype=float).reshape(-1, 5)
            if len(g) == 0 and len(d) == 0:
                continue
            res = _match_image(d[:, :4], d[:, 4], g[:, :4], g[:, 4])
            parts.append(res)
            n_gt += res[3]
        if n_gt == 0:
            continue
        matched = np.concatenate([p[0] for p in parts], axis=1)
        ignored = np.concatenate([p[1] for p in parts], axis=1)
        scores = np.concatenate([p[2] for p in parts])
        per_cat.append(_average_precision(matched, ignored, scores, n_gt))
    if not per_cat:
        return {"map": 0.0, "ap50": 0.0, "ap75": 0.0}
    ap = np.mean(per_cat, axis=0)
    return {"map": float(ap.mean()), "ap50": float(ap[0]), "ap75": float(ap[5])}
