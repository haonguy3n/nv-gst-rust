"""Shared pieces of the golden-set tools: the device's letterbox and the app's YOLOX decode.

Mirrors crates/core/src/letterbox.rs (Letterbox::new) and crates/postproc/src/yolox.rs
(score = objectness * best class score, agnostic NMS, per-class NMS, top-K), so the PC and
the device decode the same way.
"""

import json
from pathlib import Path

import numpy as np

MODEL_W = 640
MODEL_H = 640
PAD_VALUE = 114
OUTPUTS = ("boxes", "objectness", "scores")
NUM_CLASSES = 80
APP_SCORE_THRESHOLD = 0.3  # config/orin-nano-yolox.toml
IOU_THRESHOLD = 0.45
TOP_K = 100
MAX_NMS = 300


def default_root() -> Path:
    return Path.home() / ".cache" / "nv-gst-rust"


def rust_round(x: float) -> int:
    """f32::round: half away from zero (Python's round is banker's)."""
    return int(np.floor(x + 0.5))


def letterbox_geometry(src_w: int, src_h: int) -> dict:
    """Letterbox::new: uniform scale, rounded resized size, integer half pads (centred)."""
    scale = np.float32(min(np.float32(MODEL_W) / np.float32(src_w), np.float32(MODEL_H) / np.float32(src_h)))
    rw = min(max(rust_round(float(np.float32(src_w) * scale)), 1), MODEL_W)
    rh = min(max(rust_round(float(np.float32(src_h) * scale)), 1), MODEL_H)
    return {
        "scale": float(scale),
        "resized_w": rw,
        "resized_h": rh,
        "pad_x": (MODEL_W - rw) // 2,
        "pad_y": (MODEL_H - rh) // 2,
    }


def load_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text())


def read_output(path: Path, shape: list[int]) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(shape)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of xyxy boxes a [N,4] against b [M,4]."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-30), 0.0)


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, agnostic: bool) -> np.ndarray:
    """Indices kept, highest score first (nms.rs: strict `>` threshold)."""
    order = np.argsort(-scores, kind="stable")
    boxes, classes = boxes[order], classes[order]
    ious = iou_matrix(boxes, boxes)
    keep = np.ones(len(order), dtype=bool)
    for i in range(len(order)):
        if not keep[i]:
            continue
        later = np.arange(i + 1, len(order))
        hit = keep[later] & (ious[i, later] > IOU_THRESHOLD)
        if not agnostic:
            hit &= classes[later] == classes[i]
        keep[later[hit]] = False
    return order[keep]


def decode(boxes, objectness, scores, geom: dict, src_w: int, src_h: int, threshold: float) -> np.ndarray:
    """The app's YoloxPostProcessor. Returns rows [x, y, w, h (source pixels), score, class]."""
    boxes = boxes.reshape(-1, 4).astype(np.float32)
    obj = objectness.reshape(boxes.shape[0], -1)[:, 0].astype(np.float32)
    cls_scores = scores.reshape(boxes.shape[0], -1)
    best = cls_scores.argmax(axis=1)
    score = obj * cls_scores[np.arange(len(best)), best]
    ok = ~np.isnan(score) & (score >= threshold) & np.isfinite(boxes).all(axis=1)
    boxes, score, best = boxes[ok], score[ok], best[ok]
    scale, px, py = np.float32(geom["scale"]), np.float32(geom["pad_x"]), np.float32(geom["pad_y"])
    fw, fh = np.float32(src_w), np.float32(src_h)
    x = np.clip((boxes[:, 0] - px) / scale / fw, 0, 1)
    y = np.clip((boxes[:, 1] - py) / scale / fh, 0, 1)
    w = np.clip((boxes[:, 2] - px) / scale / fw - x, 0, 1 - x)
    h = np.clip((boxes[:, 3] - py) / scale / fh - y, 0, 1 - y)
    good = (w > 0) & (h > 0)
    x, y, w, h, score, best = x[good], y[good], w[good], h[good], score[good], best[good]
    if len(score) > MAX_NMS:
        top = np.argsort(-score, kind="stable")[:MAX_NMS]
        x, y, w, h, score, best = x[top], y[top], w[top], h[top], score[top], best[top]
    xyxy = np.stack([x, y, x + w, y + h], axis=1)
    keep = greedy_nms(xyxy, score, best, agnostic=True)
    xyxy, score, best, x, y, w, h = (a[keep] for a in (xyxy, score, best, x, y, w, h))
    keep = greedy_nms(xyxy, score, best, agnostic=False)[:TOP_K]
    return np.stack([x[keep] * fw, y[keep] * fh, w[keep] * fw, h[keep] * fh, score[keep], best[keep]], axis=1)
