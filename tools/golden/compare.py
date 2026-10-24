#!/usr/bin/env python3
"""Step 4: compare TensorRT FP16 outputs (trt/) with ONNX Runtime FP32 (ort/); exit 1 if a gate fails.

Numeric: per output, max and worst-image p99 of |trt - ort|. Gates:
  objectness, scores: max <= 1e-2 on the fixed frame, worst-image p99 <= 1e-2 on the set;
  boxes: pixels (model input), judged on the anchors that become detections (ORT score >= app
    threshold), p99 <= 1.0 px per image and max <= 1.0 px on the fixed frame. FP16 has 11
    significant bits, so a coordinate of 512..640 moves in 0.5 px steps: 1 px is two steps.
    All-anchor box stats are printed too (junk anchors can have huge, harmless errors).
mAP (COCO val2017 boxes, area all, maxDets 100), decoded exactly like the app: for ORT, for TRT,
  and TRT against ORT's detections as ground truth. Gate: |mAP(TRT) - mAP(ORT)| <= 0.01, at the
  app's score threshold and at a low one (0.01, the usual way to measure AP).
--self-test replaces trt/ with ORT + small noise (nothing from the device is needed).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cocomap import evaluate
from golden_common import APP_SCORE_THRESHOLD, decode, default_root, load_manifest, read_output

SCORE_GATE = 1e-2
BOX_GATE_PX = 1.0
MAP_GATE = 0.01
LOW_THRESHOLD = 0.01
NOISE = {"scores": 1e-3, "objectness": 1e-3, "boxes": 0.1}


def abs_stats(d: np.ndarray) -> tuple[float, float]:
    return (float(d.max()), float(np.percentile(d, 99))) if d.size else (0.0, 0.0)


def load_pair(root: Path, trt_dir: Path, entry: dict, shapes: dict, self_test: bool) -> tuple[dict, dict]:
    ort, trt = {}, {}
    rng = np.random.default_rng(entry["id"])
    for name, shape in shapes.items():
        ort[name] = read_output(root / "ort" / f"{entry['id']}.{name}.bin", shape)
        if self_test:
            noisy = ort[name] + rng.normal(0, NOISE[name], shape).astype(np.float32)
            trt[name] = np.clip(noisy, 0, None) if name != "boxes" else noisy
        else:
            path = trt_dir / f"{entry['id']}.{name}.bin"
            if not path.exists():
                raise SystemExit(f"missing {path}: run the device step (README.md) first")
            trt[name] = read_output(path, shape)
    return ort, trt


def numeric(ort: dict, trt: dict) -> dict:
    """Per-image stats: {name: (max, p99)} plus boxes_conf (confident anchors only)."""
    res = {n: abs_stats(np.abs(trt[n] - ort[n])) for n in ort}
    conf = ort["objectness"].reshape(-1) * ort["scores"].reshape(-1, ort["scores"].shape[-1]).max(axis=1)
    sel = conf >= APP_SCORE_THRESHOLD
    res["boxes_conf"] = abs_stats(np.abs(trt["boxes"].reshape(-1, 4)[sel] - ort["boxes"].reshape(-1, 4)[sel]))
    res["n_conf"] = int(sel.sum())
    return res


def to_coco(image_id: int, rows: np.ndarray, cat_ids: list) -> list:
    return [(image_id, cat_ids[int(c)], float(s), float(x), float(y), float(w), float(h)) for x, y, w, h, s, c in rows]


def load_gt(manifest: dict) -> tuple[dict, list]:
    ann = json.loads(Path(manifest["annotations"]).read_text())
    ids = {e["id"] for e in manifest["images"]}
    gt = {i: [] for i in ids}
    for a in ann["annotations"]:
        if a["image_id"] in ids:
            gt[a["image_id"]].append((a["category_id"], *a["bbox"], a.get("iscrowd", 0)))
    return gt, sorted(c["id"] for c in ann["categories"])


def summarize(per_image: dict, fixed_id: int) -> dict:
    out = {}
    for key in ("boxes", "boxes_conf", "objectness", "scores"):
        stats = np.array([v[key] for v in per_image.values()])
        out[key] = {
            "max": float(stats[:, 0].max()),
            "p99_worst_image": float(stats[:, 1].max()),
            "fixed_frame_max": float(per_image[fixed_id][key][0]),
        }
    return out


def gates(num: dict, maps: dict) -> list[tuple[str, bool, str]]:
    g = []
    for key in ("objectness", "scores"):
        g.append(
            (
                f"{key} fixed-frame max <= {SCORE_GATE}",
                num[key]["fixed_frame_max"] <= SCORE_GATE,
                f"{num[key]['fixed_frame_max']:.2e}",
            )
        )
        g.append(
            (
                f"{key} p99 (worst image) <= {SCORE_GATE}",
                num[key]["p99_worst_image"] <= SCORE_GATE,
                f"{num[key]['p99_worst_image']:.2e}",
            )
        )
    b = num["boxes_conf"]
    g.append(
        (
            f"boxes (confident) fixed-frame max <= {BOX_GATE_PX} px",
            b["fixed_frame_max"] <= BOX_GATE_PX,
            f"{b['fixed_frame_max']:.3f} px",
        )
    )
    g.append(
        (
            f"boxes (confident) p99 (worst image) <= {BOX_GATE_PX} px",
            b["p99_worst_image"] <= BOX_GATE_PX,
            f"{b['p99_worst_image']:.3f} px",
        )
    )
    for label, m in maps.items():
        d = abs(m["trt"]["map"] - m["ort"]["map"])
        g.append((f"mAP |TRT - ORT| <= {MAP_GATE} @ score {label}", d <= MAP_GATE, f"{d:.4f}"))
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=default_root() / "golden")
    ap.add_argument("--trt", type=Path, help="TensorRT outputs (default <root>/trt)")
    ap.add_argument("--self-test", action="store_true", help="TRT := ORT + small noise")
    args = ap.parse_args()
    manifest = load_manifest(args.root)
    trt_dir = args.trt or args.root / "trt"
    shapes = manifest["output_shapes"]
    gt, cat_ids = load_gt(manifest)
    thresholds = {"app": APP_SCORE_THRESHOLD, "low": LOW_THRESHOLD}
    dets = {(who, t): [] for who in ("ort", "trt") for t in thresholds}
    per_image = {}
    for entry in manifest["images"]:
        ort, trt = load_pair(args.root, trt_dir, entry, shapes, args.self_test)
        per_image[entry["id"]] = numeric(ort, trt)
        for who, out in (("ort", ort), ("trt", trt)):
            for t, thr in thresholds.items():
                rows = decode(
                    out["boxes"], out["objectness"], out["scores"], entry, entry["width"], entry["height"], thr
                )
                dets[(who, t)] += to_coco(entry["id"], rows, cat_ids)
    num = summarize(per_image, manifest["fixed_id"])
    num["fixed_frame_confident_anchors"] = per_image[manifest["fixed_id"]]["n_conf"]
    maps = {}
    for t in thresholds:
        ref_gt = {i: [] for i in gt}
        if t == "app":
            for d in dets[("ort", t)]:
                ref_gt[d[0]].append((d[1], *d[3:], 0))
        maps[t] = {
            "ort": evaluate(gt, dets[("ort", t)], cat_ids),
            "trt": evaluate(gt, dets[("trt", t)], cat_ids),
            "trt_vs_ort_dets": evaluate(ref_gt, dets[("trt", t)], cat_ids) if t == "app" else None,
        }
    results = gates(num, maps)
    report = {
        "images": len(manifest["images"]),
        "self_test": args.self_test,
        "numeric": num,
        "map": maps,
        "gates": [{"gate": n, "pass": ok, "value": v} for n, ok, v in results],
    }
    (args.root / ("report-selftest.json" if args.self_test else "report.json")).write_text(json.dumps(report, indent=1))
    print_table(report, results, thresholds)
    return 0 if all(ok for _, ok, _ in results) else 1


def print_table(report: dict, results: list, thresholds: dict) -> None:
    print(f"images: {report['images']}{'  (SELF-TEST: TRT = ORT + noise)' if report['self_test'] else ''}")
    print(f"{'output':<22}{'max':>12}{'p99 worst img':>16}{'fixed frame max':>18}")
    for key, v in report["numeric"].items():
        if isinstance(v, dict):
            unit = "px" if key.startswith("boxes") else ""
            print(f"{key:<22}{v['max']:>12.3e}{v['p99_worst_image']:>16.3e}{v['fixed_frame_max']:>18.3e} {unit}")
    n_conf = report["numeric"]["fixed_frame_confident_anchors"]
    print(f"(boxes_conf: anchors with ORT score >= {APP_SCORE_THRESHOLD}; {n_conf} on the fixed frame)")
    print(f"\n{'mAP (COCO)':<28}{'mAP@[.5:.95]':>14}{'AP50':>10}")
    for t, thr in thresholds.items():
        m = report["map"][t]
        rows = [("ORT FP32", m["ort"]), ("TRT FP16", m["trt"])]
        if m["trt_vs_ort_dets"]:
            rows.append(("TRT vs ORT detections", m["trt_vs_ort_dets"]))
        for label, r in rows:
            print(f"{label + f' @score {thr}':<28}{r['map']:>14.4f}{r['ap50']:>10.4f}")
    print()
    for name, ok, value in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {value}")


if __name__ == "__main__":
    sys.exit(main())
