#!/usr/bin/env python3
"""Step 1: download COCO val2017 (cached) and write the letterboxed input tensors.

Output (default ~/.cache/nv-gst-rust/golden): input/<id>.bin (f32 LE, NCHW, BGR, 0..255)
and manifest.json. Letterbox geometry is the device's (golden_common.letterbox_geometry).
"""

import argparse
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from golden_common import MODEL_H, MODEL_W, OUTPUTS, PAD_VALUE, default_root, letterbox_geometry

BASE = "http://images.cocodataset.org"
ANNOTATIONS_ZIP = "annotations_trainval2017.zip"


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def annotations(cache: Path) -> Path:
    json_path = cache / "ann" / "annotations" / "instances_val2017.json"
    if not json_path.exists():
        import zipfile

        zip_path = cache / ANNOTATIONS_ZIP
        fetch(f"{BASE}/annotations/{ANNOTATIONS_ZIP}", zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extract("annotations/instances_val2017.json", cache / "ann")
    return json_path


def letterbox(img_bgr: np.ndarray, geom: dict) -> np.ndarray:
    """Planar f32 CHW BGR, resized bilinear into the centred rectangle, pad 114 elsewhere."""
    resized = cv2.resize(img_bgr, (geom["resized_w"], geom["resized_h"]), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((MODEL_H, MODEL_W, 3), PAD_VALUE, dtype=np.uint8)
    px, py = geom["pad_x"], geom["pad_y"]
    canvas[py : py + geom["resized_h"], px : px + geom["resized_w"]] = resized
    return np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype="<f4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--count", type=int, default=500, help="first N images of val2017 by id")
    ap.add_argument("--fixed-id", type=int, help="the strict-check frame (default: the first image)")
    args = ap.parse_args()

    coco = args.root / "coco"
    out = args.root / "golden"
    ann_path = annotations(coco)
    ann = json.loads(ann_path.read_text())
    images = sorted(ann["images"], key=lambda i: i["id"])[: args.count]
    fixed = args.fixed_id if args.fixed_id is not None else images[0]["id"]
    if fixed not in {i["id"] for i in images}:
        raise SystemExit(f"--fixed-id {fixed} is not among the first {args.count} images")

    def one(info: dict) -> dict:
        jpg = coco / "val2017" / info["file_name"]
        fetch(f"{BASE}/val2017/{info['file_name']}", jpg)
        img = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (info["height"], info["width"]):
            raise RuntimeError(f"{jpg}: unreadable or size differs from the annotations")
        geom = letterbox_geometry(info["width"], info["height"])
        name = f"input/{info['id']}.bin"
        (out / "input").mkdir(parents=True, exist_ok=True)
        letterbox(img, geom).tofile(out / name)
        return {"id": info["id"], "file": name, "width": info["width"], "height": info["height"], **geom}

    with ThreadPoolExecutor(8) as pool:
        entries = list(pool.map(one, images))
    manifest = {
        "model": {"input": "images", "shape": [1, 3, MODEL_H, MODEL_W], "color": "bgr", "pad": PAD_VALUE},
        "outputs": list(OUTPUTS),
        "output_shapes": {"boxes": [1, 8400, 4], "objectness": [1, 8400, 1], "scores": [1, 8400, 80]},
        "annotations": str(ann_path),
        "fixed_id": fixed,
        "images": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(entries)} inputs in {out / 'input'}, fixed frame {fixed}")


if __name__ == "__main__":
    main()
