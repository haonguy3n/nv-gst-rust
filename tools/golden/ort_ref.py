#!/usr/bin/env python3
"""Step 2: ONNX Runtime FP32 (CPU) on every golden input; writes ort/<id>.<output>.bin."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from golden_common import default_root, load_manifest

DEFAULT_ONNX = "<camera-viber>/embedded/test-data/models/yolox-s/1.0.0/model.onnx"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help=f"YOLOX-S with boxes/objectness/scores outputs (camera-viber: {DEFAULT_ONNX})",
    )
    ap.add_argument("--root", type=Path, default=default_root() / "golden")
    args = ap.parse_args()

    manifest = load_manifest(args.root)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    if names != manifest["outputs"]:
        raise SystemExit(f"model outputs {names} differ from the manifest's {manifest['outputs']}")
    (args.root / "ort").mkdir(exist_ok=True)
    for n, entry in enumerate(manifest["images"], 1):
        x = np.fromfile(args.root / entry["file"], dtype="<f4").reshape(manifest["model"]["shape"])
        for name, arr in zip(names, sess.run(names, {manifest["model"]["input"]: x})):
            np.ascontiguousarray(arr, dtype="<f4").tofile(args.root / "ort" / f"{entry['id']}.{name}.bin")
        if n % 50 == 0:
            print(f"{n}/{len(manifest['images'])}")
    print(f"ORT FP32 outputs for {len(manifest['images'])} images in {args.root / 'ort'}")


if __name__ == "__main__":
    main()
