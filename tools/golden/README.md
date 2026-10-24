# Golden set: TensorRT FP16 against ONNX Runtime FP32 (T11)

COCO val2017 (first 500 images by id) through the camera model YOLOX-S
(`boxes` / `objectness` / `scores`, 640x640, BGR, 0..255, letterboxed centred with pad 114 like
`edge_core::Letterbox`). Everything is cached under `~/.cache/nv-gst-rust/` (never in the repo).
Python: a venv with numpy, onnxruntime, opencv-python-headless (and pytest for the unit test).

```sh
PY=~/.cache/nv-gst-rust/venv/bin/python
# 1. PC: download, letterbox, write input/<id>.bin and manifest.json (~2.3 GB for 500 images)
$PY tools/golden/prepare.py [--count 500] [--fixed-id 139]
# 2. PC: ONNX Runtime FP32 reference -> ort/<id>.<output>.bin
$PY tools/golden/ort_ref.py --onnx <camera-viber>/embedded/test-data/models/yolox-s/1.0.0/model.onnx
# 3. device: copy the golden dir (manifest.json + input/) over, run trt-golden, copy trt/ back
# 4. PC: compare (exit 1 if a gate fails; writes report.json)
$PY tools/golden/compare.py
$PY tools/golden/compare.py --self-test     # TRT := ORT + noise, no device needed
$PY -m pytest tools/golden                  # mAP unit test
```

Device step (builds or loads the FP16 plan through the app's own warm-up code; `--limit N` runs
the first N images; the fixed frame is the first image, so a small limit keeps it):

```sh
cargo run --release -p platform-jetson --features trt --example trt-golden -- \
    --onnx /opt/edge-vision/models/yolox-s.onnx --cache /var/cache/edge-vision \
    --in /data/golden --out /data/golden
```

Gates (see `compare.py` for the reasoning): objectness and scores within 1e-2 (max on the fixed
frame, p99 of the worst image on the set); boxes within 1 px on the anchors that become
detections; mAP of TRT within 0.01 of ORT, at the app's score threshold (0.3) and at 0.01.
