#!/usr/bin/env python3
"""Generate the tiny ONNX fixtures used by platform-sim tests.

`relu.onnx` is `Y = Relu(X)` with X/Y of shape [1, 3, 4, 4] float32:
enough to prove the Rust ONNX Runtime engine loads a model, runs it,
and returns correct output values.

`yolo-const.onnx` emits a constant YOLO-shaped head output
[1, 4 + 3 classes, 4 candidates] for a 64x64 model input: one strong
class-1 detection, one overlapping duplicate (NMS must suppress it),
one below-threshold candidate, one empty. Used for an end-to-end
pipeline run with `decoder = "yolo"`.

Regenerate with:
    source .venv/bin/activate
    pip install onnx
    python tools/make-test-model.py
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

out = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "platform-sim"
    / "tests"
    / "fixtures"
    / "relu.onnx"
)
out.parent.mkdir(parents=True, exist_ok=True)

x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 4, 4])
y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
node = helper.make_node("Relu", inputs=["images"], outputs=["output"])
graph = helper.make_graph([node], "relu-test", [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, out)
print(f"wrote {out} ({out.stat().st_size} bytes)")

# --- yolo-const.onnx -------------------------------------------------------
yolo_out = out.with_name("yolo-const.onnx")
# Channel-major [1, 7, 4]: rows are cx, cy, w, h, class0, class1, class2.
head = np.array(
    [
        [32.0, 33.0, 8.0, 0.0],  # cx
        [32.0, 32.0, 8.0, 0.0],  # cy
        [16.0, 16.0, 8.0, 0.0],  # w
        [16.0, 16.0, 8.0, 0.0],  # h
        [0.10, 0.10, 0.10, 0.0],  # class 0
        [0.90, 0.70, 0.10, 0.0],  # class 1
        [0.20, 0.20, 0.10, 0.0],  # class 2
    ],
    dtype=np.float32,
).reshape(1, 7, 4)

x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 7, 4])
const = helper.make_node(
    "Constant", [], ["output0"], value=numpy_helper.from_array(head, "head")
)
graph = helper.make_graph([const], "yolo-const", [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.save(model, yolo_out)
print(f"wrote {yolo_out} ({yolo_out.stat().st_size} bytes)")
