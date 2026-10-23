#!/usr/bin/env python3
"""Generate the tiny ONNX fixture used by platform-sim's ONNX engine test.

The model is `Y = Relu(X)` with X/Y of shape [1, 3, 4, 4] float32:
small, deterministic, and enough to prove the Rust ONNX Runtime
engine loads a model, runs it, and returns correct output values.

Regenerate with:
    source .venv/bin/activate
    pip install onnx
    python tools/make-test-model.py
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

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
