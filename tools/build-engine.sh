#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX model ON THE TARGET DEVICE.
#
# A TensorRT .engine only runs on the same TensorRT version and GPU
# it was built for, so we ship the ONNX model and build the engine
# on the device at first boot. The output is cached under
# models/cache/ keyed by (model sha256 + trtexec version): if that
# engine already exists it is reused and its path is printed.
#
# Usage: tools/build-engine.sh <model.onnx> [extra trtexec args...]
# Prints the engine path on stdout (all chatter goes to stderr).
set -euo pipefail

die() { echo "build-engine: $*" >&2; exit 2; }

[ $# -ge 1 ] || die "usage: $0 <model.onnx> [extra trtexec args...]"
model=$1
shift

[ -f "$model" ] || die "model not found: $model"
command -v trtexec >/dev/null 2>&1 \
    || die "trtexec not found in PATH — run this on the Jetson (JetPack) target, not the x86 host"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cache_dir="$repo_root/models/cache"
mkdir -p "$cache_dir"

model_hash=$(sha256sum "$model" | cut -c1-16)
trt_version=$(trtexec --help 2>&1 | grep -oE 'TensorRT v[0-9.]+' | head -n1 | grep -oE '[0-9.]+' || true)
[ -n "$trt_version" ] || die "could not determine TensorRT version from trtexec --help"

name=$(basename "$model" .onnx)
engine="$cache_dir/${name}-${model_hash}-trt${trt_version}.engine"

if [ -f "$engine" ]; then
    echo "build-engine: cache hit: $engine" >&2
    echo "$engine"
    exit 0
fi

echo "build-engine: building $engine (this runs on-device and can take minutes)" >&2
trtexec --onnx="$model" --saveEngine="$engine.tmp" --fp16 "$@" >&2
mv "$engine.tmp" "$engine"
echo "$engine"
