#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX model ON THE TARGET DEVICE.
#
# BRING-UP TOOL ONLY. The product builds its plan in-process through
# `trt-sys` (design D5, `plan_cache.rs`) because the camera image does
# not ship `trtexec` and the service sandbox will not let it run a shell.
# This script is for a development image / first board session.
#
# A TensorRT .engine only runs on the same TensorRT version and GPU
# it was built for, so we ship the ONNX model and build the engine
# on the device at first boot. The output is cached under
# $EDGE_VISION_CACHE (default /var/cache/edge-vision, the same
# directory plan_cache.rs uses) keyed by everything that changes the
# engine bytes: model hash, TensorRT version, host architecture,
# SoC/JetPack identity, CUDA version and every extra trtexec argument.
#
# A cache hit is validated with `trtexec --loadEngine … --skipInference`
# before it is returned; a corrupt/foreign plan is rebuilt instead of
# being handed to the runtime.
#
# Usage: tools/build-engine.sh <model.onnx> [extra trtexec args...]
# Environment:
#   EDGE_VISION_CACHE       cache directory (default /var/cache/edge-vision)
#   EDGE_VISION_MIN_MEM_MB  minimum MemAvailable before building (default 1024)
# Prints the engine path on stdout (all chatter goes to stderr).
set -euo pipefail

die() { echo "build-engine: $*" >&2; exit 2; }

[ $# -ge 1 ] || die "usage: $0 <model.onnx> [extra trtexec args...]"
model=$1
shift
extra_args=("$@")

[ -f "$model" ] || die "model not found: $model"
command -v trtexec >/dev/null 2>&1 \
    || die "trtexec not found in PATH — this bring-up script needs a development image with \
tensorrt-trtexec; the product path builds in-process (design D5). Do not add trtexec to a \
release image."
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found"

cache_dir="${EDGE_VISION_CACHE:-/var/cache/edge-vision}"
mkdir -p "$cache_dir" 2>/dev/null || true
if [ ! -w "$cache_dir" ]; then
    die "cache directory is not writable: $cache_dir
Set EDGE_VISION_CACHE to a writable directory (the service unit uses \
CacheDirectory=edge-vision, which binds /var/cache/edge-vision)."
fi

# One build at a time: two cameras or two starts must not build the
# same plan in parallel and race on the cache.
if command -v flock >/dev/null 2>&1; then
    exec 9>"$cache_dir/.build.lock"
    flock 9
else
    echo "build-engine: flock not found; builds are not serialised" >&2
fi

model_hash=$(sha256sum "$model" | cut -c1-16)

# Refuse a build that would OOM an 8 GB (or 4 GB) shared-memory
# device while another camera process is running.
min_mem_mb="${EDGE_VISION_MIN_MEM_MB:-1024}"
avail_mb=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0)
if [ "${avail_mb:-0}" -lt "$min_mem_mb" ]; then
    die "only ${avail_mb} MB MemAvailable; need at least ${min_mem_mb} MB \
(override with EDGE_VISION_MIN_MEM_MB)"
fi

# TensorRT version: try the tool banner and --version first, then the
# installed library name, then package/database fallbacks. The image
# has no dpkg and no python tensorrt package, so neither is required.
trt_version=""
for candidate in \
    "$(trtexec --version 2>&1 || true)" \
    "$(trtexec --help 2>&1 || true)"; do
    trt_version=$(printf '%s\n' "$candidate" \
        | grep -oE 'TensorRT v?[0-9]+(\.[0-9]+)+' | head -n1 \
        | grep -oE '[0-9]+(\.[0-9]+)+' || true)
    [ -n "$trt_version" ] && break
done
if [ -z "$trt_version" ]; then
    lib=$(ls -1 /usr/lib/libnvinfer.so.* 2>/dev/null | sort -V | tail -n1 || true)
    if [ -n "$lib" ]; then
        trt_version=$(basename "$lib" | grep -oE '[0-9]+(\.[0-9]+)+' || true)
    fi
fi
if [ -z "$trt_version" ] && command -v dpkg-query >/dev/null 2>&1; then
    trt_version=$(dpkg-query -W -f='${Version}' tensorrt 2>/dev/null \
        | grep -oE '^[0-9]+(\.[0-9]+)+' || true)
fi
if [ -z "$trt_version" ]; then
    trt_version=$(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true)
fi
if [ -z "$trt_version" ]; then
    die "could not determine TensorRT version (tried trtexec --version/--help, \
/usr/lib/libnvinfer.so.*, dpkg-query, python3 tensorrt)"
fi

# Engines are tied to the SoC, BSP and CUDA version as well as the
# arch: an Orin NX plan must never be reused on an AGX Orin or after
# an L4T/CUDA bump.
arch=$(uname -m)
soc=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0' || true)
compatible=""
if [ -r /proc/device-tree/compatible ]; then
    compatible=$(tr '\0' '|' < /proc/device-tree/compatible)
fi
l4t=$(cat /etc/nv_tegra_release 2>/dev/null | tr '\n' ' ' || true)
if [ -z "$l4t" ]; then
    l4t="unknown"
    echo "build-engine: warning: /etc/nv_tegra_release missing; L4T identity is 'unknown'" >&2
fi
cuda=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+(\.[0-9]+)+' | head -n1 || true)
platform_id="${arch}|${soc}|${compatible}|${l4t}|${cuda}"
platform_hash=$(printf '%s' "$platform_id" | sha256sum | cut -c1-8)

# Hash the actual argument list, including precision/DLA/shape flags.
args_text=""
if [ ${#extra_args[@]} -gt 0 ]; then
    args_text=$(printf '%s\n' "${extra_args[@]}")
fi
args_hash=$(printf '%s' "$args_text" | sha256sum | cut -c1-8)

name=$(basename "$model" .onnx)
engine="$cache_dir/${name}-${model_hash}-trt${trt_version}-${platform_hash}-${args_hash}.engine"
sidecar="${engine}.args"

# Default to FP16 only when the caller did not choose a precision;
# `--fp16` plus `--int8` used to be passed together.
has_precision=0
has_workspace=0
for arg in "${extra_args[@]}"; do
    case "$arg" in
        --fp16|--int8|--fp32|--best|--precision=*) has_precision=1 ;;
        --memPoolSize*|--workspace*) has_workspace=1 ;;
    esac
done

if [ -f "$engine" ]; then
    if trtexec --loadEngine="$engine" --skipInference >/dev/null 2>&1; then
        echo "build-engine: cache hit: $engine" >&2
        echo "$engine"
        exit 0
    fi
    echo "build-engine: cached engine failed to load, rebuilding: $engine" >&2
    rm -f "$engine"
fi

echo "build-engine: building $engine (this runs on-device and can take minutes)" >&2
# A failed trtexec must not leave a partial plan behind for the next
# cache check to trip over.
trap 'rm -f "$engine.tmp"' EXIT
build_args=(--onnx="$model" --saveEngine="$engine.tmp")
[ "$has_precision" -eq 1 ] || build_args+=(--fp16)
# On Orin, workspace is system RAM. The C++ product uses 256 MiB;
# the trtexec default is "all device memory", which can OOM a Nano.
[ "$has_workspace" -eq 1 ] || build_args+=(--memPoolSize=workspace:256)
if [ ${#extra_args[@]} -gt 0 ]; then
    build_args+=("${extra_args[@]}")
fi
trtexec "${build_args[@]}" >&2
mv "$engine.tmp" "$engine"
{
    printf 'trt=%s\narch=%s\nsoc=%s\ncompatible=%s\nl4t=%s\ncuda=%s\nargs=%s\n' \
        "$trt_version" "$arch" "$soc" "$compatible" "$l4t" "$cuda" "$args_text"
} > "$sidecar"
echo "$engine"
