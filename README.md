# Edge Vision Pipeline in Rust

One Rust binary that captures camera frames, runs a neural network and
streams the result, with pixels intended to stay in hardware memory from
sensor to encoder. Jetson Orin NX/Nano with JetPack 7 is the first
target; NXP and Ambarella are later `platform-*` crates.

The design and the current implementation status live in
[docs/design.md](docs/design.md). The device plan, pinned versions and
M1 exit gates are in [docs/jetson-bringup.md](docs/jetson-bringup.md);
building and packaging on the camera image is in
[docs/yocto.md](docs/yocto.md).

## Status

Everything below runs and is tested on x86:

- Workspace with `core`, `pipeline`, `postproc`, `sinks`, `conformance`,
  `platform-sim`, `platform-jetson` and `apps/edge-vision`.
- HAL: `HwFrame`/`HwBuffer`/`Tensor`, plane descriptors (byte offsets,
  pitch, rows), declared memory domain/layout, and the stage traits.
- TOML config with typed validation and `deny_unknown_fields`; the
  builder validates capabilities, precision/device, encoder choice and
  the engine's bindings (one static F32 NCHW input, static F32 outputs).
- Runner: capture + processing (+ dedicated encoder) threads, bounded
  drop-oldest queues, explicit `FramePoll` timeout reasons, stop handle,
  idle limit, partial metrics on errors, live counters.
- Metrics: per-stage totals, drop/timeout counts, queue depth, and
  p50/p95/p99 end-to-end frame age; Prometheus text output.
- Sim backend: synthetic/file NV12 sources and a real NV12 letterbox
  pre-processor; deterministic engine or ONNX Runtime with
  `--features onnx`.
- YOLOv8/v11 and YOLOX decoders with NMS, pre-NMS caps, and a
  zero-allocation steady-state gate. A committed end-to-end test runs
  the ONNX fixture through ONNX Runtime, the YOLO decoder, NMS and a
  sink.
- Sinks: JSONL metadata, raw file video, fan-out, counting/null.
- Jetson CPU H.264 encode (Orin Nano path) with `x264enc`, running on
  its own thread, plus a relaxed-mode `config/orin-nano-yolo.toml`;
  device streaming still awaits the NVMM capture branch and the
  VIC/TensorRT stages.
- Conformance suite every backend must pass.
- Jetson groundwork: NVENC/NVJPG sysfs probe, pinned GStreamer launch
  strings, a GStreamer capture source (`--features gst`, dmabuf branch
  plus a PC-only copy fallback), an integrity-checked TensorRT plan
  cache, and Argus EOS treated as a capture fault rather than a clean
  exit.
- CI: fmt, clippy (`-D warnings`) across default/onnx/gst/shipped
  features, workspace/ONNX/GStreamer tests, feature powerset, MSRV 1.92,
  `cargo deny`, and an aarch64 cross build.

Not implemented yet (device work, tracked in the docs):

- NVMM/NvBufSurface access, CUDA/VIC pre-processing, TensorRT execution,
  NVENC encode, RTSP output and the overlay stage on Jetson.
- The device zero-copy proof (`copied_frames == 0`, Nsight, EMC load).
- Argus process-level recovery, the in-flight surface budget and
  multi-camera orchestration.
- The `rtsp_port` config field was removed until the sink exists; it had
  no reader.

## Run it (no hardware needed)

```sh
cargo run -p edge-vision -- config/x86-dev.toml --frames 30
cargo test --workspace
cargo test -p platform-sim --features onnx
cargo test -p platform-jetson --features gst   # needs GStreamer dev libs
cargo test -p edge-vision --features onnx      # end-to-end YOLO fixture
```

For the Yocto image, see [docs/yocto.md](docs/yocto.md).
`tools/build-engine.sh` is a bring-up tool that needs `trtexec`; the
product builds TensorRT plans in-process and caches them under
`/var/cache/edge-vision`.

`--frames N` limits frames *captured*; under load the drop-oldest queue
may process fewer (the metrics line reports both). Add `--prometheus`
for a Prometheus text exposition and a live summary every 5 seconds.

The sim backend has no encoder, so it streams detections only.
`zero_copy = "strict"` (the default) refuses CPU-encode profiles at load
time; on a platform that offers a CPU path, video without NVENC is the
explicit opt-in `zero_copy = "relaxed"` + `encoder.hardware = false`,
and the builder enforces exactly that matrix.

## Layout

```text
crates/core/          traits, HwFrame, Tensor, plane descriptors, config, errors — #![forbid(unsafe_code)]
crates/conformance/   trait-level checks every platform backend must pass
crates/pipeline/      builder, capture/processing threads, drop-oldest backpressure, live metrics
crates/postproc/      YOLO/YOLOX decoders, NMS (platform-neutral)
crates/sinks/         JSONL metadata, file video, fan-out/counting sinks
crates/platform-sim/  synthetic/file source, deterministic or ONNX engine — runs anywhere
crates/platform-jetson/ media-engine probe, launch strings, GStreamer capture, plan cache
apps/edge-vision/     main: parse config, pick platform, run
config/               x86-dev.toml (sim), orin-nx-yolo.toml, orin-nano-yolo.toml
docs/                 design + Jetson M1 bring-up
tools/                build-engine.sh, cross-compile Docker image, bench, test-model generator
```

Dependencies point downward (app → pipeline → core); no crate outside
`platform-*` links a vendor SDK.
