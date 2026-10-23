# Edge Vision Pipeline in Rust

One Rust binary that captures camera frames, runs a neural network on
them and streams the result, with frames staying in hardware memory
end to end. Jetson Orin NX/Nano is the first hardware target; NXP and
Ambarella come later as new `platform-*` crates. The full plan, the
zero-copy contract and the milestone gates live in [design.md](design.md).

## Status

M0 scaffold (the "first week" checklist from design.md):

- [x] Workspace with `core`, `pipeline`, `platform-sim`, `apps/edge-vision`
- [x] `HwFrame`, `HwBuffer`, `Tensor` and the stage traits from the HAL design
- [x] TOML config schema + loader with typed validation errors
- [x] `platform-sim` synthetic/file source and deterministic CPU engine
- [x] CI: fmt, clippy (`-D warnings`), tests, `cargo deny`, aarch64 cross-build
- [x] ONNX Runtime engine in `platform-sim` (`--features onnx`), verified
      against generated fixtures and a real model (below)
- [x] `edge-postproc`: YOLOv8/v11 and YOLOX decoders with letterbox
      inversion + NMS; real NV12 letterbox preprocessor in the sim
- [x] Real-model end-to-end (manual run, not yet a committed test): the
      reference products' YOLOX-tiny package (external path, not copied)
      on a photo clip — bus 0.92 and three persons 0.86–0.90,
      corroborated against Python onnxruntime to within 0.005/box
- [x] External review (REVIEW-2026-10-23) worked through: runner
      deadlock and pool-truncation fixes with regression tests, CI
      aarch64 linker + gst-feature jobs, relaxed CPU-encode opt-in
      reachable in the builder
- [x] `edge-sinks`: JSONL metadata (`stream.meta_file`), file video,
      fan-out sinks; Prometheus rendering for pipeline metrics
- [x] `edge-conformance`: trait-level suite every backend must pass
      (sim passes; deliberately broken stubs are caught by the suite)
- [x] Tools: `build-engine.sh` (on-device TensorRT engine build, cached by
      model hash + TRT version), cross-compile Docker image (verified to
      produce an aarch64 binary), pipeline throughput bench
- [x] Reference review (camera-viber, LPR app, gst-nvmm-cpp) folded into
      design.md revision 2026-10-23: GStreamer-first hybrid M1, JetPack 7
      pin, Nano CPU-encode as explicit opt-in, sysfs media-engine probe
- [ ] Flash the Orin NX (JetPack 7 / L4T r39.2.1); record exact versions in
      `config/orin-nx-yolo.toml` (M1: GStreamer capture/encode on device)

## Run it (no hardware needed)

```sh
cargo run -p edge-vision -- config/x86-dev.toml --frames 30
cargo test --workspace
```

The sim backend has no encoder at all, so it streams detections only.
`zero_copy = "strict"` (the default) refuses CPU-encode profiles at
load time; on a platform that offers a CPU path, video without NVENC
is the explicit opt-in `zero_copy = "relaxed"` + `encoder.hardware =
false`, and the builder enforces exactly that matrix.

## Layout

```text
crates/core/          traits, HwFrame, Tensor, config, errors — #![forbid(unsafe_code)]
crates/conformance/   trait-level checks every platform backend must pass
crates/pipeline/      builder, capture/processing threads, drop-oldest backpressure, metrics
crates/postproc/      YOLO decoder + NMS (platform-neutral)
crates/sinks/         JSONL metadata, file video, fan-out/counting sinks
crates/platform-sim/  synthetic/file source, deterministic or ONNX engine — runs anywhere
crates/platform-jetson/ media-engine probe + pinned GStreamer launch strings (M1 runtime on device)
apps/edge-vision/     main: parse config, pick platform, run
config/               x86-dev.toml (sim), orin-nx-yolo.toml (jetson target)
```

Dependencies point downward (app → pipeline → core); no crate outside
`platform-*` will ever link a vendor SDK.
