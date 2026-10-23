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
- [ ] ONNX Runtime engine in `platform-sim` + a YOLO model on a recorded
      clip (needs the model/clip committed or fetched in CI)
- [ ] Flash the Orin NX with pinned JetPack 6.x; record CUDA/TensorRT/L4T
      versions in `config/orin-nx-yolo.toml` (M1: direct Argus capture)

## Run it (no hardware needed)

```sh
cargo run -p edge-vision -- config/x86-dev.toml --frames 30
cargo test --workspace
```

The sim backend has no hardware encoder, so — like the Orin Nano
profile — it streams detections only and never hands frames to a CPU
encoder. `zero_copy = "strict"` in the config refuses CPU-encode
profiles at load time, and the pipeline builder refuses an encoder
request on any platform without hardware encode.

## Layout

```text
crates/core/          traits, HwFrame, Tensor, config, errors — #![forbid(unsafe_code)]
crates/pipeline/      builder, capture/processing threads, drop-oldest backpressure, metrics
crates/platform-sim/  synthetic/file source, deterministic engine — runs anywhere
apps/edge-vision/     main: parse config, pick platform, run
config/               x86-dev.toml (sim), orin-nx-yolo.toml (jetson target)
```

Dependencies point downward (app → pipeline → core); no crate outside
`platform-*` will ever link a vendor SDK.
