# Jetson inference path in Rust: implementation plan

**Date:** 2026-10-23
**Status:** plan, not started
**Base:** `8760268`

**Scope:** the Jetson stages behind the existing traits: NVMM frame access, VIC/CUDA pre-processing, TensorRT inference and the plan cache. Encode and the runner split are touched only where inference needs them.

**Reference.** The same pieces already run in the C++ product, `orin-nx-camera-viber`:
- `embedded/Machines/Jetson/TensorRt/` (`TensorRtEngine`, `NvmmFramePreprocessor`, `NvmmGpuTensorWriter`, `CudaTensorKernels.cu`, `BuildProgressAdapter`)
- `docs/INFERENCE-ENGINE.md`

This plan ports its *behaviour*, not its code layout.

**Pinned versions** (meta-tegra in the camera image):

| Component | Version |
|---|---|
| L4T | r39.2.1 |
| CUDA | 13.2.86 (`cuda-cudart`, `cuda-nvcc`, with `gcc-for-nvcc`) |
| TensorRT | 10.16.2.10 (`tensorrt-core`, `tensorrt-trtexec`) |
| NvBufSurface | `tegra-libraries-multimedia-utils_39.2.1` |
| Orin GPU | compute capability 8.7 |

---

## 1. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Two `-sys` crates.** `nvbuf-sys` (bindgen over C headers) and `trt-sys` (a C++ shim through `cxx`). | NvBufSurface is a C API that Rust must read field by field: the fd, layout and pitches. TensorRT is C++ only. |
| D2 | **CUDA calls live inside the C++ shim**, not in `cudarc`. | We need only about 12 calls (stream, events, mapped host alloc, graph capture, EGL interop). They share lifetimes with the TensorRT objects, and `cudarc` support for CUDA 13 is unverified. Revisit after M1. |
| D3 | **I/O tensors use mapped host memory** (`cudaHostAlloc(..., cudaHostAllocMapped)`). | The Orin's memory is integrated; the reference does this and measured it. The CPU sees an `&[f32]`, so `edge-postproc` runs **unchanged**, and the GPU sees a device pointer. |
| D4 | **Default pre-processing: the VIC scales, the CPU writes the planar tensor.** The CUDA kernel is an opt-in feature (`cuda-preprocess`). | This is the reference's default and its fallback. The kernel needs nvcc and EGL in the build, so it lands second. |
| D5 | **The engine is built in-process** (builder plus `nvonnxparser` in the shim, with progress reported), cached, and warmed up on its own thread. `trtexec` serves bring-up only. | The device builds its plan at first boot after any image or model change, which takes minutes. In-process gives progress and a lock against concurrent builds, and needs no extra binary on the image. |
| D6 | **The video does not wait for the engine.** Until a plan is ready, the pipeline runs with "no detections". | A build of several minutes must not mean several minutes without video (round 3, L3). |
| D7 | **One engine context and one CUDA stream per camera.** IExecutionContext is `Send`, never `Sync`. | This follows TensorRT's threading rules and the reference. |
| D8 | **`unsafe` lives only in the `-sys` crates.** `platform-jetson` keeps `#![forbid(unsafe_code)]`. | It is the design's rule, and it keeps the unsafe code reviewable. |

## 2. Crate layout

```text
crates/platform-jetson/
├── nvbuf-sys/              # unsafe: bindgen output + link lines; no logic
│   ├── build.rs            # bindgen over nvbufsurface.h, nvbufsurftransform.h from $SYSROOT
│   └── src/lib.rs          # include!(bindings) + layout tests
├── trt-sys/                # unsafe: cxx bridge + C++ shim (+ .cu behind a feature)
│   ├── build.rs            # cxx_build (Yocto g++), nvcc (gcc-for-nvcc, sm_87) when cuda-preprocess
│   ├── shim/trt_shim.h     # the C++ side of the bridge, ~300 lines with the .cpp
│   ├── shim/trt_shim.cpp
│   ├── shim/tensor_kernel.cu   # feature cuda-preprocess
│   └── src/lib.rs          # #[cxx::bridge] declarations only
└── src/                    # safe: #![forbid(unsafe_code)]
    ├── nvmm.rs             # NvmmSurface::from_gst(&gst::Buffer): fd, layout, pitch, planes
    ├── vic.rs              # VicScaler: NvBufSurfTransform into a pooled input-size surface
    ├── cuda.rs             # CudaStream, MappedTensor (impl TensorBuffer), GpuTimer
    ├── trt.rs              # TrtEngine (impl InferenceEngine)
    ├── plan_cache.rs       # key, path, atomic write, prune: pure Rust, tested on the PC
    ├── warmup.rs           # EngineWarmup thread + process-wide build lock
    └── preprocess.rs       # JetsonPreprocessor (impl Preprocessor): VIC + planar write
```

**Features:**
- `gst` exists today.
- `trt` adds `nvbuf-sys` and `trt-sys`, and needs the Jetson sysroot.
- `cuda-preprocess` adds the kernel and needs nvcc.
- Without `trt`, `platform-jetson` builds on any PC, as it does now.

## 3. The shim's API (`trt-sys`)

`cxx` writes both sides and checks that their types match. Every function catches C++ exceptions and returns a `Result`.

```rust
#[cxx::bridge(namespace = "edge::trt")]
mod ffi {
    struct IoInfo { name: String, dims: Vec<i64>, is_input: bool, dtype: u8 }
    struct BuildOptions { fp16: bool, int8: bool, workspace_mb: u32 }

    extern "Rust" {
        type LogSink;                                             // Rust receives TensorRT's log
        fn on_log(sink: &LogSink, severity: u8, msg: &str);       // + build progress
        fn on_progress(sink: &LogSink, phase: &str, step: i32, steps: i32);
    }
    unsafe extern "C++" {
        include!("trt-sys/shim/trt_shim.h");
        type Engine;   // IRuntime + ICudaEngine (runtime outlives engine)
        type Context;  // IExecutionContext + cudaStream_t + optional cudaGraphExec_t
        type HostBuf;  // cudaHostAlloc(mapped): host ptr + device ptr

        fn version() -> u32;                                                  // getInferLibVersion
        fn build_plan(onnx: &[u8], opts: &BuildOptions, log: &LogSink) -> Result<Vec<u8>>;
        fn load_engine(plan: &[u8], log: &LogSink) -> Result<UniquePtr<Engine>>;
        fn io_infos(engine: &Engine) -> Vec<IoInfo>;
        fn new_context(engine: &Engine) -> Result<UniquePtr<Context>>;        // creates its stream
        fn set_input_shape(ctx: Pin<&mut Context>, name: &str, dims: &[i64]) -> Result<()>;
        fn host_alloc(bytes: usize) -> Result<UniquePtr<HostBuf>>;
        unsafe fn host_ptr(buf: &HostBuf) -> *mut f32;      // wrapped as &mut [f32] in cuda.rs
        fn bind(ctx: Pin<&mut Context>, name: &str, buf: &HostBuf) -> Result<()>; // setTensorAddress(device ptr)
        fn enqueue(ctx: Pin<&mut Context>) -> Result<()>;   // graph launch if captured, else enqueueV3
        fn synchronize(ctx: Pin<&mut Context>) -> Result<()>;
        fn capture_graph(ctx: Pin<&mut Context>) -> Result<()>; // once, after the warm-up pass
        fn gpu_ms(ctx: &Context) -> f32;                    // CUDA events around the last enqueue; <0 = unknown
    }
}
```

The safe layer (`trt.rs`) holds `TrtEngine { ctx: UniquePtr<Context>, engine: UniquePtr<Engine>, … }`, declared in that order so the context drops before the engine. It is `Send`, not `Sync`.

`LogSink::on_log` and `on_progress` run inside `std::panic::catch_unwind`, because a panic that crosses into C++ is undefined behaviour.

## 4. Changes to the core traits first (small, testable on the PC)

These land before any device code, so the Jetson stages have somewhere to plug in.

| Change | Where | Why |
|---|---|---|
| **C1.** `GpuStream` becomes an opaque handle that the platform creates: `Platform::stream(camera_id) -> Result<GpuStream>`, wrapping an `Arc<dyn Any + Send + Sync>` that the backend downcasts. The sim returns a no-op. | `core/stage.rs`, `pipeline/lib.rs` (it does `GpuStream::new(0)` today) | A real `cudaStream_t` must reach the engine and the preprocessor. |
| **C2.** The pipeline accepts an engine that is not ready yet: `Platform::engine()` may return `EngineState::Warming(handle)`. The runner then skips pre/infer/post and pushes empty `Detections` until the handle reports ready. | `core/stage.rs`, `pipeline/lib.rs` | D6. Test on the PC with a sim engine that takes 2 s to become ready: video and metadata both flow from the first frame. |
| **C3.** `Pipeline::build` checks the engine's input `TensorSpec` (shape `[1,3,H,W]`, dtype) against `ModelConfig`, and fails with `DomainMismatch { stage: "engine" }`. | `pipeline/lib.rs` | A mismatched model should fail at startup, not at the first frame. |
| **C4.** Letterbox geometry and planar normalisation move into one shared module, reused by the sim preprocessor, the Jetson preprocessor and the decoders. | new `core/src/letterbox.rs` (no unsafe) | One definition of `scale`, `pad` and the colour/normalise steps, so the PC's golden values match the device's. |

## 5. From the reference to Rust

| Reference (C++) | What it does | Rust task |
|---|---|---|
| `gst_buffer_map(READ)`, then `map.data` as `NvBufSurface*` (`NvmmFramePreprocessor.cpp:106-113`) | Gets the surface of an NVMM GstBuffer | T2 `NvmmSurface::from_gst` |
| `NvmmFramePreprocessor` | VIC: NV12 (pitch- or block-linear) into a pitch-linear RGBA surface at the input size; the CPU writes the planar input | T8 `VicScaler` + `JetsonPreprocessor` |
| `NvmmGpuTensorWriter` + `CudaTensorKernels.cu` | EGLImage → CUDA pointer; a kernel writes the tensor | T9 (`cuda-preprocess`) |
| `TensorRtEngine::Create/Load_/Bind_` | Plan cache or build; one context and stream; batch fixed to 1; mapped buffers | T5, T6 |
| `TensorRtEngine::Prepare` + `Analytics::EngineWarmup` | Builds the plan on the `engine-warm` thread under a process-wide lock | T7 |
| `TensorRtEngine::CaptureGraph_`, `GpuMs_` | CUDA graph after the warm-up pass; GPU time from events | T10 |
| `BuildProgressAdapter` | `IProgressMonitor` → log as a percentage | T6 (`on_progress`) |

## 6. Tasks

Each task names its files and dependencies, and how it is checked:
- **PC** checks run in CI or on the dev host.
- **Device** checks run on the Orin, as `#[ignore]` tests with `cargo test --features trt -- --ignored`, or as a logged run.

| Task | What | Files | Needs | Done when |
|---|---|---|---|---|
| **T0** | **Toolchain.** Build against the Yocto SDK sysroot: `build.rs` reads `$SDKTARGETSYSROOT` (or `JETSON_SYSROOT`); bindgen gets `--sysroot`; `cxx_build` and nvcc use the SDK's cross g++ and `gcc-for-nvcc`. Docker image `tools/Dockerfile.jetson-sdk`, Rust matching Yocto's 1.98. | `trt-sys/build.rs`, `nvbuf-sys/build.rs`, `tools/` | — | **PC:** `cargo build -p platform-jetson --features trt --target aarch64-unknown-linux-gnu` succeeds in the SDK container. Without the sysroot, the build fails with one clear message. |
| **T1** | **`nvbuf-sys`.** bindgen with an allowlist (`NvBufSurface*`, `NvBufSurfTransform*`, `NvBufSurfaceMapEglImage`), link lines, generated at build time (never committed), with bindgen's layout tests. | `nvbuf-sys/*` | T0 | **PC:** cross build plus the layout tests pass on the host target where possible. **Device:** the layout tests pass on the Orin. |
| **T2** | **`NvmmSurface`.** Map the GstBuffer and check `map.size() == size_of::<NvBufSurface>()`. Read plane 0's `bufferDesc` (the fd), `layout`, `pitch` and `planeParams`. It holds the `MappedBuffer` for its lifetime. `gst_source` uses it instead of the "NVMM refused" error. `FrameMeta` gains a plane descriptor (round 1, P1-7), and the layout comes from the surface, not hardcoded. | `src/nvmm.rs`, `src/gst_source.rs`, `core/frame.rs` | T1 | **PC:** a unit test of plane-descriptor maths with a fake surface. **Device:** `gst-capture` on cam0 logs the fd, layout and pitch, `copied_frames == 0` over 300 frames, and the fd is stable while buffers recycle. |
| **T3** | **`trt-sys` skeleton.** Logger, `version`, `load_engine`, `io_infos`, `new_context`, `enqueue`, `synchronize`. Exceptions become `Result`; logs go to `tracing`. | `trt-sys/*` | T0 | **Device:** load a `trtexec`-built `yolox_s` FP16 plan, list its I/O, and run one enqueue on zeroed buffers without error (an `#[ignore]` test). |
| **T4** | **CUDA buffers.** `host_alloc` and `bind`. `MappedTensor` implements `TensorBuffer`: `as_f32_slice(_mut)` over the host pointer, and the device pointer is reachable inside the backend through `as_any`. `CudaStream` wraps the context's stream for C1. | `src/cuda.rs`, `trt-sys` | T3, C1 | **PC:** a `MappedTensor` with a fake allocator passes the tensor tests. **Device:** write a pattern from the CPU, read it back after a GPU no-op on the same stream. |
| **T5** | **`TrtEngine`** (implements `InferenceEngine`). Bind every I/O tensor to mapped memory; fix a dynamic batch to 1; refuse non-float I/O; `infer` = enqueue + synchronize; outputs are `Tensor`s with real `TensorSpec`s, so `YoloxPostProcessor` reads them unchanged. | `src/trt.rs` | T4, C3 | **Device:** feed the same input tensor (dumped from the PC's sim preprocessor) to the PC's ONNX Runtime and to `TrtEngine`. FP32 outputs agree within 1e-3, FP16 within 1e-2, and the decoded boxes are identical. |
| **T6** | **Plan cache and builder.** `build_plan` (builder + `nvonnxparser`, FP16/INT8 flags, progress callback) in the shim. In pure Rust, `plan_cache.rs`: key = SHA-256 of the ONNX + TRT version + precision + SM 8.7 + builder flags; path `/var/cache/edge-vision/<key>.plan`; write to a temp file, fsync, rename, fsync the directory; a plan that fails to load is rebuilt; old keys are pruned. | `trt-sys/shim`, `src/plan_cache.rs` | T5 | **PC:** key, path, prune and atomic-write unit tests (temp dir), including "a truncated plan file is treated as missing". **Device:** first start builds and logs progress; the second loads in under 1 s; a corrupted plan rebuilds. |
| **T7** | **Warm-up.** An `EngineWarmup` thread (named `engine-warm`) builds through T6 under a process-wide `flock`. `JetsonPlatform::engine()` returns `EngineState::Warming` until the plan exists (C2). | `src/warmup.rs` | T6, C2 | **PC:** the C2 test. **Device:** after deleting the cache, video starts at once, detections start when the plan is built, and the log shows the build time. |
| **T8** | **VIC preprocessor.** `VicScaler`: a pool of input-size, pitch-linear RGBA surfaces (`NvBufSurfaceCreate`); `NvBufSurfTransform` from the frame's surface with a letterbox destination rectangle (C4) and the pad colour; then a CPU planar write (C4 normalisation) into the `MappedTensor`. The VIC reads block-linear directly. | `src/vic.rs`, `src/preprocess.rs` | T2, T4, C4 | **Device:** on a recorded frame, the input tensor matches the sim preprocessor's to within the VIC's bilinear difference (mean absolute error < 2/255), with per-frame timing logged. |
| **T9** | **CUDA preprocess** (`cuda-preprocess`). Map the RGBA surface: `NvBufSurfaceMapEglImage`, then `cuGraphicsEGLRegisterImage` once per pooled surface. `tensor_kernel.cu` writes the planar, normalised tensor straight to the mapped input's device pointer, on the camera's stream. | `trt-sys/shim/tensor_kernel.cu`, `src/preprocess.rs` | T8 | **Device:** output equals T8's within 1e-6, and the CPU time per frame drops (logged). |
| **T10** | **Graph and timing.** `capture_graph` after the first good inference; `gpu_ms` from events; both feed `PipelineMetrics` (infer GPU time vs launch time). A failed capture falls back to `enqueueV3` once, with a log line. | `trt-sys/shim`, `src/trt.rs`, `pipeline/metrics.rs` | T5 | **Device**, with clocks pinned: per-inference time logged. Reference: 3.05 ms with a graph vs 7.04 ms without (README 0.8.0). |
| **T11** | **Conformance on the device.** Run `conformance::check_all` with `JetsonPlatform`. Nsight Systems trace: no `memcpy HtoD/DtoH/DtoD` in steady state. `tegrastats` EMC load recorded. | `crates/conformance`, `platform-jetson/tests/device.rs` | T5–T8 | **Device:** the suite passes and the trace is clean. This is the M1 inference exit gate. |
| **T12** | **Packaging** (propose to the image owner; don't change the camera layer unasked). A recipe with OE's `cargo` class and crate fetch from `Cargo.lock`; `DEPENDS` on `tensorrt-core`, `cuda-cudart`, `cuda-nvcc-native` with `gcc-for-nvcc` (for `cuda-preprocess`) and `tegra-libraries-multimedia-utils`; the systemd unit's `CacheDirectory=edge-vision`, `video` group, camera-guard exec/write entries. | proposal doc only | T11 | The recipe draft is reviewed; nothing is added to the image without approval. |

**Order and parallelism.** The C1–C4 core changes and T0 (toolchain) can start now, on the PC, in parallel. T1→T2 and T3→T4→T5 are two independent tracks until T8 joins them. T6/T7 follow T5, and T9/T10 are optimisations after T8.

**First board session** (decides M1's size): T2's `copied_frames == 0` plus layout, and T3's "load and enqueue". Run both before investing in T6–T10.

## 7. Safety rules for the FFI code

- **Ownership order** is explicit in field declaration order (the context before the engine before the runtime) and commented as such.
- **No panic crosses FFI.** Every `extern "Rust"` callback is wrapped in `catch_unwind`, and the C++ shim wraps every entry point in `try { … } catch (...)`, mapping errors to `Result` (cxx does the mapping for `Result<T>` returns).
- **Raw pointers** never leave `-sys` crates. `host_ptr` becomes `&mut [f32]` with the buffer's lifetime, in exactly one function.
- **Sticky CUDA errors.** After a failed `enqueue` or `synchronize`, the context is marked broken. `infer` returns `Err` from then on, and the runner rebuilds the engine at most once (the C++ product's lesson: a sticky CUDA error failed every frame forever).
- **Bindings** are generated at build time from the pinned sysroot, never committed, and checked with layout tests (struct layouts change between L4T releases).
- **Threading:** one `Context` per camera thread; the CUDA device is set on each thread that enqueues work.

## 8. Testing

| Layer | Runs on | What it covers |
|---|---|---|
| Unit | PC (CI) | plan key/path/prune/atomic write, plane maths, letterbox (C4), shape checks (C3), warming engine (C2), `MappedTensor` with a fake allocator |
| Golden | PC → device | The PC dumps the input tensor and ONNX Runtime outputs for a fixed frame; the device compares T5/T8/T9 against them (tolerances in the table) |
| Device `#[ignore]` | Orin | T2–T10 checks; `cargo test -p platform-jetson --features trt -- --ignored` |
| Conformance + trace | Orin | T11 |
| CI | PC | Default features as now, plus a **compile-only** `--features trt` job in the SDK container (T0), so the shim cannot rot the way the `gst` feature did (round 1, U-1) |

## 9. Open questions

1. **`cudarc` instead of the CUDA calls in the shim (D2).** Check whether a `cudarc` release supports CUDA 13 on aarch64. If it does and the shim's CUDA part grows, move streams, events and memory to it.
2. **DLA** (Orin NX has two). Out of scope for M1. The builder flag and `model.device = "dla0"` hook are reserved in `BuildOptions` and `PlatformCaps`.
3. **INT8 needs a calibration set** (the design's targets say INT8). M1 ships FP16; INT8 is a T6 follow-up with its own accuracy gate.
4. **Async inference.** M1 synchronises inside `infer`. Overlapping pre-processing of frame N+1 with inference of N needs events and a two-slot input; only do it if T10's numbers say so.
