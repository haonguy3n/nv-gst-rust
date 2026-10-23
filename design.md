# Edge Vision Pipeline in Rust — Project Plan

Oct 23, 2026 · @Hao

## Goals and non-goals

Build one Rust binary that captures camera frames, runs a neural network on them and streams the encoded video, with every frame staying in hardware memory end to end. Jetson Orin Nano / Orin NX is the first target; NXP i.MX and Ambarella must be addable as new backend crates without touching the pipeline core.

**Goals**

1. **Zero-copy path** on Jetson: Argus → NvBufSurface (NVMM) → CUDA pre-process → TensorRT → overlay → NVENC → RTSP. No CPU `memcpy` of pixel data in the hot loop.
2. **Platform-agnostic core**: the pipeline only knows traits (`FrameSource`, `Preprocessor`, `InferenceEngine`, `Encoder`, `Sink`) and an opaque `HwFrame` handle. Vendor code lives in `platform-*` crates behind Cargo features.
3. **Clean, safe Rust**: all `unsafe` FFI confined to `*-sys` crates plus one thin safe wrapper per vendor SDK; RAII for every hardware buffer.
4. **Config-driven**: one TOML file picks the platform, camera, model, encoder and stream outputs — no recompiling to swap a model or resolution.
5. **Observable**: per-stage latency, FPS, dropped frames and GPU/DLA load exported as logs and Prometheus metrics.

**Performance targets (Orin NX 16 GB, 1080p30, YOLO-class detector, INT8)**

| Metric | Target |
| --- | --- |
| Sustained throughput | 30 FPS per camera, 2 cameras |
| Glass-to-network latency | < 100 ms p95 |
| CPU load (whole process) | < 25% of one core per stream |
| Steady-state allocations in hot loop | 0 |

Orin Nano targets are the same at one camera; confirm once the model is chosen.

**Non-goals (v1)**

- Training or model conversion tooling beyond a script that runs `trtexec`.
- A GUI; a web preview page comes from the RTSP/WebRTC output.
- Supporting x86 dGPU as a production target (it is useful as a dev/CI backend only).

## Architecture overview

Six stages run as one thread each, joined by bounded channels; only handles to hardware buffers move between them, never pixel bytes.

&#91;embedded content: pipeline architecture · 6 stages, 3 platform columns\]

A new platform fills in the four middle-column boxes for its SoC; post-processing and streaming are reused as they are.

## Zero-copy contract

After the sensor writes a frame, no CPU instruction and no `cudaMemcpy` ever copies its pixels: the frame lives in one NvBufSurface from capture to encoder, and every other hardware buffer is written exactly once. Resizing to the model input is a transform (it makes new pixels), done in one fused pass straight into TensorRT's input.

**Per-frame buffer budget (Orin NX, 1080p camera, 640×640 model)**

| Buffer | Memory | Written by | Read in place by |
| --- | --- | --- | --- |
| Capture surface, NV12 pitch-linear 1920×1080 | NVMM dmabuf, pool of 6 | ISP via Argus | CUDA pre-process, CUDA overlay (draws boxes in place), NVENC (dmabuf import) |
| Input tensor, FP16 3×640×640 | CUDA device memory = the TensorRT input binding | fused CUDA pre-process kernel | TensorRT |
| Output tensors | pinned mapped host memory (`cudaHostAllocMapped`) | TensorRT | CPU post-process |
| Bitstream | V4L2 capture-plane mmap | NVENC | RTSP sink, wrapped as a GstBuffer; returned to the encoder on drop |

Only the output tensors (around 1 MB for a YOLO-class head) and the compressed bitstream are touched by the CPU. The kernel's socket send is the one copy no user-space design can avoid.

**How the code enforces it**

- **No byte access in the type.** `HwBuffer` has no `as_slice()`. `map_cpu()` exists only behind a `cpu-debug` feature that release builds never enable, so code that needs CPU pixels does not compile in release.
- **Strict domain matching.** The pipeline builder never inserts a converter silently. If two stages disagree on memory domain or layout (block-linear vs pitch-linear), startup fails with the mismatch named. At startup it prints the buffer budget above.
- **Write into, never return.** Stages write into buffers owned by the next stage: the pre-processor writes into the engine's input binding; the encoder hands packets out through a callback instead of allocating a `Vec`.
- **Same fd end to end.** `FrameMeta` carries the capture dmabuf fd; the encoder asserts it is encoding that same fd.
- **Frames move by value to the encoder.** The encoder holds the `HwFrame` until NVENC releases it, then dropping it returns the buffer to the Argus pool. Pool size ≥ frames in flight + 2.
- **Zero allocations in the hot loop.** A counting global allocator in tests fails the build if steady-state allocations are above 0.

The trait sketch in the HAL section is written to these rules: `HwFrame` has no CPU-memory variant, every stage takes a GPU stream, and the encoder is optional per platform.

**How it is verified**

- Nsight Systems trace in hardware CI: zero `memcpy HtoD / DtoH / DtoD` in steady state.
- `tegrastats` memory-controller (EMC) load stays near the cost of one write and a few reads per frame; a jump means a hidden copy.
- The M2 exit gate (0 CPU pixel copies) is measured this way, not reviewed by eye.

**No frame copies, no exceptions**

- **Direct Argus capture from M1.** Frames go straight into our own NVMM pool. `nvarguscamerasrc` is never in the product path, so there is no capture-side copy at any milestone.
- **Orin Nano never hands frames to the CPU.** It has no NVENC, so the Nano profile runs capture → infer → overlay fully in NVMM and streams detections only (MQTT / WebSocket metadata). If you need video from a Nano-class product, the answer is a module with NVENC (Orin NX), not a CPU encoder. `zero_copy = "strict"` (the default) refuses any CPU-encode profile.
- **Layout changes stay on hardware.** If a stage needs a different layout (block-linear → pitch-linear), VIC converts NVMM → NVMM, and the startup budget lists it. The CPU never touches pixels.

## Platform abstraction (HAL) design

The core crate defines five stage traits and one buffer abstraction; each platform implements them and registers a factory. The pipeline never names a vendor type.

**Design rules**

- **Buffers are handles, not bytes.** A frame is an `HwFrame` that owns a platform buffer (NvBufSurface on Jetson, DMA-BUF on NXP, IAV buffer on Ambarella). Frames have no CPU access in release builds; the only way to see pixels is a debug-only `map_cpu()` call.
- **Memory domains are explicit.** Each frame buffer advertises where it lives (NVMM or DMA-BUF, never host memory) and its layout. Stages declare what they accept; the pipeline builder refuses at startup when they differ — it never inserts a copy silently (see Zero-copy contract).
- **Static dispatch in the hot path, dynamic at the edges.** Stages are boxed trait objects chosen from config at startup; per-frame calls go through them once per stage (cheap). Generic associated types are avoided so a new backend does not ripple through the core.
- **Capabilities, not `cfg` in the core.** Each backend returns a `PlatformCaps` (formats, max resolution, encoder codecs, accelerators like DLA/NPU). The builder validates the config against it.
- **Bounded pool + backpressure.** Every source allocates a fixed buffer pool; stages are connected by bounded channels; when the sink is slow the oldest frame is dropped and counted.

**Core traits (sketch)**

```rust
// crates/core/src/frame.rs
/// Where a frame's pixels live. There is deliberately no Host variant:
/// a frame can never be in CPU memory.
pub enum FrameMemory {
    Nvmm,    // Jetson NvBufSurface
    DmaBuf,  // NXP / Ambarella / generic V4L2
}

pub enum Layout { PitchLinear, BlockLinear }

pub struct FrameMeta {
    pub seq: u64,
    pub pts: Duration,          // sensor timestamp
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,    // NV12 in v1: ISP, CUDA and encoder all read it natively
    pub layout: Layout,
    pub camera_id: u8,
}

/// Handle to a hardware frame buffer: exposes ids, never bytes.
/// Dropping the last reference returns it to its preallocated pool.
pub trait HwBuffer: Send + Sync + 'static {
    fn memory(&self) -> FrameMemory;
    fn dmabuf_fd(&self) -> RawFd;            // every backend must provide one
    fn as_any(&self) -> &dyn Any;            // downcast inside its own backend only
    #[cfg(feature = "cpu-debug")]            // never compiled into release builds
    fn map_cpu(&self) -> Result<CpuMapping<'_>>;
}

pub struct HwFrame {
    pub meta: FrameMeta,
    pub buf: Arc<dyn HwBuffer>,
}

/// Model input/output only (small). Output tensors may be host-visible.
pub struct Tensor {
    pub spec: TensorSpec,
    pub buf: Arc<dyn TensorBuffer>,
}

// crates/core/src/stage.rs
pub trait FrameSource: Send {
    fn start(&mut self) -> Result<()>;
    /// Hands out a frame from the source's NVMM pool; never allocates, never copies.
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<HwFrame>>;
    fn stop(&mut self) -> Result<()>;
}

pub trait Preprocessor: Send {
    /// Reads the frame in place on the GPU, writes straight into the engine's input binding.
    fn run(&mut self, frame: &HwFrame, dst: &mut Tensor, stream: &GpuStream) -> Result<()>;
}

pub trait InferenceEngine: Send {
    fn input_bindings(&mut self) -> &mut [Tensor];
    fn output_bindings(&self) -> &[Tensor];      // host-visible, read in place
    fn infer(&mut self, stream: &GpuStream) -> Result<()>;
}

pub trait PostProcessor: Send {  // platform-neutral; reads only the small output tensors
    fn decode(&self, outputs: &[Tensor], meta: &FrameMeta, out: &mut Detections) -> Result<()>;
}

pub trait Overlay: Send {
    /// Draws boxes into the frame in place (CUDA on Jetson).
    fn draw(&mut self, frame: &HwFrame, det: &Detections, stream: &GpuStream) -> Result<()>;
}

pub trait Encoder: Send {
    /// Imports the frame's dmabuf into the HW encoder; owns it until the HW releases it.
    fn encode(&mut self, frame: HwFrame, out: &mut dyn FnMut(EncodedPacket)) -> Result<()>;
}

pub trait Sink: Send {           // RTSP, WebRTC, MQTT metadata
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()>;  // wraps the encoder's mmap
    fn push_meta(&mut self, det: &Detections) -> Result<()> { Ok(()) }
}

/// One per platform crate; the binary picks it from config.
pub trait Platform {
    fn name(&self) -> &'static str;
    fn caps(&self) -> PlatformCaps;
    fn source(&self, cfg: &SourceConfig) -> Result<Box<dyn FrameSource>>;
    fn preprocessor(&self, cfg: &PreConfig) -> Result<Box<dyn Preprocessor>>;
    fn engine(&self, cfg: &ModelConfig) -> Result<Box<dyn InferenceEngine>>;
    fn overlay(&self) -> Result<Box<dyn Overlay>>;
    /// None when the SoC has no HW encoder (Orin Nano): the pipeline streams metadata only.
    fn encoder(&self, cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>>;
}
```

`Tensor` is the same idea as `HwFrame`: shape, dtype and an `Arc<dyn TensorBuffer>`, so a CUDA tensor on Jetson and an NPU tensor on NXP flow through the same signature. `PostProcessor` and `Sink` are platform-neutral and live in shared crates, so YOLO decoding or RTSP serving is written once.

## Workspace / crate layout

One Cargo workspace; dependencies only point downward (app → pipeline → core), and no crate outside `platform-*` links a vendor SDK.

```text
edge-vision/
├── Cargo.toml                 # [workspace], shared lints, release profile (lto, panic=abort)
├── config/
│   ├── orin-nx-yolo.toml
│   └── x86-dev.toml
├── crates/
│   ├── core/                  # traits, HwFrame, Tensor, errors, config types — no unsafe
│   ├── pipeline/              # builder, threads/channels, backpressure, metrics
│   ├── postproc/              # YOLO / SSD / classification decoders, NMS, tracker hook
│   ├── sinks/                 # RTSP (gst-rtsp-server), WebRTC, file, MQTT metadata
│   ├── overlay/               # draw boxes/labels; trait so each backend can do it on GPU
│   ├── platform-jetson/       # Argus source, NvBufSurface, CUDA pre, TensorRT, NVENC
│   │   ├── nvbuf-sys/         # bindgen: nvbufsurface.h, nvbufsurftransform.h
│   │   ├── argus-sys/         # small C++ shim (cxx) over libargus
│   │   └── tensorrt-sys/      # C shim over TensorRT C++ API (cxx)
│   ├── platform-sim/          # file/V4L2 source + ONNX Runtime CPU: runs on any laptop & CI
│   ├── platform-nxp/          # (later) V4L2/libcamera, G2D/ISI, NPU delegate, Hantro VPU
│   └── platform-ambarella/    # (later) IAV, CVflow, Amba encoder
├── apps/
│   └── edge-vision/           # main.rs: parse config, pick platform, run, handle signals
├── models/                    # onnx + build scripts → .engine (gitignored outputs)
├── tools/                     # build-engine.sh, bench, cross-compile docker
└── tests/                     # golden-frame tests run against platform-sim
```

**Platform selection**

- Each `platform-*` crate is an optional dependency of the app behind a feature: `cargo build --features jetson`.
- At runtime `platform = "jetson"` in TOML selects a registered `Box<dyn Platform>`; a build without that feature fails with a clear error, not a link error.
- `platform-sim` is always on, so unit and integration tests never need hardware.

**Code conventions**

- `#![forbid(unsafe_code)]` in `core`, `pipeline`, `postproc`, `sinks`, `apps`.
- Errors: `thiserror` in libraries, `anyhow` only in the app; every FFI status code mapped to a typed error.
- Logging/metrics: `tracing` with per-stage spans; `metrics` + Prometheus exporter.
- Threads, not async, in the frame loop (one thread per stage, `crossbeam` bounded channels); `tokio` only inside sinks/HTTP.
- `clippy -D warnings`, `cargo deny`, and `rustfmt` in CI.

## Jetson Orin backend details

The Jetson backend keeps every frame as an NvBufSurface (NVMM) from sensor to encoder, using VIC for scaling, CUDA for tensor prep, TensorRT (GPU or DLA) for inference and the V4L2 hardware encoder. One hardware difference shapes the design: **Orin Nano has no hardware video encoder (NVENC)**, so the Nano build streams detections only and never hands frames to a CPU encoder — an optional `Encoder` trait makes that a config switch, not a fork.

| Stage | Orin NX | Orin Nano | Rust integration (zero-copy detail) |
| --- | --- | --- | --- |
| Capture | libargus buffer output stream writing into our own pitch-linear NV12 NvBufSurfaces (dmabuf), from M1 | same | `argus-sys` C++ shim via `cxx`; never `copyToNvBuffer` (it copies); confirm the buffer-stream API against the pinned JetPack's Argus samples |
| Pre-process | one fused CUDA kernel: NV12 → letterbox resize → normalise → FP16, written straight into the TensorRT input binding | same | surface → EGLImage → `cuGraphicsEGLRegisterImage` gives a device pointer to the same memory; kernels built by `nvcc` in `build.rs`, launched with `cudarc`. VIC path (`NvBufSurfTransform`, NVMM → NVMM) kept as an option to offload the GPU |
| Inference | TensorRT 10 on GPU or DLA (1–2 DLA cores) | TensorRT on GPU only (no DLA) | `tensorrt-sys` C shim; `setTensorAddress` points at our pooled tensors, `enqueueV3` on the camera's CUDA stream; outputs in `cudaHostAllocMapped` memory |
| Post-process | CPU reads only the small output tensors, in place | same | shared `postproc` crate, borrows `&[Tensor]` |
| Overlay | CUDA kernel draws boxes in place on the capture surface | same | `Overlay` trait, Jetson impl in CUDA |
| Encode | HW H.264/H.265 via V4L2 M2M: output plane `V4L2_MEMORY_DMABUF` importing the capture fd | none: no HW encoder, frames never go to the CPU | `Platform::encoder()` returns `None` on Nano; on NX the `Encoder` takes `HwFrame` by value and requeues it to Argus when NVENC releases it |
| Stream | RTSP / WebRTC video + metadata | metadata only (MQTT / WebSocket) | `sinks` crate: `gst-rtsp-server` with `appsrc`; bitstream wrapped from the V4L2 mmap, no copy |

**Key implementation notes**

- **Engines are device-specific.** A TensorRT `.engine` only runs on the same TensorRT version and GPU it was built for. Ship the ONNX model; build the engine on the device at first boot (`tools/build-engine.sh` wrapping `trtexec`) and cache it keyed by TRT version + model hash.
- **One CUDA stream per camera**, with pre-process, inference and overlay queued on it; the CPU thread only waits on an event. This keeps the CPU near idle.
- **Synchronisation across engines** (VIC, CUDA, encoder) uses `NvBufSurfaceSyncForDevice` / CUDA events; wrap them so a stage cannot forget to sync.
- **Pool sizing:** capture pool ≥ pipeline depth + 2 (e.g. 6 buffers); exhausting the pool drops the oldest frame rather than blocking the sensor.
- **Version pinning:** pin one JetPack release (6.x line, L4T r36) for v1 and record it in `config/`; bindgen headers are generated from that release's `/usr/src/jetson_multimedia_api`.
- **Build:** cross-compile from x86 with a Docker image holding the L4T sysroot, or build natively on the Orin; both covered in CI.
- **Avoid a DeepStream dependency** in v1: it is heavy and its licensing/versions tie you to NVIDIA's release cadence. The backend uses only the Multimedia API, CUDA and TensorRT.

## Porting guide: NXP and Ambarella

Adding a platform means writing one new `platform-<vendor>` crate that implements `Platform` and its four stage factories; `core`, `pipeline`, `postproc` and `sinks` stay untouched. The table maps each stage to the vendor block it would use (approximate — check the exact SoC's reference manual).

| Stage | Jetson Orin | NXP i.MX 8M Plus / 95 | Ambarella CV2x / CV5 / CV7x |
| --- | --- | --- | --- |
| Buffer type | NvBufSurface (NVMM) | DMA-BUF (dma-heap) | IAV buffer / DSP shared memory |
| Capture | libargus | V4L2 / libcamera (ISI + ISP), `VIDIOC_EXPBUF` | IAV driver, ISP pipeline |
| Scale / convert | VIC | G2D (8M Plus) or ISI/PXP | IAV canvas / pyramid, VProc |
| Inference | TensorRT (GPU/DLA) | NPU via TFLite VX delegate (8M Plus), Neutron (i.MX 95) | CVflow (models compiled with Ambarella CNNGen toolchain) |
| Encode | V4L2 NVENC (NX), none on Nano (metadata only) | Hantro VPU via V4L2 M2M (absent on some parts, e.g. i.MX 93) | IAV hardware encoder |
| SDK access | public | public (Yocto BSP) | under NDA |

**Porting checklist**

- [ ] Create `crates/platform-<vendor>` with a `-sys` crate for its C headers and a safe wrapper.
- [ ] Implement `HwBuffer` for the vendor buffer; expose a dmabuf fd if one exists (it is the lingua franca for cross-API import).
- [ ] Implement `FrameSource`, `Preprocessor`, `InferenceEngine`, `Encoder`; return honest `PlatformCaps`.
- [ ] Convert the model with the vendor toolchain; keep the same output tensor layout so `postproc` is reused, or add a small layout adapter.
- [ ] Add a feature flag, a `config/<board>.toml` and a cross-compile Docker target.
- [ ] Pass the shared conformance suite (below) on the board.

**Conformance suite** — a crate of trait-level tests every backend must pass: frame order and timestamps monotonic, pool exhaustion drops instead of deadlocking, golden-image pre-processing within tolerance, detection mAP within 1–2 points of the ONNX reference on a fixed clip, clean shutdown with no leaked buffers.

## Milestones and roadmap

Ship a working Jetson demo early on direct Argus capture (M1), then move every stage onto the zero-copy GPU path; the NXP port in M5 is the real test of the abstraction.

&#91;embedded content: roadmap · 6 phases, each with an exit gate\]

Each phase ends only when its gate is measured on hardware, not when its code is merged.

**First week, concretely**

- [ ] `cargo new --lib` the workspace; add `core`, `pipeline`, `platform-sim`, `apps/edge-vision`
- [ ] Write `HwFrame`, `HwBuffer`, `Tensor` and the five stage traits from the HAL section
- [ ] TOML config schema + loader with validation errors
- [ ] `platform-sim` file source and ONNX Runtime engine; run one YOLO ONNX model on a recorded clip
- [ ] CI: fmt, clippy, tests, `cargo deny`; aarch64 cross-build job
- [ ] Flash the Orin NX with the pinned JetPack; record versions of CUDA, TensorRT, L4T

## Risks, testing and performance

The biggest risks are FFI glue to C++-only SDKs and hidden copies; both are caught early by measuring every stage from M1.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| libargus and TensorRT are C++-only | slow start, unsafe surface grows | thin C shims via `cxx`, kept under \~500 lines each; direct Argus capture from M1 |
| Hidden CPU copies (map/unmap, wrong memory type) | CPU load and latency balloon | `HwBuffer::map_cpu` counted in metrics; CI fails if it is called in the hot loop |
| Orin Nano has no NVENC | no video stream on Nano | Nano profile streams detections only; no CPU encode, no frame copy; use Orin NX where video is required |
| JetPack / TensorRT upgrades break ABI | rebuilds, engine rebuilds | pin JetPack, rebuild engines at boot, headers regenerated per release |
| Abstraction leaks Jetson assumptions | NXP port forces core changes | build `platform-sim` from day one as a second backend; review traits before M4 |
| Ambarella SDK under NDA | cannot publish backend | keep that crate in a private repo, depending on the public core by version |

**Testing layers**

1. Unit tests on `core`, `postproc`, config parsing — run on any machine.
2. Pipeline integration tests on `platform-sim` with a recorded clip and golden detections.
3. Hardware-in-the-loop job on a real Orin NX (self-hosted CI runner): 10-minute soak, FPS, latency, memory growth = 0.
4. Conformance suite per backend (see porting guide).

**What to measure from day one:** per-stage time (capture → pre → infer → post → encode → send), queue depths, dropped frames, `tegrastats` GPU/DLA/EMC load, and process RSS.

