# Jetson M1 bring-up

This is the device-side plan behind `docs/design.md`: what the Orin NX
must prove, which decisions are already fixed, and which tasks remain.
Nothing in this file is claimed as working until it is measured on the
board.

## Pinned target

| Component | Version |
| --- | --- |
| Board | Orin NX 16 GB (Orin Nano is metadata-only strict) |
| JetPack / L4T | JetPack 7 / L4T r39.2.1 |
| CUDA | 13.2.86 (`cuda-cudart`, `cuda-nvcc`, `gcc-for-nvcc`) |
| TensorRT | 10.16.2.10 (`tensorrt-core`; `tensorrt-trtexec` is development-image only) |
| NvBufSurface | `tegra-libraries-multimedia-utils_39.2.1` |
| GPU | compute capability 8.7 |
| Sensors | dual VC MIPI IMX296, 1440×1080 |
| Image | camera-viber Yocto (`yocto/kas`), OE-core Rust 1.98.1 |

Record the exact flashed versions and module/carrier revision in
`config/orin-nx-yolo.toml` (or a sidecar environment record) as the M1
exit artifact: BSP, kernel, CUDA/TensorRT, container digest if any,
model hash, power mode, cooling, ambient, checks run/pending.

## M1 path

```
nvarguscamerasrc → NV12 NVMM → appsink
      → HwFrame (same gst::Buffer / NvBufSurface)
      → [pre → infer → post] (M1: CPU post; GPU stages per T5/T8/T9)
      → encode thread (leaky one-deep hand-off)
      → NX:  appsrc → nvv4l2h26xenc → byte-stream appsink
      → Nano: appsrc → nvvidconv → I420 → x264enc → h264parse → appsink
      → RTSP (M1)
```

- Capture and encode strings are built in
  `crates/platform-jetson/src/launch.rs` and pinned by unit tests.
  `nvv4l2h264enc` defaults to Baseline; the builder sets `profile=2`
  (Main) and derives the GOP from `encoder.gop_frames`. Derive and pin
  `iframeinterval` as well when the GOP is long.
- `appsink` is bounded and dropping (`max-buffers=2 drop=true`);
  `appsrc` is bounded and non-blocking (`block=false max-buffers=2`),
  and the encode `appsink` is bounded too (`max-buffers=8 drop=false`).
  Freshness belongs in the Rust drop-oldest channel, where drops are
  counted; a stalled encoder must return an error, never hang.
- The encode output is asynchronous. `GstEncoder::encode` pushes a
  frame and drains samples non-blockingly; delayed packets are collected
  on later calls and `Encoder::finish` sends EOS and flushes. The runner
  calls it on the encoder thread and drains the packet channel before
  joining, so teardown cannot deadlock.

### Same-buffer NVENC hypothesis (prove first)

On JetPack, `gst_buffer_map(READ)` of an NVMM buffer returns an
`NvBufSurface*` in `map.data`, not pixels. Read
`surfaceList[0].bufferDesc` (dmabuf fd), `.layout`, `.pitch` and
`.planeParams`. This is what fills `FrameMeta::planes`; do not hardcode
`Layout::BlockLinear` in the platform once the surface is available.

If the encoder pushes **the same `gst::Buffer`** received from capture
into the encode `appsrc`, NVENC receives NVMM memory from Argus's own
allocator. This is what happens inside one `nvarguscamerasrc !
nvv4l2h264enc` pipeline. A separate NVMM `GstAllocator` (gst-nvmm's
`nvmmalloc`, LGPL) is needed only when we allocate our own NVMM pool,
i.e. the later direct-libargus path.

Check on the board: does `nvv4l2h264enc` accept a buffer from another
pipeline, are there `NvVideoRegisterMemHandle` errors, and does
`copied_frames == 0` hold? Rebase PTS to the encode pipeline's running
time. If it holds, M1 loses its largest dependency.

## What the shipping C++ product learned (design inputs)

These change traits or the process model, not just implementation.

### Argus failure is a process-level event

- A lost `nvargus-daemon` session never reconnects in-process. Only a
  new process can open the camera again.
- One sensor's fault breaks Argus for every camera in that process.
  Default to **one process per camera** for fault isolation; they share
  the daemon anyway.
- The failure usually arrives as bare EOS; “Failed to create
  CaptureSession” / “Device is in use” go to the journal, not the bus.
  The GStreamer source now returns an error for an `argus` EOS instead
  of a clean `EndOfStream` (PC-tested), so the run cannot exit 0 after
  a dead camera. The once-per-boot policy below is still M1 work.
- Tearing down a pipeline after the daemon went away can segfault
  inside the Argus client. Today `GstLaunchSource::drop` calls
  `set_state(Null)` unconditionally. Before teardown, ask systemd
  whether `nvargus-daemon.service` restarted since the pipeline started
  (`ActiveEnterTimestampMonotonic`); if it did, skip `set_state(Null)`
  and exit.
- Recover once per boot: record the failure, queue
  `systemctl --no-block restart nvargus-daemon`, exit with code 75, and
  let the unit use `Restart=on-failure` with
  `RestartForceExitStatus=75`. A second recovery in the same boot is
  only logged, or a reboot loop follows. Not implemented yet; the
  current code fails the run (exit 1) and relies on `Restart=on-failure`.
- Treat “no first frame within N seconds” as a capture fault, not as
  `Timeout` forever (the runner's `max_idle_ms` is the current generic
  version).

### The capture pool belongs to Argus

`source.pool_size` is a sim concept; on Jetson it does not size the
Argus pool. Every `gst::Buffer` the Rust side holds is a surface Argus
cannot refill. Worst case on the M1 path:

| Holder | Frames |
| --- | --- |
| appsink | 2 |
| Rust channel | 2 (`queue_depth = 2` live) |
| processing | 1 |
| appsrc | 2 |
| encoder in flight | 1+ |
| **total** | **≈8–10** |

Check `nvarguscamerasrc`'s buffer count in the JetPack 7 source. Holding
more than the pool stalls Argus; it shows up as capture timeouts or
“frame took too long” journal errors. Print the in-flight budget at
startup and keep live queues small. Four frames of channel is 133 ms at
30 fps — over the latency budget before any work happens.

There is a second, worse failure: if the capture pool reuses a surface
NVENC is still encoding, NVENC fails with
`NvVideoRegisterMemHandle failed`, then hangs or crashes. An encoder
must hold each `HwFrame` until NVENC releases its `GstBuffer`. An
in-place element behind a tee caused this in C++; in Rust it returns if
any stage drops the frame early.

### Clean video, metadata boxes, inference off the video path

- Do not draw boxes into the capture surface by default. Boxes go out as
  metadata (JSONL today; MQTT/WebSocket later; ONVIF metadata track once
  RTSP exists). In-place drawing writes a surface other readers may
  hold, burns boxes into recordings, and slows the video path.
- Video must not wait for inference. The reference product infers one
  frame in `interval + 1` (`interval=2` on device) and a tracker carries
  boxes between frames.
- The first TensorRT plan build takes minutes after any image or model
  change. The camera streams without inference until the plan is ready.
  `EngineState::Warming` exists for this; the runner pushes empty
  detections until the engine is ready.
- The runner encodes on its own `encode-cam` thread now, behind a
  leaky one-deep frame hand-off. Pre/infer/post/overlay still share the
  processing thread; splitting inference from the video path entirely
  (a one-deep leaky inference side path joining detections by
  `seq`/`pts`) is M1 work before the NVENC stage. Make `Overlay`
  optional and off on Jetson.

### IMX296 on Argus

- AE swings brightness: cap gain at 16 (`gainrange="1 16"`) and/or pin
  exposure. `SourceConfig` has `exposure_us`, `gain_min`, `gain_max`;
  `GstLaunchSource::start` applies them to `camsrc` after the pipeline
  reaches PLAYING, never in the launch string (Nano black-frame fix).
- White balance from the vendor NITO does nothing yet. Expect a green
  cast until the NITO lands; pin the golden-set mAP gate to the tuned
  image, not the first boot.
- The first ~1 s of frames are dark while AE converges. Skip them for
  any golden/mAP comparison and for “first frame” health.
- Cam1 stops after 13–15 s under Argus in the reference product, cause
  open. The “30 FPS × 2 cameras” target is blocked on that hardware
  issue.
- On the Nano, an at-start `gainrange`/`exposuretimerange` produced
  black frames in another app; the image patch fixes range changes while
  PLAYING. Ship the Nano profile without those keys first, check
  brightness, then apply ranges after PLAYING and log the sensor's
  reported limits.
- Orin Nano has no NVENC. The CPU H.264 stage now exists (GStreamer
  `appsrc` → `nvvidconv` → I420 → `x264enc` → `h264parse` → `appsink`;
  one deliberate VIC copy) and `PlatformCaps::cpu_encode` advertises
  the compiled stage, and `encoder()` checks `x264enc` before building
  the string (missing element names
  `gstreamer1.0-plugins-ugly-x264`). A strict profile still refuses it; the Nano
  profile is `config/orin-nano-yolo.toml` with `zero_copy = "relaxed"`
  and `encoder.hardware = false`. Device streaming still depends on the
  NVMM capture branch, which refuses buffers rather than copying them.

### Building the engine on the image

The camera image does not install `trtexec`, and the service sandbox
will not let it run one. The product path is D5: build the plan
in-process through `trt-sys` and cache it under
`/var/cache/edge-vision` (systemd `CacheDirectory=edge-vision`).
`tools/build-engine.sh` is a **bring-up tool** for a development image
only; it defaults the cache to `/var/cache/edge-vision` (override with
`EDGE_VISION_CACHE`), takes a `flock`, limits the builder to a 256 MiB
workspace, refuses to start below 1 GiB `MemAvailable`, and includes
the model, TensorRT, SoC/L4T/CUDA and trtexec-argument hashes in the
cache key. `config/orin-nx-yolo.toml` points at an absolute model path
(`/data/models/yolox-tiny.onnx`); the image must install it there.

### Encoder details

- `nvv4l2h26xenc` defaults to Baseline; set `profile=2` (Main).
- GOP is config-driven (`gop_frames`), including `iframeinterval` for
  long GOPs.
- The nvv4l2 encoders report “Failed to process frame” when a
  flush-start reaches them. A looping file source must restart at EOS,
  not flush-seek.
- Encode output is asynchronous (see above); decide the packet model
  before the NVENC stage.

### Clocks and measurement

The reference product's numbers (3.05 ms inference, 3.9 ms capture to
encoded on NX) came from a CUDA graph plus pinned engine clocks through
devfreq, which also works on fused units. Make “clocks pinned” a
precondition of M1/M2 measurements, and list power mode, cooling and
ambient next to every number.

### Running on the camera Yocto image

- **boot-ok**: the A/B update verifier counts a boot as good only when
  `/run/camera-streamer/boot-ok` is written. Write it only after a real
  health check (a frame from an enabled camera), not after “up 60 s”.
- **systemd watchdog**: `WatchdogSec=15`; the watchdog pet must not sit
  behind disk I/O.
- **Sandbox / camera-guard (BPF LSM)**: the service runs non-root under
  `ProtectSystem=strict`. camera-guard limits exec/write/connect. A new
  binary needs its own entries in `ebpf/camera-guard/guard.h` and grants
  for the `video` group (`/dev/nvhost-*`, the Argus socket).
- **Disk stalls**: `/recordings` is a DRAM-less NVMe that stalls under
  sustained writes. `FileVideoSink`/`JsonlMetaSink` write synchronously
  on the processing thread today. Move disk writes to a bounded writer
  thread before running on the image.
- **Ports**: use RTSP port 8555; 8554 belongs to camera-streamer. The
  `rtsp_port` config field is removed until the RTSP sink exists.
- **Toolchain**: OE-core ships Rust 1.98.1; the workspace MSRV is 1.92
  and `tools/Dockerfile.cross` uses 1.98. The recipe `nv-gst-rust` (OE's
  `cargo` class, crates fetched offline from `Cargo.lock`) builds it:
  [yocto.md](yocto.md).
- **Model licence**: YOLOX (Apache-2.0) is the device default and
  `config/orin-nx-yolo.toml` points at it. Ultralytics YOLOv8 weights
  are AGPL-3.0 and stay sim/test-only.

## Inference stages (TensorRT/CUDA/VIC)

The reference pieces already run in the C++ product
(`TensorRtEngine`, `NvmmFramePreprocessor`, `NvmmGpuTensorWriter`,
`CudaTensorKernels.cu`, `BuildProgressAdapter`). Port behaviour, not
layout.

### Decisions

| # | Decision | Why |
| --- | --- | --- |
| D1 | Two `-sys` crates: `nvbuf-sys` (hand-written read-only `repr(C)` view for T2; bindgen only when transform APIs land) and `trt-sys` (C++ shim through `cxx`) | NvBufSurface is C with fields to read; TensorRT is C++ only |
| D2 | CUDA calls live inside the C++ shim, not `cudarc` | ~12 calls sharing lifetimes with TensorRT; revisit after M1 |
| D3 | I/O tensors in mapped host memory (`cudaHostAlloc(..., cudaHostAllocMapped)`) | integrated memory; CPU sees `&[f32]`, GPU sees a device pointer, `edge-postproc` unchanged |
| D4 | Default pre-processing: VIC scales, CPU writes the planar tensor; CUDA kernel behind `cuda-preprocess` | reference default/fallback; kernel needs nvcc + EGL |
| D5 | Build the plan in-process (builder + `nvonnxparser`, progress callback), cached, warmed on its own thread | progress, build lock, no extra binary on the image |
| D6 | Video never waits for the plan | `EngineState::Warming` |
| D7 | One `IExecutionContext` and one CUDA stream per camera | TensorRT threading rules |
| D8 | `unsafe` only in `-sys` crates | reviewability |

### Crate layout

```text
crates/platform-jetson/
├── nvbuf-sys/              # unsafe: repr(C) read-only view; no link, no pixel copy
├── trt-sys/                # unsafe: cxx bridge + C++ shim (+ .cu behind a feature)
└── src/                    # safe: #![forbid(unsafe_code)]
    ├── nvmm.rs             # NvmmSurface::from_gst: fd, layout, pitch, planes
    ├── vic.rs              # VicScaler into a pooled input-size surface
    ├── cuda.rs             # CudaStream, MappedTensor, GpuTimer
    ├── trt.rs              # TrtEngine (InferenceEngine)
    ├── plan_cache.rs       # key/path/atomic write/prune (done on PC)
    ├── warmup.rs           # engine-warm thread + process-wide build lock
    └── preprocess.rs       # JetsonPreprocessor: VIC + planar write
```

Features: `gst` exists; `trt` adds the `-sys` crates and needs the L4T
sysroot; `cuda-preprocess` needs nvcc; without `trt`, the crate builds
on any PC.

### Tasks and status

| Task | What | Status / done when |
| --- | --- | --- |
| **C1** | Opaque `GpuStream` created by `Platform::stream`; `Arc<dyn Any>` downcast by the backend | **Done** on PC (sim no-op; downcast test) |
| **C2** | `EngineState::Warming`; runner flows empty detections until ready | **Done** on PC (`DelayedSimPlatform` test) |
| **C3** | Builder checks engine input shape/dtype against config | **Done** |
| **C4** | Shared letterbox geometry/normalisation used by sim, device and decoders | **Done** (`core/src/letterbox.rs`) |
| **T0** | Build against the Yocto SDK sysroot (`$SDKTARGETSYSROOT`/`JETSON_SYSROOT`); `bindgen --sysroot`; cxx/nvcc cross; `tools/Dockerfile.jetson-sdk`, Rust 1.98 | Not started; success = cross build of `--features trt` in the SDK container; clear error without a sysroot |
| **T1** | `nvbuf-sys`: read-only `repr(C)` view for surface fd/layout/planes (done); bindgen/link lines for `NvBufSurfTransform`/EGL when T8/T9 land | **PC part done** (layout tests on fabricated surfaces); transform/EGL bindings wait for T0/T8/T9 |
| **T2** | `NvmmSurface::from_gst`: check `map.size() == size_of::<NvBufSurface>()`; read fd/layout/pitch/planes while the `GstBuffer` is held; replace the NVMM refusal; `FrameMeta.planes` from the surface | **PC implementation done** (`nvbuf-sys` + `GstFrameBuffer::Nvmm`). Device: `gst-capture` logs fd/layout/pitch, `copied_frames == 0` over 300 frames, stable fd |
| **T3** | `trt-sys` skeleton: logger, version, load, io_infos, context, enqueue, synchronize; exceptions → `Result` | Not started; device: load a `trtexec` FP16 plan and enqueue once |
| **T4** | CUDA buffers: `host_alloc`/`bind`; `MappedTensor`; `CudaStream` | Not started |
| **T5** | `TrtEngine`: bind every I/O tensor to mapped memory, batch 1, refuse non-float I/O, `infer` = enqueue + synchronize; outputs with real specs | Builder-side binding checks **done** (one static F32 NCHW input, at least one static F32 output); engine stage not started. Device: FP32 within 1e-3 / FP16 within 1e-2 of ONNX Runtime, identical boxes |
| **T6** | Plan cache (done) + shim `build_plan` (builder + `nvonnxparser`, FP16, progress) | Cache half **done** (`plan_cache.rs`; key includes SoC/L4T/CUDA/GPU; plans live under `<cache>/<model_id>/`; exclusive temp files). Shim half waits for T3/T5. INT8 stays out of caps until calibration is in the key |
| **T7** | Engine-warm thread under a process-wide lock; `engine()` returns Warming until the plan exists | Waits for T5/T6 |
| **T8** | VIC preprocessor: pooled pitch-linear RGBA surfaces, `NvBufSurfTransform` letterbox, CPU planar write | Waits for T2/T4 |
| **T9** | CUDA preprocessor: `NvBufSurfaceMapEglImage`, `cuGraphicsEGLRegisterImage`, kernel into the input binding | Waits for T8; output equals T8 within 1e-6 |
| **T10** | CUDA graph after the first good inference; GPU time from events; fall back to `enqueueV3` once on capture failure | Waits for T5 |
| **T11** | Conformance on the board + Nsight/`tegrastats` zero-copy gate | Waits for T5–T8; M1 inference exit gate |
| **T12** | Packaging: OE recipe, `DEPENDS`, systemd `CacheDirectory`, video group, camera-guard grants | Recipe `nv-gst-rust` written in camera-viber and built for NX and Nano ([yocto.md](yocto.md)); proposal units now in `tools/systemd/` (Conflicts=, CacheDirectory); camera-guard grants and installation still image-owner work |

**Order:** T0 first, then T1→T2 and T3→T4→T5 in parallel until T8 joins
them. T6/T7 follow T5; T9/T10 are optimisations. The first board
session runs T2 and T3 before investing in T6–T10.

### Reference: gst-nvmm-cpp

[gst-nvmm-cpp](https://github.com/PavelGuzenfeld/gst-nvmm-cpp) (read at
commit `060a8c0`, 2026-10-24) has NVMM-native GStreamer elements in C++14
without DeepStream: an NvBufSurface allocator, VIC convert/composite, and
TensorRT inference on NVMM. It is validated on Xavier NX (JP5.1) and Orin
NX (JP6.2, CUDA 12.5, TRT 10.3) only. Nothing covers JP7, CUDA 13 or TRT
10.16, and it has no Argus source, so it gives no guidance on the capture
pool. Where it and camera-viber disagree, camera-viber wins: it runs on
JP7.

**Licence: LGPL-2.1-or-later.** Reimplement the API call sequences below
from NVIDIA's headers and camera-viber. Never copy its code into this
MIT/Apache workspace: not `preprocess.cpp`, `gstnvmminfer.cpp`,
`trt_engine.cpp`, `nvbufsurface_mock.h`, the allocator and pool, or its
docs' wording.

**Adopt (patterns):**

| Task | What the reference does (file:line under `gst/`) | Take for us |
| --- | --- | --- |
| T8, T9 | It never maps the capture surface into CUDA. VIC converts NV12 into a private pitch-linear RGBA surface at network size (`nvmminfer/preprocess.cpp:12-52`). That surface is mapped once: `NvBufSurfaceMapEglImage`, `cudaGraphicsEGLRegisterImage`. camera-viber's `NvmmGpuTensorWriter` does the same on JP7. | VIC into our own scratch surface, mapped once at configure; never the capture buffer per frame |
| T2 | Pitch, offset and psize come from `surfaceList[0].planeParams` (`nvmmalloc/gstnvmmbufferpool.cpp:112-121`). Real pitch differed from GstVideoInfo's stride on Xavier. Block-linear is only de-tiled by VIC; its bytes are never read. | `FrameMeta.planes` from `planeParams` with layout as a field; tight NV12 only in the host fake; refuse block-linear outside VIC |
| T9 | It branches on the mapped EGL frame type: pitch (`pPitch[0]`) or array (`cudaMemcpy2DFromArrayAsync`) (`preprocess.cpp:98-110`). | Handle both frame types; the array case is unverified on JP7 |
| T4, T5 | One CUDA stream per element and one `cudaStreamSynchronize` per frame, before the buffer returns to the pool (`nvmminfer/gstnvmminfer.cpp:145, 406`). No `SyncForDevice` on the GPU path. | One stream per camera; one sync per frame before the buffer guard drops; `SyncForDevice` only after a CPU write (camera-viber's letterbox fill) |
| T8 | Letterbox in one VIC call with `CROP_DST` and a destination rectangle; the padding is filled once and VIC never overwrites it (`preprocess.cpp:73-88`). | The same. Set the VIC session params on **every** call: they are per thread (camera-viber `NvmmFramePreprocessor`). The reference calls them process-global and rarely sets them |
| T8, T9 | Float/planar conversion with NPP `_Ctx` calls on the stream (`preprocess.cpp:119-141`). | Our own small CUDA kernel (camera-viber's `CudaTensorKernels` pattern): NPP would be one more `-sys` crate, and its CUDA 13 status is unknown |
| T5 | It rejects anything but one input and one output, FP32 I/O, static NCHW 1x3xHxW. An output whose shape does not match is rejected, so the decoder cannot over-read (`gstnvmminfer.cpp:167-196`). | The same checks as builder errors and tests. FP16 plans: kFP16 builder flag only, I/O stays FP32 |
| T11 | A header-only, calloc-backed NvBufSurface mock lets the same sources build on x86 (`common/nvbufsurface_mock.h`). Its pitch is tight and its transform a stub, so it missed the real-pitch bug. | A Rust trait fake of a narrow HAL (create, map, transform, sync), with deliberately padded pitch (e.g. rounded to 256) and a block-linear flag that scrambles bytes. One contract suite runs against the fake and the board |

**Avoid:**

- Its TensorRT lifecycle (T6, T7, T10): engines prebuilt by `trtexec`,
  with no cache, no warm-up and no build lock. It also copies the outputs
  to the host and synchronizes every frame, about 23 ms per frame at 1080p
  on Orin (its B5 design doc, phase 1). Follow camera-viber's
  `docs/INFERENCE-ENGINE.md` instead: the engine-warm thread, plan and
  timing cache, mapped host memory, and a CUDA graph after warm-up.
- Its `dla-core` property: a no-op for prebuilt engines, and DLA is no
  option with TRT 10.16 on JP7 (camera-viber `docs/INFERENCE-ENGINE.md`).

**Pitfalls it documents (check each on JP7):**

- `numFilled` is 0 after `NvBufSurfaceCreate`; Map and Transform then
  reject index 0. Set it after every create (T1, T8).
- `NvBufSurfaceCopy` does not de-tile block-linear; use
  `NvBufSurfTransform`.
- `NvBufSurfTransform` rejects some CUDA-memory destinations; use a
  surface-array (`NVBUF_MEM_DEFAULT`) RGBA destination.
- A resolution change needs the pool drained before it is recreated.
- `tee` or `make_writable` on an NVMM buffer whose memory is no-share
  copies it or fails. This touches the encoder hand-off (E5 below).
- A frame whose inference was skipped must carry no stale detections.

## Safety rules for FFI

- Ownership order is explicit in field declaration order (context before
  engine before runtime) and commented.
- No panic crosses FFI: every `extern "Rust"` callback is wrapped in
  `catch_unwind`; every C++ entry point catches exceptions and maps them
  to `Result`.
- Raw pointers never leave `-sys` crates; `host_ptr` becomes
  `&mut [f32]` with the buffer's lifetime in exactly one function.
- Sticky CUDA errors mark the context broken; `infer` returns `Err` from
  then on and the runner rebuilds at most once.
- Bindings are generated at build time from the pinned sysroot, never
  committed, and checked with layout tests.
- One context per camera thread; set the CUDA device on each thread that
  enqueues work.

## Testing on the device

| Layer | Runs on | Covers |
| --- | --- | --- |
| Unit | PC/CI | plan key/path/prune/atomic write, plane maths, letterbox, shape checks, warming engine, `MappedTensor` with a fake allocator |
| Golden | PC → device | PC dumps input tensor and ONNX Runtime outputs for a fixed frame; device compares T5/T8/T9 within tolerances |
| Device `#[ignore]` | Orin | T2–T10 checks via `cargo test -p platform-jetson --features trt -- --ignored` |
| Conformance + trace | Orin | T11 |
| CI | PC | default features + compile-only `--features trt` in the SDK container, so the shim cannot rot like the `gst` feature did |

## Open questions

1. **`cudarc` instead of CUDA in the shim (D2).** Check support for CUDA
   13 on aarch64. If it is solid and the shim's CUDA part grows, move
   streams/events/memory there.
2. **DLA** (Orin NX has two). Out of scope for M1; reserve the builder
   flag and `model.device = "dla0"` hook.
3. **INT8** needs a calibration set and its own accuracy gate. M1 ships
   FP16.
4. **Async inference.** M1 synchronises inside `infer`. Overlapping
   pre-processing of N+1 with inference of N needs events and a two-slot
   input; only do it if T10's numbers justify it.

## Remaining findings from the 2026-10-24 device review

Resolved in the PC pass: the CPU H.264 encoder stage exists with an
element check and a Nano profile (N2/N16); encoding runs on its own
thread with a leaky one-deep hand-off (N7); input dtype is no longer
tied to execution precision (N6); Argus EOS fails the run instead of
exiting 0 (N8, first half); the copy-fallback pool is lazy and `O_EXCL`
(N10); plan cache keys include SoC/L4T/CUDA/GPU (N13); INT8 is not
advertised without calibration (N15); encode appsinks are bounded and
x264 is capped at three threads (N17); the engine-build script uses
`EDGE_VISION_CACHE`, a 256 MiB workspace, a build lock and a memory
check (N1/N3/N4/N5).

Still device-gated:

| Item | Status / next step |
| --- | --- |
| N9 teardown after daemon loss | PC implementation: typed `CaptureServiceLost`, teardown skipped after Argus EOS, once-per-boot marker + `systemctl --no-block restart nvargus-daemon` + exit 75. Device verification pending |
| N11 in-flight budget | `source.argus_pool` + builder validation and a startup `live in-flight` line; the board's exact pool size still has to be recorded |
| N14 module identity | Done: `probe_module_identity` + `dla_cores`; caps add `dlaN` only when `/dev/nvhost-nvdlaN` exists |
| N18 VIC clock floor | Proposal unit added at `tools/systemd/edge-vision-clocks.service`; install with the recipe |
| N19 cross build | Docker now carries the aarch64 C++ toolchain; `--features jetson`/`trt-sys` still need the Yocto SDK (T0) |
| N20 disk writer | `sinks::AsyncSink` moves writes to a bounded `sink-writer` thread; the app uses it |
| N21 sensor ownership | Proposal unit uses `Conflicts=camera-streamer.service MainApp.service lpr-app.service`; install with the recipe |
| X1 same-`gst::Buffer` NVENC | First board session, as above |
| X3 DLA | `model.device = "dlaN"` is accepted only when the probe finds `/dev/nvhost-nvdlaN`; TensorRT execution still M1 |
| X4 two cameras | Blocked on the cam1 Argus timeout |
| Nano measurement | Record `tegrastats`, process CPU, `edge_encode_dropped_total` and the x264 `threads=3` setting alongside the H.264 stream once the NVMM/VIC stages run on the board |

## Findings from the 2026-10-24 reviews (rules and encoder)

Two reviews after the N/X pass. The rules review checked the workspace
against camera-viber's frame-path, inference-engine and embedded rules
and this repo's zero-copy contract. The encoder review covers commit
`489c4503` (the Nano CPU H.264 stage). Items the table above already
tracks are not repeated.

**Rules review: what follows the rules.** No CPU pixel path on the Jetson
route; the only copy out of NVMM is the CPU-encoder branch. Caps are
`memory:NVMM,NV12`, and a frame holds its `gst::Buffer` until the encoder
releases it. The plan cache key and integrity check are complete for
what exists. No `unsafe` anywhere; `#![forbid(unsafe_code)]` in every
crate but platform-sim. State changes run on the capture thread. Vendor
libraries stay in `platform-*`. Not yet written, so not reviewed: T2–T10.

### Encoder stage (commit `489c4503`)

| ID | Severity | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| E1 | blocker | `gst_encoder.rs` (bus never read), `launch.rs` `encode_launch` (`block=true max-buffers=2`) | When any element after the `appsrc` fails (caps not negotiating, likely on the first NVMM run), `push_buffer` queues two buffers and then blocks forever. Reproduced on the PC: three pushes OK, the fourth still blocked after 3 s, bus "Internal data stream error". The encoder thread hangs inside `encode()`, `encoder_failed` is never set, video stops silently, two NVMM buffers are held, and teardown's `join` hangs on SIGTERM. `finish()` waits 5 s and returns `Ok` on the same error. | Pop bus errors after each push and in `finish`; never block unbounded (`block=false` and count a full queue as a drop, or set the pipeline to `Null` from the stop path); add a test with `new_with_launch` and a downstream caps mismatch that must fail and join in bounded time |
| E2 | major | `gst_encoder.rs` `finish` | After EOS the pipeline, `appsrc` and `appsink` stay set; a second `run()` pushes into an ended `appsrc` and fails with `Eos`, although the runner hands the encoder back "so the pipeline can run again". | End `finish` with `set_state(Null)` and clear the three fields; test two runs |
| E3 | minor | `pipeline/src/lib.rs` (encoder thread) | `encode_total` now times `push_buffer` plus a non-blocking drain, not x264's work; on the Nano it will look tiny. | Rename (`encode_push`) or measure from appsink timestamps |
| E4 | minor | `gst_encoder.rs` `drain` | Packets are drained only after a push; when the hand-off drops frames, finished packets wait for the next frame. | Also drain (non-blocking) in the `recv_timeout` `None` branch |
| E5 | unverified (device) | `gst_encoder.rs` `encode` | The `appsrc` gets a second ref of the capture buffer; base source marks the first buffer DISCONT, which makes it writable, so it is copied. With no-share NVMM memory that is a deep copy or a failure (the gst-nvmm-cpp `tee` pitfall). | Log the memory flags and whether the first pushed buffer was copied on the first Nano run |
| E6 | unverified (device) | encoder path | Up to four more capture buffers in flight (hand-off 1, `appsrc` queue 2, `nvvidconv` 1) on top of the capture and processing queues. | Count it in N11's budget against the Argus pool |
| E7 | minor | `platform-jetson/src/lib.rs` `caps` | The `cpu_encode` comment says "only when the image actually has x264's GStreamer element"; the code is `cfg!(feature = "gst")` and the element check is in `encoder()`. | Fix the comment |
| E8 | minor | `launch.rs`, `gst_encoder.rs` `required_elements` | The CPU H.265 launch string and the x265 row are unreachable now that `encoder()` refuses software H.265. | Remove, or mark as unused |

Resolved by the same commit: video no longer stops silently when the
encoder *returns* an error (the `encoder_failed` flag and the disconnected
packet channel stop the run); E1 is the case where it hangs instead.

### Rules review

| ID | Severity | Where | Finding | Fix |
| --- | --- | --- | --- | --- |
| R1 | minor | `launch.rs` `capture_launch` (`exposuretimerange`, `gainrange`) | A profile that sets the ranges still writes them into the launch string at start, which gave black frames on the Nano; the Nano profile omits them, but the code allows it. | Set them on `camsrc` after PLAYING, as this doc's IMX296 section says |
| R2 | minor | `plan_cache.rs` `prune_except` | Deletes every other `.plan` in the directory; two models or cameras evict each other's plans and rebuild for minutes. | Prune only plans of the same model (key prefix) |
| R3 | minor | `plan_cache.rs` (temp file) | `.{key}.tmp-{pid}` opened with `truncate(true)`, not `create_new`; two builders in one process clash. | `create_new`, plus T7's lock |
| R4 | minor | `pipeline/src/lib.rs` `check_input_spec` | Accepts F16 input, but no F16 writer exists and the mapped-host path is F32. | Accept F32 only until an F16 writer exists (T5) |
| R5 | minor | `gst_source.rs` `GstFrameBuffer::Copied::memory()` | A CPU copy reports `DmaBuf`, claiming the zero-copy domain. Harmless on the Jetson. | Report an honest memory tag |
| R6 | minor | `crates/core/Cargo.toml` `cpu-debug` | Nothing enforces "never in release"; feature unification could enable it for the shipped app through a dependency. | CI: `cargo tree -e features -p edge-vision` must not show `cpu-debug` |
| R7 | minor | `launch.rs` `file_capture_launch` | `location=` is unquoted; a path with a space or `!` breaks or injects elements. | Set the property instead of formatting it |
| R8 | minor | `config/orin-nx-yolo.toml` | No `max_idle_ms` (the Nano profile has 5000); a sensor that never delivers a first frame is caught only by the default. | Set it in the NX profile too |
| R9 | minor | `tools/build-engine.sh` | Leaves `$engine.tmp` behind when `trtexec` fails under `set -e`. | `trap` to remove it |
| R10 | minor | `apps/edge-vision/src/main.rs` | The unknown-platform error says "known: sim", though `jetson` exists. | List both |

**Resolved in this pass:** R1–R10, E1–E5, E7, T2, N9, N11, N14,
N18–N21 and X3.
- R1–R10: ranges after PLAYING, per-model cache namespaces and
  `create_new` temp files, F32-only input, honest `shm-copy` tag,
  release guard + CI feature check, file path as a property, NX
  `max_idle_ms`, engine temp cleanup and the platform error text.
- E1: non-blocking appsrc, bus polling and a bounded caps-mismatch
  test; a hard error for every negotiation failure still needs the
  board. E2: encoder reusable after `finish`. E3: `encode_push`
  metric. E4: idle packet drain. E5: first-buffer flags/writability
  log. E6: in-flight budget. E8 remains a documented future hook.
- T2/T1: `nvbuf-sys` read-only surface view, exact map-size check and
  `GstFrameBuffer::Nvmm` with plane params. T5 binding checks are in
  the builder/conformance suite.
- N9: typed capture loss, skipped teardown and once-per-boot recovery.
  N11: `argus_pool` validation and startup budget. N14/X3: module and
  DLA probing. N18/N21: systemd proposals. N19: cross C++ toolchain.
  N20: async sink writer.
Still open: E1 device verification, T0 SDK build, T3/T4/T5
(TensorRT/CUDA engine), T8/T9/T10 (VIC/CUDA pre-process and graph),
T11 on-board conformance, T12 installation/camera-guard grants,
X1 NVENC dmabuf, the Nano measurement and X4's cam1 hardware issue.
## First device run: Orin Nano, 2026-10-24

Board: Orin Nano 8 GB (p3767-0003) on the LPR image (L4T r39.2.1),
sensor 0 (IMX296). MainApp was stopped for the test and started again
afterwards. Binary: the `gst-capture` example, cross-built by the
camera-viber recipe with `--features gst --example gst-capture`.
`edge-vision` itself cannot run on the device yet: the Jetson
preprocessor, engine and overlay return "M1 pending".

| Check | Result |
| --- | --- |
| Launch string | `nvarguscamerasrc name=camsrc sensor-id=0 ! video/x-raw(memory:NVMM),format=NV12,width=1440,height=1080,framerate=30/1 ! appsink name=frames max-buffers=2 drop=true sync=false`, accepted by GStreamer and Argus |
| First buffer | Argus delivered it; `GstLaunchSource` refused it: "NVMM buffer without dmabuf memory: the NvBufSurface branch is not wired yet; refusing to copy". 0 frames reached the app |
| Exit | The panic aborted the process (`panic = "abort"`, exit 134). nvargus-daemon logged "CameraProvider was not destroyed", cleaned up the session, and `SCF: Error 0x00000008` (critical error event for the session) |
| After | MainApp restarted and processed frames again (its LPR/motion timings resumed); no reboot was needed this time |

What it means:

- **T2 is the first blocker on JP7, and it is confirmed.**
  nvarguscamerasrc's NVMM `GstMemory` is not a dmabuf memory, so the
  "same buffer via dmabuf" path never applies. `from_gst` must map the
  memory, check `map.size() == size_of::<NvBufSurface>()` and read the fd
  and planes from the NvBufSurface (gst-nvmm-cpp reference, T2 row).
- **An unclean exit tears down the Argus session.** Here the next client
  recovered, but the LPR app has seen sensor 1 fail with Error 8 until a
  reboot after the same kind of exit. Device tools must stop the
  pipeline before they exit: no `expect`/panic after `start()` in the
  example. Return an error, call `stop()`, then exit non-zero.
- Next board session: rerun this check after T2. The PC implementation
  now parses `NvBufSurface` via `nvbuf-sys` and exposes the fd/layout/
  planes; pass = 300 frames, `copied=0`, a stable fd set, layout and
  pitch logged.
