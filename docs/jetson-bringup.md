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
  `appsrc` is bounded/blocking (`block=true max-buffers=2`), and the
  encode `appsink` is bounded too (`max-buffers=8 drop=false`). Freshness
  belongs in the Rust drop-oldest channel, where drops are counted.
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
- The current runner still runs `pre → infer → post → overlay → encode`
  in series on one thread. Splitting the video path
  (capture → encode → sink) from a one-deep leaky inference side path
  that joins detections by `seq`/`pts` is M1 work before the NVENC
  stage. Make `Overlay` optional and off on Jetson.

### IMX296 on Argus

- AE swings brightness: cap gain at 16 (`gainrange="1 16"`) and/or pin
  exposure. `SourceConfig` has `exposure_us`, `gain_min`, `gain_max`,
  rendered into the capture string.
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
| D1 | Two `-sys` crates: `nvbuf-sys` (bindgen over C headers) and `trt-sys` (C++ shim through `cxx`) | NvBufSurface is C with fields to read; TensorRT is C++ only |
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
├── nvbuf-sys/              # unsafe: bindgen output + link lines; no logic
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
| **T1** | `nvbuf-sys`: bindgen allowlist (`NvBufSurface*`, `NvBufSurfTransform*`, `NvBufSurfaceMapEglImage`), link lines, layout tests | Not started |
| **T2** | `NvmmSurface::from_gst`: check `map.size() == size_of::<NvBufSurface>()`; read fd/layout/pitch/planes; hold the mapping; replace the NVMM refusal; `FrameMeta.planes` from the surface | Not started; device: `gst-capture` logs fd/layout/pitch, `copied_frames == 0` over 300 frames, stable fd |
| **T3** | `trt-sys` skeleton: logger, version, load, io_infos, context, enqueue, synchronize; exceptions → `Result` | Not started; device: load a `trtexec` FP16 plan and enqueue once |
| **T4** | CUDA buffers: `host_alloc`/`bind`; `MappedTensor`; `CudaStream` | Not started |
| **T5** | `TrtEngine`: bind every I/O tensor to mapped memory, batch 1, refuse non-float I/O, `infer` = enqueue + synchronize; outputs with real specs | Not started; device: FP32 within 1e-3 / FP16 within 1e-2 of ONNX Runtime, identical boxes |
| **T6** | Plan cache (done) + shim `build_plan` (builder + `nvonnxparser`, FP16, progress) | Cache half **done** (`plan_cache.rs`, key includes SoC/L4T/CUDA/GPU); shim half waits for T3/T5. INT8 stays out of caps until calibration is in the key |
| **T7** | Engine-warm thread under a process-wide lock; `engine()` returns Warming until the plan exists | Waits for T5/T6 |
| **T8** | VIC preprocessor: pooled pitch-linear RGBA surfaces, `NvBufSurfTransform` letterbox, CPU planar write | Waits for T2/T4 |
| **T9** | CUDA preprocessor: `NvBufSurfaceMapEglImage`, `cuGraphicsEGLRegisterImage`, kernel into the input binding | Waits for T8; output equals T8 within 1e-6 |
| **T10** | CUDA graph after the first good inference; GPU time from events; fall back to `enqueueV3` once on capture failure | Waits for T5 |
| **T11** | Conformance on the board + Nsight/`tegrastats` zero-copy gate | Waits for T5–T8; M1 inference exit gate |
| **T12** | Packaging proposal: OE recipe, `DEPENDS`, systemd `CacheDirectory`, video group, camera-guard grants | Proposal only; do not change the image unasked |

**Order:** T0 first, then T1→T2 and T3→T4→T5 in parallel until T8 joins
them. T6/T7 follow T5; T9/T10 are optimisations. The first board
session runs T2 and T3 before investing in T6–T10.

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
| N9 teardown after daemon loss | `set_state(Null)` still unconditional; add the `nvargus-daemon` restart check and skip teardown when it fired |
| N11 in-flight budget | Table above is the design; add a validated Jetson budget and log it at startup |
| N14 module identity | Probe `/proc/device-tree/compatible`, `MemTotal` and DLA devices; derive caps from them |
| N18 VIC clock floor | Propose `edge-vision-clocks.service` with the recipe; no unit is shipped |
| N19 cross build | Docker covers the workspace/sim only; `--features jetson` and `trt-sys` need the Yocto SDK (T0) |
| N20 disk writer | Sinks still write/flush on the processing thread; add a bounded writer thread |
| N21 sensor ownership | No check that camera-streamer/lpr-app is inactive; add a startup check or unit `Conflicts=` |
| X1 same-`gst::Buffer` NVENC | First board session, as above |
| X3 DLA | Deferred; reserve the builder flag and `model.device` hook |
| X4 two cameras | Blocked on the cam1 Argus timeout |
| Nano measurement | Record `tegrastats`, process CPU, `edge_encode_dropped_total` and the x264 `threads=3` setting alongside the H.264 stream once the NVMM/VIC stages run on the board |
