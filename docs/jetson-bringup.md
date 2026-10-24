# Jetson M1 bring-up

This is the device-side plan behind `docs/design.md`: what is still open
on the Orin NX and Orin Nano, the constraints the open work must respect,
and the facts measured on the board so far. Finished work is not listed
here; `docs/design.md` describes what exists. Nothing open in this file
is claimed as working until it is measured on the board.

## Pinned target

| Component     | Version                                                                                            |
| ------------- | -------------------------------------------------------------------------------------------------- |
| Boards        | Orin NX 16 GB (NVENC); Orin Nano 8 GB p3767-0003 (no NVENC: CPU H.264, `zero_copy = "relaxed"`)    |
| JetPack / L4T | JetPack 7 / L4T r39.2.1                                                                            |
| CUDA          | 13.2.86 (`cuda-cudart`, `cuda-nvcc`, `gcc-for-nvcc`)                                               |
| TensorRT      | 10.16.2.10 (`tensorrt-core`; `tensorrt-trtexec` is development-image only)                         |
| NvBufSurface  | `tegra-libraries-multimedia-utils_39.2.1`                                                          |
| GPU           | compute capability 8.7                                                                             |
| Sensors       | dual VC MIPI IMX296, 1440×1080                                                                     |
| Image         | camera-viber Yocto (`yocto/kas`), OE-core Rust 1.98.1; recipe `nv-gst-rust` ([yocto.md](yocto.md)) |

Record the exact flashed versions and module/carrier revision in
`config/orin-nx-yolo.toml` (or a sidecar environment record) as the M1
exit artifact: BSP, kernel, CUDA/TensorRT, container digest if any,
model hash, power mode, cooling, ambient, checks run/pending.

## Measured on the board

Orin Nano 8 GB (p3767-0003), LPR image, L4T r39.2.1, sensor 0, 2026-10-24.
Capture rows use the `gst-capture` example; inference rows use `edge-vision`
with `config/orin-nano-yolox.toml` (YOLOX-S 640×640 FP16, TensorRT 10.16.2,
Apache-2.0 model from camera-viber's yolox-s package). These are reference
values for the open work below.

| Item         | Value                                                                                                                                                                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Capture      | 300/300 frames, 30.0 fps, `copied=0`                                                                                                                                                                                                                        |
| Surface      | block-linear NV12, 1440×1080; the mapped NVMM memory is exactly one `NvBufSurface` (72 bytes)                                                                                                                                                               |
| Planes       | Y: offset 0, pitch 1472, 1080 rows. UV: offset 1736704, pitch 1472, 540 rows (not `pitch × 1080`)                                                                                                                                                           |
| Argus pool   | **6 buffers** (fds rotate through 6 values)                                                                                                                                                                                                                 |
| Clean exit   | no nvargus-daemon warnings when the pipeline is stopped before exit                                                                                                                                                                                         |
| Unclean exit | a panic after `start()` aborted the process; nvargus-daemon logged "CameraProvider was not destroyed" and `SCF: Error 0x00000008`. The next client recovered here, but the LPR app has seen sensor 1 stay broken until a reboot after the same kind of exit |
| `edge-vision` view profile | inference off, x264, `TcpH264Sink` on :8555, 60 s: 1780 frames ≈ 30 fps, `frames_dropped` 0, `capture_timeouts` 0, 1779 packets, keyframe every 30 frames, ≈ 8 Mbit/s |
| Viewer | `ffmpeg -f h264 -i tcp://192.168.55.1:8555`: 301 frames in 10 s, Constrained Baseline 1440×1080, sharp and correctly exposed; a viewer disconnect logs "viewer dropped: Broken pipe" and the run continues |
| Budget | startup warning "in-flight budget is 8 surfaces (static worst case) but the Argus pool has 6", yet 0 timeouts |
| Encoder buffer | first buffer `flags=DISCONT writable=false`: no copy of the capture buffer (E5) |
| Exit | SIGINT exits 0; no "CameraProvider was not destroyed" |
| Image | `h264parse` was missing; `gstreamer1.0-plugins-bad-videoparsersbad` installed from the feed (the camera-viber recipe now RDEPENDS on it) |
| Zero copy up to the encoder | `GST_DEBUG=appsink:6,appsrc:6`, 6 s: 163 of 163 buffers queued into the encoder appsrc are the same `GstBuffer` objects the capture appsink handed out (same address, 0 from elsewhere); only 6 distinct buffers, recycled (the Argus pool); every one is 72 bytes, the `NvBufSurface` descriptor, so the pixels stay in NVMM. The only copy is `nvvidconv` (VIC, NVMM → I420) for x264, the Nano's `relaxed` opt-in |
| TensorRT plan (trtexec precursor) | `tools/build-engine.sh` with `trtexec --fp16` on the LPR app's plate/vehicle detector (`mmdet_07032024.onnx`, private model, stays on the camera): built in 284 s, 5.4 MB, FP16 layers with FP32 I/O; input 1×3×320×416, outputs o1 1×2730×4, o2 1×2730×4, o3 1×2730×5. `trtexec --loadEngine`: GPU compute 2.00 ms mean (p99 2.01 ms), 500 inferences/s, with edge-vision streaming. Needed on the camera: `tensorrt-trtexec-prebuilt` and `tensorrt-plugins-prebuilt` (trtexec loads `libnvinfer_plugin.so.10`), both installed from the feed |
| Engine build (first start) | the `engine-warm` thread built the plan in-process in 398 s (21 MB, cached in `/var/cache/edge-vision/<model-hash>/`) while video streamed: 30 fps, 0 drops, 0 timeouts |
| Engine reload (restart) | plan reloaded from the cache in 0.9 s |
| CUDA graph | captured after the first inference; each inference launches one graph |
| Steady state, inference on | 30 fps, every frame inferred, 0 drops, 0 capture timeouts |
| Per-frame time | mean pre 2.7 ms (VIC letterbox + CPU planar write), infer 7.0 ms (TensorRT FP16 via CUDA graph, incl. sync), post 2.0 ms (YOLOX decode + NMS) |
| Latency | capture → detections 11.6 ms p50, 11.7 ms p99 |
| CPU, inference on | about 19 % busy in total; x264 dominates |
| Detections on video | drawn on a decoded frame, they land on the right objects: a round fan → "clock" 0.37, the carton → "book" 0.35 |
| Zero copy, inference on | 219 of 219 buffers queued into the encoder appsrc are the capture appsink's own `GstBuffer`s, 6 distinct (the Argus pool); the VIC preprocessor reads the `NvBufSurface` with no CPU copy |
| Encoder PTS | `x264enc` shifts output PTS by 1000 h; `GstEncoder` maps it back to the capture clock; a constant ~3.5 ms residual remains between detection pts and video pts (inside the web UI's 150 ms matching window) |

The `nvbuf-sys` struct mirror matches the r39.2 `nvbufsurface.h` (sizes
72/416/232, field order, `NVBUF_LAYOUT_PITCH = 0`,
`NVBUF_LAYOUT_BLOCK_LINEAR = 1`), checked by compiling the header.

Run on the board: inference on the Nano through `edge-vision` (the rows above).
RTSP also ran on the board: `rtsp://<cam>:8556/stream` played over RTSP/TCP in VLC and
ffmpeg (30 fps, 1440×1080), with and without inference.
Not run on the board yet: anything on the Orin NX (NVENC, X1) and the two-camera
setup (X4).

## M1 path

```
nvarguscamerasrc → NV12 NVMM → appsink
      → HwFrame (same gst::Buffer / NvBufSurface)
      → [pre → infer → post] (T9 CUDA preprocessor: not written)
      → encode thread (leaky one-deep hand-off)
      → NX:  appsrc → nvv4l2h26xenc → byte-stream appsink
      → Nano: appsrc → nvvidconv → I420 → x264enc → h264parse → appsink
      → RTSP (PC sink; device playback not yet run)
```

### Same-buffer NVENC hypothesis (X1, prove first on the NX)

If the encoder pushes **the same `gst::Buffer`** received from capture
into the encode `appsrc`, NVENC receives NVMM memory from Argus's own
allocator, as inside one `nvarguscamerasrc ! nvv4l2h264enc` pipeline. A
separate NVMM `GstAllocator` (gst-nvmm's `nvmmalloc`, LGPL) is needed
only when we allocate our own NVMM pool (the later direct-libargus path).

Check on the board: does `nvv4l2h264enc` accept a buffer from another
pipeline, are there `NvVideoRegisterMemHandle` errors, and does
`copied_frames == 0` hold? Rebase PTS to the encode pipeline's running
time. If it holds, M1 loses its largest dependency.

## Constraints from the shipping C++ product

These change traits or the process model, not just implementation.

### Argus failure is a process-level event

- A lost `nvargus-daemon` session never reconnects in-process; only a
  new process can open the camera again.
- One sensor's fault breaks Argus for every camera in that process.
  Default to **one process per camera**; they share the daemon anyway.
- The failure usually arrives as bare EOS; "Failed to create
  CaptureSession" / "Device is in use" go to the journal, not the bus.
- Tearing down a pipeline after the daemon went away can segfault
  inside the Argus client.
- Recover once per boot (restart the daemon, exit 75,
  `RestartForceExitStatus=75`); a second loss in the same boot is only
  logged, or a reboot loop follows.
- Never exit without stopping the pipeline (see the unclean-exit row
  above): no panic or `expect` after `start()` in any device binary.

### The capture pool belongs to Argus

`source.pool_size` is a sim concept. On the Nano the Argus pool is 6
buffers, and every `gst::Buffer` the Rust side holds is one Argus cannot
refill. The open budget (N11, E6):

| Holder                          | Frames                     |
| ------------------------------- | -------------------------- |
| appsink                         | 1                          |
| Rust channel                    | 1 (`queue_depth = 1` live) |
| processing                      | 1                          |
| encoder hand-off                | 1                          |
| appsrc                          | 1 (leaky downstream)       |
| **total**                       | **5, under the pool of 6** |

The CPU-encode `nvvidconv` output is a separate system-memory I420
surface, not another Argus buffer.

Holding more than the pool stalls Argus; it shows up as capture
timeouts or "frame took too long" journal errors. Keep live queues small.
One frame of channel is 33 ms at 30 fps; deeper queues spend the
Argus pool.

If the capture pool reuses a surface NVENC is still encoding, NVENC
fails with `NvVideoRegisterMemHandle failed`, then hangs or crashes. An
encoder must hold each `HwFrame` until NVENC releases its `GstBuffer`.

### Clean video, metadata boxes, inference off the video path

- Do not draw boxes into the capture surface by default. Boxes go out as
  metadata (JSONL today; MQTT/WebSocket later; ONVIF metadata track once
  RTSP exists). Make `Overlay` optional and off on Jetson.
- Video must not wait for inference. The reference product infers one
  frame in `interval + 1` (`interval=2` on device) and a tracker carries
  boxes between frames. Pre/infer/post still share the processing
  thread: split inference into a one-deep leaky side path joining
  detections by `seq`/`pts` before the NVENC stage.
- The first TensorRT plan build takes minutes after any image or model
  change; the camera streams without inference until the plan is ready.

### IMX296 on Argus

- White balance from the vendor NITO does nothing yet. Expect a green
  cast; pin the golden-set mAP gate to the tuned image, not the first
  boot.
- The first ~1 s of frames are dark while AE converges. Skip them for
  golden/mAP comparisons and for "first frame" health.
- Cam1 stops after 13–15 s under Argus in the reference product, cause
  open. The "30 FPS × 2 cameras" target is blocked on it (X4).
- On the Nano, at-start ranges gave black frames in another app. Check
  brightness on the board with the ranges applied after PLAYING, and log
  the sensor's reported limits.

### Building the engine on the image

The camera image does not install `trtexec`, and the service sandbox
will not let it run one. The product path is D5: build the plan
in-process through `trt-sys` and cache it under
`/var/cache/edge-vision` (systemd `CacheDirectory=edge-vision`).
`tools/build-engine.sh` is a development-image bring-up tool only. The
image must install the model at `model.path` (the Nano profile uses
`/opt/edge-vision/models/yolox-s.onnx`, camera-viber's yolox-s package).

### Encoder details

- The nvv4l2 encoders report "Failed to process frame" when a
  flush-start reaches them. A looping file source must restart at EOS,
  not flush-seek.
- Derive and pin `iframeinterval` from `gop_frames` when the GOP is long.

- gstreamer-rs 0.25.4: `Bus::timed_pop_filtered(ClockTime::ZERO, ..)` never pops. Any new bus poll must use `pop_filtered`.
- The encoder `appsrc` caps need `framerate`, or `nvvidconv` refuses them.

### Clocks and measurement

The reference product's numbers (3.05 ms inference, 3.9 ms capture to
encoded on NX) came from a CUDA graph plus pinned engine clocks through
devfreq, which also works on fused units. Make "clocks pinned" a
precondition of M1/M2 measurements, and list power mode, cooling and
ambient next to every number.

### Running on the camera Yocto image

- **boot-ok**: the A/B update verifier counts a boot as good only when
  `/run/camera-streamer/boot-ok` is written. Write it only after a real
  health check (a frame from an enabled camera), not after "up 60 s".
- **systemd watchdog**: `WatchdogSec=15` once the app calls `sd_notify`;
  the watchdog pet must not sit behind disk I/O.
- **Sandbox / camera-guard (BPF LSM)**: the service runs non-root under
  `ProtectSystem=strict`. camera-guard limits exec/write/connect. A new
  binary needs its own entries in `ebpf/camera-guard/guard.h` and grants
  for the `video` group (`/dev/nvhost-*`, the Argus socket).
- **Ports**: RTSP on 8555; 8554 belongs to camera-streamer.
- **Model licence**: YOLOX (Apache-2.0) is the device default.
  Ultralytics YOLOv8 weights are AGPL-3.0 and stay sim/test-only.

## Inference stages (TensorRT/CUDA/VIC)

The reference pieces already run in the C++ product
(`TensorRtEngine`, `NvmmFramePreprocessor`, `NvmmGpuTensorWriter`,
`CudaTensorKernels.cu`, `BuildProgressAdapter`). Port behaviour, not
layout.

### Decisions

| #   | Decision                                                                                                                                             | Why                                                                                        |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| D1  | Two `-sys` crates: `nvbuf-sys` (hand-written read-only `repr(C)` view; bindgen only when transform APIs land) and `trt-sys` (C++ shim through `cxx`) | NvBufSurface is C with fields to read; TensorRT is C++ only                                |
| D2  | CUDA calls live inside the C++ shim, not `cudarc`                                                                                                    | ~12 calls sharing lifetimes with TensorRT; revisit after M1                                |
| D3  | I/O tensors in mapped host memory (`cudaHostAlloc(..., cudaHostAllocMapped)`)                                                                        | integrated memory; CPU sees `&[f32]`, GPU sees a device pointer, `edge-postproc` unchanged |
| D4  | Default pre-processing: VIC scales, CPU writes the planar tensor; CUDA kernel behind `cuda-preprocess`                                               | reference default/fallback; kernel needs nvcc + EGL                                        |
| D5  | Build the plan in-process (builder + `nvonnxparser`, progress callback), cached, warmed on its own thread                                            | progress, build lock, no extra binary on the image                                         |
| D6  | Video never waits for the plan                                                                                                                       | `EngineState::Warming`                                                                     |
| D7  | One `IExecutionContext` and one CUDA stream per camera                                                                                               | TensorRT threading rules                                                                   |
| D8  | `unsafe` only in `-sys` crates                                                                                                                       | reviewability                                                                              |

### Crate layout (planned parts)

```text
crates/platform-jetson/
├── trt-sys/                # unsafe: cxx bridge + C++ shim (+ .cu behind a feature)
└── src/
    ├── vic.rs              # VicScaler into a pooled input-size surface
    ├── cuda.rs             # CudaStream, MappedTensor, GpuTimer
    ├── trt.rs              # TrtEngine (InferenceEngine)
    ├── warmup.rs           # engine-warm thread + process-wide build lock
    └── preprocess.rs       # JetsonPreprocessor: VIC + planar write
```

`trt` adds the `-sys` crates and needs the L4T sysroot; `cuda-preprocess`
needs nvcc; without `trt`, the crate builds on any PC.

### Open tasks

| Task    | What                                                                                                                                                      | Done when                                                                           |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **T0**  | **Done, not run on a device.** `tools/Dockerfile.jetson-sdk` (Ubuntu 24.04, Rust 1.98; takes a mounted sysroot or the Yocto SDK, see `docs/yocto.md`) and the CI compile-only check of `--features trt` (`TRT_SYS_SKIP_SHIM=1`); the image itself was not built or tried | compile-only `--features trt` passes in CI |
| **T4 test** | Unit test of `MappedTensor` with a fake allocator (`src/mapped.rs`, done)                                                                              | `cargo test` on the PC                                                              |
| **Device tests** | `#[ignore]` checks for T3–T10 on the Orin (`tests/device_trt.rs`, written, not yet run; `EDGE_TEST_ONNX`, `EDGE_TEST_CACHE`)                       | `cargo test -p platform-jetson --features trt -- --ignored` passes on the Orin      |
| **T9**  | CUDA preprocessor (optional optimisation): `NvBufSurfaceMapEglImage`, `cuGraphicsEGLRegisterImage`, kernel into the input binding; pre is 2.7 ms on the VIC path | output equals T8 within 1e-6                                                        |
| **T10** | GPU time from CUDA events (`GpuTimer`, `InferenceEngine::last_gpu_time`, `edge_infer_gpu_seconds_total`, `gpu=` in the live line; not yet run on the board) | GPU time reported from events on the board                                          |
| **T11** | Conformance on the board + Nsight/`tegrastats` zero-copy gate; FP16 vs ONNX Runtime numeric check (within 1e-2) not done; golden-set mAP not done        | M1 inference exit gate                                                              |
| **T12** | Install the systemd units and grants on the image (`tools/systemd/`): edge-vision runs from `/opt/edge-vision` via `tools/systemd/edge-vision-dev.service` as root; `CacheDirectory`, video group, camera-guard grants | image-owner work; the recipe builds the binary only                                 |

**Order:** T11 is the M1 inference exit gate, with the device `#[ignore]`
tests. T9 and T10's timing are optimisations. T0 and T12 belong to the
image and SDK owners.

### Reference: gst-nvmm-cpp

[gst-nvmm-cpp](https://github.com/PavelGuzenfeld/gst-nvmm-cpp) (read at
commit `060a8c0`, 2026-10-24) has NVMM-native GStreamer elements in C++14
without DeepStream: an NvBufSurface allocator, VIC convert/composite, and
TensorRT inference on NVMM. It is validated on Xavier NX (JP5.1) and Orin
NX (JP6.2, CUDA 12.5, TRT 10.3) only. Nothing covers JP7, CUDA 13 or TRT
10.16, and it has no Argus source. Where it and camera-viber disagree,
camera-viber wins: it runs on JP7.

**Licence: LGPL-2.1-or-later.** Reimplement the API call sequences below
from NVIDIA's headers and camera-viber. Never copy its code into this
MIT/Apache workspace: not `preprocess.cpp`, `gstnvmminfer.cpp`,
`trt_engine.cpp`, `nvbufsurface_mock.h`, the allocator and pool, or its
docs' wording.

**Adopt (patterns):**

| Task   | What the reference does (file:line under `gst/`)                                                                                                                                                                                                                                                               | Take for us                                                                                                                                                                                                            |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T8, T9 | It never maps the capture surface into CUDA. VIC converts NV12 into a private pitch-linear RGBA surface at network size (`nvmminfer/preprocess.cpp:12-52`). That surface is mapped once: `NvBufSurfaceMapEglImage`, `cudaGraphicsEGLRegisterImage`. camera-viber's `NvmmGpuTensorWriter` does the same on JP7. | VIC into our own scratch surface, mapped once at configure; never the capture buffer per frame                                                                                                                         |
| T9     | It branches on the mapped EGL frame type: pitch (`pPitch[0]`) or array (`cudaMemcpy2DFromArrayAsync`) (`preprocess.cpp:98-110`).                                                                                                                                                                               | Handle both frame types; the array case is unverified on JP7                                                                                                                                                           |
| T4, T5 | One CUDA stream per element and one `cudaStreamSynchronize` per frame, before the buffer returns to the pool (`nvmminfer/gstnvmminfer.cpp:145, 406`). No `SyncForDevice` on the GPU path.                                                                                                                      | One stream per camera; one sync per frame before the buffer guard drops; `SyncForDevice` only after a CPU write (camera-viber's letterbox fill)                                                                        |
| T8     | Letterbox in one VIC call with `CROP_DST` and a destination rectangle; the padding is filled once and VIC never overwrites it (`preprocess.cpp:73-88`).                                                                                                                                                        | The same. Set the VIC session params on **every** call: they are per thread (camera-viber `NvmmFramePreprocessor`)                                                                                                     |
| T8, T9 | Float/planar conversion with NPP `_Ctx` calls on the stream (`preprocess.cpp:119-141`).                                                                                                                                                                                                                        | Our own small CUDA kernel (camera-viber's `CudaTensorKernels` pattern): NPP would be one more `-sys` crate, and its CUDA 13 status is unknown                                                                          |
| T5     | It rejects an output whose shape does not match, so the decoder cannot over-read (`gstnvmminfer.cpp:167-196`).                                                                                                                                                                                                 | Bounds-check the output shape against the decoder. FP16 plans: kFP16 builder flag only, I/O stays FP32                                                                                                                 |
| T11    | A header-only NvBufSurface mock lets the same sources build on x86 (`common/nvbufsurface_mock.h`). Its pitch is tight, so it missed the real-pitch bug.                                                                                                                                                        | A Rust trait fake of a narrow HAL (create, map, transform, sync) with a padded pitch (1472 for 1440, as measured) and a block-linear flag that scrambles bytes. One contract suite runs against the fake and the board |

**Avoid:**

- Its TensorRT lifecycle (T6, T7, T10): engines prebuilt by `trtexec`,
  no cache, no warm-up, no build lock, and a host copy plus sync every
  frame (about 23 ms per frame at 1080p on Orin, its B5 design doc).
  Follow camera-viber's `docs/INFERENCE-ENGINE.md` instead.
- Its `dla-core` property: a no-op for prebuilt engines, and DLA is no
  option with TRT 10.16 on JP7.

**Pitfalls it documents (check each on JP7):**

- `numFilled` is 0 after `NvBufSurfaceCreate`; Map and Transform then
  reject index 0. Set it after every create (T8).
- `NvBufSurfaceCopy` does not de-tile block-linear; use
  `NvBufSurfTransform`.
- `NvBufSurfTransform` rejects some CUDA-memory destinations; use a
  surface-array (`NVBUF_MEM_DEFAULT`) RGBA destination.
- A resolution change needs the pool drained before it is recreated.
- `tee` or `make_writable` on an NVMM buffer whose memory is no-share
  copies it or fails (E5).
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

| Layer               | Runs on     | Covers                                                                                                       |
| ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------ |
| Unit                | PC/CI       | plane maths, letterbox, shape checks, warming engine, `MappedTensor` with a fake allocator                   |
| Golden              | PC → device | PC dumps input tensor and ONNX Runtime outputs for a fixed frame; device compares T5/T8/T9 within tolerances |
| Device `#[ignore]`  | Orin        | T3–T10 checks via `cargo test -p platform-jetson --features trt -- --ignored`                                |
| Conformance + trace | Orin        | T11                                                                                                          |
| CI                  | PC          | default features + compile-only `--features trt` in the SDK container                                        |

Device binaries are cross-built by the camera-viber recipe; the
`gst-capture` example is built with a temporary override
(`CARGO_SRC_DIR = "crates/platform-jetson"`, `--features gst --example
gst-capture`). Stop MainApp (or camera-streamer) before a run and start
it again afterwards; never restart `nvargus-daemon` by hand.

## Open questions

1. **`cudarc` instead of CUDA in the shim (D2).** Check support for CUDA
   13 on aarch64. If it is solid and the shim's CUDA part grows, move
   streams/events/memory there.
2. **DLA** (Orin NX has two). Out of scope for M1; the `model.device =
"dlaN"` hook exists.
3. **INT8** needs a calibration set and its own accuracy gate. M1 ships
   FP16.
4. **Async inference.** M1 synchronises inside `infer`. Overlapping
   pre-processing of N+1 with inference of N needs events and a two-slot
   input; only do it if T10's numbers justify it.

## Open findings

| ID               | Severity | Where                                                                   | Finding                                                                                                                                                                                                                                                                                                                              | Next step                                                                                                                                    |
| ---------------- | -------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| E1b              | device   | `gst_encoder.rs` | Drop counting was dead until `dropped_frames` was overridden (the leaky `appsrc` dropped 0 frames at 30 fps on the Nano) | Verify `edge_encode_dropped_total` counts under load (slow encoder, NX) |
| N9b              | major    | `apps/edge-vision/src/recovery.rs`, `tools/systemd/edge-vision.service` | The unit runs as `User=edge-vision`, but recovery calls `systemctl restart nvargus-daemon`, which a non-root user cannot do                                                                                                                                                                                                          | Proposal polkit rule added at `tools/systemd/50-edge-vision-nvargus.rules`; install and verify recovery on the board (root helper is the fallback)                              |
| N11              | major    | capture/encoder budget                                                  | ≈9 buffers held on the M1 path against a measured pool of 6 (table above)                                                                                                                                                                                                                                                            | Static worst case is queue_depth + 6 = 7 surfaces at queue_depth 1 (was + 7 until `encode` dropped its frame ref before the push) against a measured pool of 6; the build warns instead of refusing (60 s on the Nano: 0 capture timeouts) | The capture channel already evicts before it inserts (no gain left there; the pulled frame coexists with a full queue, only an eviction before the pull would change that). Cut holders: done: `GstEncoder::encode` drops the `HwFrame` before `push_buffer` (-1); next: a direct encode instead of the hand-off slot 
| NB1              | minor    | `nvbuf-sys` `surface_from_bytes`                                        | Reads batch 0 without checking `num_filled >= 1`                                                                                                                                                                                                                                                                                     | Fixed: `num_filled == 0` is rejected                                                                                                          |
| X1               | device   | NX encoder                                                              | Same-`gst::Buffer` NVENC (see above)                                                                                                                                                                                                                                                                                                 | First NX session                                                                                                                             |
| RTSP             | device   | `crates/sinks`                                                          | No RTSP sink: video cannot leave the device                                                                                                                                                                                                                                                                                          | PC sink added (`edge-sinks --features rtsp`, `stream.rtsp_port`); a real `rtspsrc` client test passes on the PC; verify playback/port 8555 on the board |
| X4               | blocked  | two cameras                                                             | cam1 stops after 13–15 s under Argus (hardware/driver, cause open)                                                                                                                                                                                                                                                                   | —                                                                                                                                            |
| Nano measurement | device   | Nano                                                                    | `tegrastats`, process CPU, `edge_encode_dropped_total` and x264 `threads=3` with the H.264 stream                                                                                                                                                                                                                                    | Inference runs on the board now (total CPU ≈19 %, 0 drops measured); take `tegrastats` and per-process CPU with inference on          |
| P1               | minor    | `gst_encoder.rs` (x264 PTS)                                             | Constant ~3.5 ms residual between detection pts and video pts after the 1000 h shift is mapped back (inside the web UI's 150 ms matching window)                                                                                                                                                                                    | Find the residual's source (x264 latency or the clock mapping); no action while it stays inside 150 ms |
| E8               | minor    | `launch.rs`                                                             | The CPU H.265 launch string is an unreachable future hook                                                                                                                                                                                                                                                                            | Keep documented or remove                                                                                                                    |
