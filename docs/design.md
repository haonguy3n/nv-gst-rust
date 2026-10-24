# Edge Vision Pipeline — Design

**Status:** M0/M1 groundwork. Everything CPU-side is implemented and
tested on x86; Jetson device stages (NVMM surface access, CUDA/VIC
pre-processing, TensorRT, NVENC, RTSP) are not implemented yet. The
tables below are deliberately explicit about which side of that line
each claim sits on.

The project builds one Rust binary that captures camera frames, runs a
neural network on them and streams the result, keeping pixels in
hardware memory from sensor to encoder. Jetson Orin NX/Nano with
JetPack 7 is the first target; NXP i.MX and Ambarella are intended to
be new `platform-*` crates without changes to the pipeline core.

## Implementation status

| Area | State | Code path / verification |
| --- | --- | --- |
| Traits, `HwFrame`, `Tensor`, plane descriptors, errors | Done | `crates/core` (`#![forbid(unsafe_code)]`); unit tests |
| Config: typed, `deny_unknown_fields`, validated against `PlatformCaps` | Done | `crates/core/src/config.rs`; config and builder tests |
| Runner: two threads, bounded drop-oldest queue, `FramePoll`, stop handle, idle limit, partial metrics on error | Done | `crates/pipeline`; integration and regression tests |
| Live metrics: counters, queue depth, p50/p95/p99 frame age, Prometheus text | Done | `crates/pipeline/src/metrics.rs`; unit + live-run tests |
| Sim backend: synthetic/file source, pooled DMA-BUF-like buffers, real NV12 letterbox CPU pre-processor | Done | `crates/platform-sim`; unit + conformance tests |
| ONNX Runtime engine (sim, `--features onnx`) | Done | `crates/platform-sim/src/onnx.rs`; fixture tests |
| YOLOv8/v11 and YOLOX decoders, NMS, zero-allocation steady state | Done | `crates/postproc`; unit tests, allocation gate, committed end-to-end fixture test |
| Sinks: JSONL metadata, file video, fan-out, counting/null | Done | `crates/sinks`; unit tests |
| Conformance suite every backend must pass | Done | `crates/conformance`; sim passes, broken stubs are caught |
| Jetson: NVENC/NVJPG sysfs probe, launch strings, plan cache | Done (PC-testable) | `crates/platform-jetson`; unit tests with fake sysfs |
| Jetson: GStreamer capture (`appsink`) | Partial | `platform-jetson --features gst`: dmabuf branch and a PC-only copy fallback; NVMM buffers are refused, never copied |
| Jetson: Argus EOS is a fault, not a clean end | Done (PC-tested) | `gst_source.rs` returns `Err` for an `argus` source; once-per-boot recovery/exit 75 still M1 |
| Jetson: NVMM/NvBufSurface access (`gst_buffer_map` → `NvBufSurface*`) | Not implemented | Needs `nvbuf-sys` and the L4T sysroot; see `docs/jetson-bringup.md` |
| Jetson: VIC/CUDA pre-process, TensorRT engine, overlay, NVENC encode, RTSP | Not implemented | Device bring-up; launch strings are the contract |
| Device zero-copy proof (`copied_frames == 0`, Nsight, EMC) | Not verified | M1 exit gate on the flashed board |
| Multi-camera orchestration | Not implemented | One process per camera is the default for Argus fault isolation |

`RTSP` is intentionally absent from the config until the sink exists:
the old `stream.rtsp_port` field never had a reader and was removed
rather than left as a false promise. Port 8555 is reserved for it.

## Goals and non-goals

**Goals**

1. **Zero-copy path on Jetson**: Argus → NvBufSurface (NVMM) → CUDA/VIC
   pre-process → TensorRT → optional overlay → NVENC → RTSP, with no
   CPU pixel copy in the hot loop.
2. **Platform-agnostic core**: the pipeline knows only traits
   (`FrameSource`, `Preprocessor`, `InferenceEngine`, `Encoder`, `Sink`)
   and opaque `HwFrame`/`Tensor` handles. Vendor code lives in
   `platform-*` crates behind Cargo features.
3. **Safe Rust by default**: `unsafe` is confined to `*-sys` crates and
   a thin safe wrapper; RAII for every hardware buffer.
4. **Config-driven**: one TOML selects platform, camera, model,
   encoder and outputs; unknown keys and impossible combinations are
   startup errors.
5. **Observable**: per-stage time, queue depth, drop/timeout counters,
   end-to-end percentiles, exported as logs and Prometheus text.

**Non-goals (v1)**

- Training or model-conversion tooling beyond running `trtexec` /
  loading ONNX.
- A GUI; a preview page comes from the RTSP/WebRTC output later.
- x86 dGPU as a production target (it is a dev/CI backend only).

## Performance targets

Target hardware: Orin NX 16 GB, 1080p30, YOLO-class detector, FP16 on
M1 (INT8 after a calibration set and its own accuracy gate).

| Metric | Target | Qualification |
| --- | --- | --- |
| Throughput | 30 FPS per camera, 2 cameras | One process per camera. Cam1 currently stops after 13–15 s under Argus in the reference product; the 2-camera target is blocked on that hardware cause |
| Glass-to-network latency | < 100 ms p95 | The in-process part (source hand-off → sink accept) is now measured as p50/p95/p99; network/RTP is added with the RTSP sink. Live configs use `queue_depth = 2` |
| CPU load | < 25 % of one core per stream | Strict (no-CPU-copy) profiles only. The relaxed CPU-encode profile is exempt: ≈1.1 NX cores for H.264/1080p30 in the reference product, more on a Nano's slower cores |
| Steady-state allocations | 0 in the hot loop | Decoders are gated by a counting allocator. Capture/engine/encode allocation behaviour is backend-specific and must be measured on the device; the sim allocates one handle per frame by design |

Measurements are meaningful only with SoC clocks pinned (devfreq
floors) and the power mode, cooling and ambient conditions recorded.
Unpinned DVFS numbers measure the governor, not the code.

## Architecture

```
source ──HwFrame──> [capture thread] ──bounded drop-oldest channel──>
         [processing thread] pre → infer → post → overlay? → encode? → sink
```

The runner uses two threads. The capture thread pulls from the
`FrameSource` and never blocks on a slow consumer: when the bounded
queue is full the oldest `HwFrame` is evicted (returning its buffer to
the pool) and counted. The processing thread runs the remaining stages
in order. Per-stage threads are a latency/parallelism decision to be
made from device measurements, not baked into the traits.

`Platform::engine` may return `EngineState::Warming` instead of a ready
engine. The runner polls it once per frame; while the TensorRT plan is
being built, video and metadata flow with empty `Detections`. A
several-minute engine build must never mean several minutes without
video.

## Zero-copy contract

After the sensor writes a frame, no CPU instruction and no
`cudaMemcpy` may copy its pixels in steady state. A frame lives in one
hardware allocation from capture to encoder. Resizing to the model
input is a transform (it makes new pixels) done in one fused pass into
the TensorRT input binding.

### How the types enforce it

- **No CPU pixel access in release.** `HwBuffer` has no `as_slice()`.
  `map_cpu()` exists only behind the `cpu-debug` feature; release
  builds do not compile code that needs CPU pixels.
- **Memory domain and layout are declared, then checked.** Each source
  states its `FrameMemory` and `Layout`; the builder compares them with
  `PlatformCaps` and hands the declared layout to the preprocessor. It
  never inserts a converter silently.
- **Plane geometry is authoritative.** `FrameMeta::planes` carries
  byte offsets, pitches, visible row bytes and rows for up to three
  planes inside one contiguous dmabuf. A padded or block-linear surface
  reports its real pitch; consumers never infer stride from width.
- **Same fd end to end.** `FrameMeta` is immutable after the source
  hands it out (private field behind accessors). The runner checks the
  meta fd against the buffer's fd before encode and returns a typed
  error in release builds.
- **Write into, never return.** The preprocessor writes into the
  engine's input binding. The encoder hands packets out as shared
  `Arc<[u8]>` payloads (hardware backends will keep the encoder buffer
  alive instead), so fan-out clones a refcount, not the bytes.
- **Frames move by value to the encoder.** The encoder owns the
  `HwFrame` until the hardware releases it; dropping it returns the
  buffer to the pool.

### Per-frame buffer budget (device, illustrative)

| Buffer | Memory | Written by | Read in place by |
| --- | --- | --- | --- |
| Capture surface NV12 1440×1080 (IMX296) | NVMM dmabuf, block-linear from `nvarguscamerasrc`; pool owned by Argus | ISP via Argus | CUDA/VIC pre-process, optional overlay, NVENC |
| Input tensor, e.g. F32 3×416×416 (mapped host; FP16/INT8 select layer execution, not the I/O dtype) | host-visible mapped memory = the TensorRT input binding | fused CUDA/VIC pre-process | TensorRT |
| Output tensors | pinned mapped host memory (`cudaHostAllocMapped`) | TensorRT | CPU post-process |
| Bitstream | encoder output buffer (`GstBuffer` from `nvv4l2h26xenc` in M1) | NVENC | RTSP sink by reference; returned on drop |

For a YOLOv8 640×640 80-class head, `[1, 84, 8400]` is 2.82 MB in F32
(≈1.4 MB in F16); the actual model shape is used, not a guessed “~1 MB”.
Two small bitstream copies are budgeted, not denied: RTP payloading
repacketizes the encoded bytes, and the kernel copies on socket send.
Neither touches pixels. The relaxed CPU-encode profile adds one
deliberate VIC copy out of NVMM into I420, listed in the startup
budget.

### M1 capture/encode shape

M1 uses the GStreamer hybrid path proven in the owner's camera
products:

- capture: `nvarguscamerasrc` → NV12 NVMM → `appsink` (tiny, drop
  policy in Rust);
- encode: `appsrc` → `nvv4l2h264enc`/`nvv4l2h265enc` → byte-stream
  `appsink`.

The hypothesis to prove first on the board is that pushing **the same
`gst::Buffer`** received from capture into the encode `appsrc` makes
NVENC consume Argus's own NVMM memory with no extra allocator. A
separate NVMM `GstAllocator` is required only once we allocate our own
NVMM pool (the later direct-libargus path). Direct libargus capture
replaces the capture string behind the same `FrameSource` trait, not in
M1.

In-place analytics must sit **before** any tee/branch: after a tee,
branches hold shallow copies of a surface the capture pool may reuse,
and NVENC can end up encoding a recycled surface (crash/hang).

## HAL design

The core defines five stage traits and the opaque buffer/tensor handles;
each platform implements a `Platform` factory. Design rules:

- **Buffers are handles, not bytes.** A frame owns a platform buffer
  (NvBufSurface on Jetson, DMA-BUF on NXP, IAV on Ambarella).
- **Memory domains are explicit.** `FrameMemory::{Nvmm, DmaBuf,
  Other(&'static str)}`. A new backend can add a domain tag without a
  core release; equality is exact.
- **Capabilities, not `cfg`, in the core.** `PlatformCaps` declares
  frame memory, formats, resolution, codecs, precisions, device names,
  hardware-encode presence and whether a CPU-encode path exists. The
  builder validates the config against it.
- **Bounded pool + backpressure.** Sources allocate a fixed pool;
  stages are joined by a bounded channel; a slow sink drops and counts
  the oldest frame instead of stalling the sensor.
- **Boxed stages at startup, one call each per frame.** Dynamic
  dispatch once per stage per frame is cheap and keeps a new backend
  from rippling through the core.

Key types:

```rust
enum FrameMemory { Nvmm, DmaBuf, Other(&'static str) }
enum Layout { PitchLinear, BlockLinear }
struct PlaneDesc { offset: usize, pitch: usize, row_bytes: usize, rows: u32 }
struct FrameMeta { seq, pts, width, height, format, layout, planes, camera_id, dmabuf_fd }
struct HwFrame { /* private meta */ pub buf: Arc<dyn HwBuffer>, /* received_at */ }

enum FramePoll {
    Frame(HwFrame),
    Timeout(TimeoutReason), // NoFrame | PoolExhausted
    EndOfStream,
}

enum EngineState { Ready(Box<dyn InferenceEngine>), Warming(Box<dyn EngineWarmup>) }
struct GpuStream { /* opaque Arc<dyn Any> the backend downcasts */ }
```

`FramePoll` keeps “no frame yet” and “end of stream” separate. Only
pool starvation is counted as a capture timeout; ordinary sensor
pacing is not. An `argus` source treats EOS as a capture fault (the
sensor or `nvargus-daemon` may be gone) and returns an error, so a dead
camera cannot end the run with exit 0. Once-per-boot daemon recovery
and exit code 75 are still M1 device work; see
`docs/jetson-bringup.md`.

## Configuration

One TOML, validated before any hardware is opened:

```toml
platform = "sim"          # "sim"; "jetson" with --features jetson
zero_copy = "strict"      # strict refuses CPU-encode profiles

[source]
kind = "synthetic"        # sim: synthetic|file; jetson: argus|file
width = 640
height = 480
fps = 30
pool_size = 8             # sim pool; must be >= queue_depth + 3
frames = 30               # stop after N captured frames (optional)

[model]
input_width = 320
input_height = 320
num_classes = 80
decoder = "sim"           # sim|yolo|yolox
precision = "fp32"        # layer execution; validated against caps, not the I/O dtype
# device = "gpu"
# path = "models/yolox-tiny.onnx"   # needs --features onnx in sim

[encoder]
enabled = false
hardware = true
codec = "H264"
bitrate_kbps = 8000
gop_frames = 30

[stream]
metadata_only = true      # video requires an enabled encoder
# meta_file = "out/detections.jsonl"

[pipeline]
queue_depth = 4
max_idle_ms = 5000        # 0 disables the dead-source limit
```

Validation highlights: unknown keys are errors; NV12 dimensions are
even; `pool_size >= queue_depth + 3`; strict mode refuses
`hardware = false`; `metadata_only = false` requires an enabled
encoder and vice versa; gain/exposure/bitrate/GOP/num_classes/IoU/
`max_nms`/`top_k` are bounded. The builder additionally checks the
hardware capability (codec, precision, device name, CPU-encode
availability) and the engine's declared input shape/dtype against the
config.

## Runner behaviour and metrics

- **Shutdown**: `stop_handle()` is a shared `AtomicBool`; SIGINT/SIGTERM
  set it. A second signal exits immediately (130). A stop requested
  before or between runs is honoured; `clear_stop()` is the explicit
  reset. The capture thread stops on the external flag or the per-run
  teardown flag, and the receiver is dropped before `join`, so an
  unbounded source cannot deadlock shutdown.
- **Errors**: capture and processing always restore the source and run
  `stop()`; the first processing error wins; partial metrics are
  returned with the error; `Sink::finish()` runs on every exit path.
- **Idle limit**: no frame for `max_idle_ms` fails the run. The message
  blames process-side stalls when polls starved on the pool, and the
  source otherwise.
- **`--frames N`** limits frames *captured*. Under load the drop-oldest
  queue may process fewer; both counts are reported.
- **Latency**: measured from the source hand-off to sink accept, so it
  includes queueing and every stage. Reported as p50/p95/p99 and max
  from a fixed-size ring (no per-frame allocation, bounded memory).
- **Queue depth**: current and peak depth are read directly from the
  channel, so they stay visible while processing is stalled.
- **Prometheus**: `--prometheus` prints a text exposition at exit and a
  live summary every 5 s to stderr. There is no HTTP endpoint yet;
  the sink/HTTP layer owns that.
- Counters reset at every `run()`; a scraper sees a counter reset.

## Building

- **Host / CI:** `cargo build --workspace`; `--features onnx` for the
  sim ONNX engine; `--features jetson` for the GStreamer runtime.
- **Yocto (device image):** the camera image's `meta-vc-camera` layer
  has an `nv-gst-rust` recipe that builds `apps/edge-vision` with
  `--features jetson`, fetches crates offline from `Cargo.lock`, and
  installs the binary plus `/etc/edge-vision/*.toml`. It deliberately
  does not build `onnx`. See [yocto.md](yocto.md).
- **TensorRT engines:** the product builds plans in-process through
  `trt-sys` (D5) and caches them under `/var/cache/edge-vision`.
  `tools/build-engine.sh` is a bring-up tool for a development image:
  the release image ships no `trtexec`. It limits the builder to a
  256 MiB workspace, takes a build lock, checks `MemAvailable`, and
  refuses to write the repo directory (`EDGE_VISION_CACHE` selects the
  cache, default `/var/cache/edge-vision`).

## Platforms and porting

- `platform-sim` is the reference second backend: it runs anywhere, uses
  real pooled fds and a real NV12 pre-processor, and is what conformance
  and CI exercise.
- `platform-jetson` currently holds the PC-testable parts (sysfs media
  probe, pinned launch strings, GStreamer capture with a dmabuf branch,
  TensorRT plan cache). Device stages return `Unsupported`.
- Adding a platform means a new `platform-<vendor>` crate implementing
  `Platform` and the stage traits; `core`, `pipeline`, `postproc` and
  `sinks` stay untouched. Expose the vendor buffer's dmabuf fd and plane
  geometry, return honest caps, and pass the conformance suite.

The conformance suite checks honest caps, frame order/monotonic
timestamps, metadata matching the config (including layout and plane
geometry and the same-fd contract), pool exhaustion returning `Timeout`
and recovering, the preprocessor writing finite non-trivial values into
the engine binding, repeatable inference, the encoder option matching
caps, and clean shutdown with every pooled buffer returned. Golden
pixel values and detection mAP are backend/model gates, not part of the
trait suite.

## Testing and CI

- Unit tests across `core`, `config`, `postproc`, `sinks`, `metrics`,
  `probe`, `launch`, `plan_cache`.
- Pipeline integration tests on the sim: ordering, drop accounting,
  slow sink, stage error without deadlock, stop handle, idle limit,
  run-twice, live metrics, warming engine, encoder refusal.
- A counting-allocator gate on steady-state decoder work.
- A committed end-to-end detector test (`crates/platform-sim` fixture →
  ONNX Runtime → YOLO decoder → NMS → sink) under
  `--features onnx`.
- Feature-gated tests for `platform-jetson --features gst` and
  `platform-sim --features onnx`.
- CI: fmt, clippy `-D warnings` (default, onnx, gst, app `jetson,onnx`),
  workspace tests, ONNX tests, GStreamer tests, feature powerset,
  MSRV 1.92, `cargo deny`, and an aarch64 cross build. The GStreamer
  aarch64 cross-check needs an arm64 sysroot and is left to the device
  sysroot work.

Not tested anywhere yet: NVMM/NvBufSurface access, NVENC dmabuf import,
TensorRT inference, camera/Argus behaviour, RTSP output, and zero-copy
on hardware. Those are device gates below.

## Milestones and exit gates

**M0 (done, x86):** workspace, HAL, config, sim backend, ONNX/YOLO,
sinks, conformance, CI, live metrics, end-to-end fixture test.

**M1 (Jetson hybrid, in progress):**

1. Flash the Orin NX with JetPack 7 / L4T r39.2.1 and record exact
   versions in `config/orin-nx-yolo.toml`.
2. Open the NVMM surface via `gst_buffer_map` → `NvBufSurface*`; read
   fd, layout and pitch from the surface; `copied_frames == 0` over
   ~300 frames.
3. Prove the same-`gst::Buffer` NVENC path: no
   `NvVideoRegisterMemHandle` errors, Nsight shows no steady-state
   `DtoD`/`HtoD` copies, `tegrastats` EMC load matches one write plus a
   few reads per frame.
4. Argus recovery: EOS already fails the run; add once-per-boot daemon
   restart, exit code 75, and the skip of `set_state(Null)` after a
   daemon restart.
5. In-flight surface budget under the Argus pool size; small live
   queues.
6. Clean video with metadata boxes; overlay optional/off; video path
   independent of inference.
7. Clocks pinned for every measurement.
8. Camera-image integration: boot-ok marker after a real first frame,
   systemd watchdog, sandbox/camera-guard grants, bounded disk writer,
   port 8555.
9. RTSP sink; then glass-to-network p95 replaces the in-process number.

**M2+:** INT8 calibration gate, DLA on NX, per-stage threads if
measurements justify them, multi-camera, NXP/Ambarella ports.

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| libargus and TensorRT are C++-only | slow start, unsafe surface grows | thin C shims via `cxx`, kept small; `unsafe` confined to `-sys` crates |
| Hidden CPU copies (map/unmap, wrong memory type) | CPU load and latency balloon | plane descriptors, `copied_frames` counter, refusing NVMM copies, Nsight/EMC gates |
| Argus failure is process-level | hung or crash-looping camera | one process per camera, EOS ambiguity, once-per-boot recovery, exit 75 |
| Argus pool starvation | captured frames stall the sensor | in-flight surface budget, tiny appsink queue, `queue_depth = 2` live |
| JetPack/TensorRT upgrades break ABI | rebuilds, invalid engines | pin JetPack, build plans at boot, plan cache keyed by model/TRT/SM/flags |
| Abstraction leaks Jetson assumptions | NXP port forces core changes | sim is a full second backend; conformance suite; plane/domain tags are open |
| Ambarella SDK under NDA | cannot publish backend | private crate depending on the public core |
