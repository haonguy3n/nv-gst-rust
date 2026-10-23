# Round 3 review: the Orin NX lead's read

**Date:** 2026-10-23
**Reviewer:** the lead of `orin-nx-camera-viber` (camera-streamer, the shipping C++ product on the same Orin NX, IMX296 sensors, JetPack 7 and Yocto image)
**Scope:** `design.md` and the Jetson path (`platform-jetson`, `pipeline`, `sinks`, `config/orin-nx-yolo.toml`), checked against what the C++ product learned on the hardware.

Rounds 1 and 2 already cover the generic Rust issues. This round does not repeat them. It adds what only the device history can tell you.

**This file changes no product code.**

**State of the tree.** The working tree was being edited while this review ran: `crates/core` and `platform-sim` changed at 15:22, and a new `Precision` type was half-added. So `cargo clippy` and `cargo test --workspace` did not compile at 15:23. That is an in-progress edit, not a finding. `cargo test -p platform-jetson --features gst` passed: 10 unit tests and 4 integration tests. Re-run the full gate once that edit lands.

## Verdict

The scaffold is sound, and the hybrid GStreamer decision (revision 1) is the right one. But the M1 plan still assumes a cooperative Argus. Argus is not cooperative. The five things below cost the C++ product weeks of device time; each has a dated incident in `orin-nx-camera-viber/docs/EMBEDDED.md` or `docs/FRAME-PATH.md`. Design them in now, because they change the traits and the process model, not just the implementation.

---

## L1. Argus failure is a process-level event, not a pipeline error (High)

What the device taught us (`orin-nx-camera-viber/docs/EMBEDDED.md`, "Capture recovery"):

- **The Argus client never reconnects.** Once `nvargus-daemon` restarts or loses its session, no new pipeline in the same process can open a camera. Only a new process helps.
- **One sensor's fault breaks Argus for every camera in the process.** Two cameras in one process share that fate.
- **The failure is only an EOS.** "Failed to create CaptureSession" and `Error 0x0000000d: Device 0 (of 1) is in use` reach the journal, never the bus. The pipeline just ends with an EOS, so the pipeline cannot tell a dead sensor from a lost daemon.
- **Tearing a pipeline down after the daemon went away segfaults inside the Argus client** (r20). `GstLaunchSource::drop` calls `set_state(Null)` unconditionally (`crates/platform-jetson/src/gst_source.rs`, `impl Drop`), so a daemon restart under a running camera crashes the Rust process in C++ code.
- **After a crash, the daemon keeps the dead process's session** and refuses the camera to the next process. A plain `Restart=on-failure` loops forever.

What to change:

1. **The source's EOS must be ambiguous in the type.** `FramePoll::EndOfStream` from an argus source should carry "may be a lost capture service", like the C++ `CaptureStage_t::bEosMayBeServiceLoss`.
   - Before tearing the pipeline down, ask systemd whether `nvargus-daemon.service` restarted since the pipeline started (`ActiveEnterTimestampMonotonic`).
   - If it did, skip `set_state(Null)` (leak the pipeline) and exit.
2. **The process owns recovery, once per boot.** On "a camera that worked can no longer start", or "a crash restart cannot open the camera":
   - record it,
   - queue `systemctl --no-block restart nvargus-daemon`,
   - exit with code 75,
   - and let the unit use `Restart=on-failure` with `RestartForceExitStatus=75`.

   A second recovery in the same boot is only logged; otherwise you get a reboot loop. That policy belongs in the app (`apps/edge-vision`), not in `platform-jetson`. Port `CaptureRecoveryPolicy` and its self-check (`embedded/Apps/CameraStreamer/test/CaptureRecoveryPolicyCheck.cpp` in the reference) as a pure function with tests.
3. **Treat "no first frame within N seconds" as a capture fault**, not as `Timeout` forever. This is round 2's R2-2, but the cause is specific: Argus gives you nothing else.
4. **Decide one process per camera, or one process for both, on purpose.** With both cameras in one process, cam1's fault takes cam0 down. With one process each, they share the daemon anyway. Record the choice in `design.md`.

## L2. Who owns the capture pool, and how many frames you may hold (High)

The `pool_size >= queue_depth + 3` rule (`crates/core/src/config.rs`) is right for the sim, which owns its pool. **On the Jetson, the pool belongs to `nvarguscamerasrc`, and `pool_size` does not size it.**

Every `gst::Buffer` the Rust side holds is a surface Argus cannot refill. Count the worst case on the M1 path:

| Holder | Frames |
|---|---|
| appsink | 2 |
| Rust channel | 4 |
| processing | 1 |
| appsrc | 4 |
| encoder in flight | 1+ |

That is about 12 surfaces. Check the `nvarguscamerasrc` buffer count in the JetPack 7 source (`gstnvarguscamerasrc.cpp`); it is a small fixed number. Holding more than the pool stalls Argus. You then see it as capture timeouts, or as the "frame took too long" errors Argus prints to the journal.

There is a second half to this, and it was our worst NVENC bug (`docs/FRAME-PATH.md`, 2026-10-16):

- The capture pool reused a surface NVENC was still encoding.
- NVENC failed with `NvVideoRegisterMemHandle failed`, then hung or crashed.

In the C++ product, an in-place element behind a tee caused it. In Rust, the same bug comes back if any stage drops its `HwFrame` before NVENC is done with the surface.

What to change:

- Make `pool_size` a sim-only setting, and give Jetson a validated **in-flight budget** instead: `appsink + channel + 1 + appsrc + encoder < argus_pool`. Print it in `buffer_budget()`, which today also says "pitch-linear" while the source declares `BlockLinear` (`crates/pipeline/src/lib.rs`, `buffer_budget`).
- Use `queue_depth = 1` (or 2) for live sources. Four frames is 133 ms at 30 fps, already above the < 100 ms p95 target before any work happens.
- The encoder's `HwFrame` must live until the encode pipeline releases its `GstBuffer`. If you push the very same `gst::Buffer` into `appsrc`, GStreamer's refcount does this for you; see L5. A copy or a re-wrap does not.

## L3. Don't draw boxes into the capture surface; don't gate video on inference (High, design)

The design's overlay "draws boxes in place on the capture surface", and `process_one` runs `pre → infer → post → overlay → encode` in series on one thread. The shipping product made the opposite choice, for reasons that apply here:

- **Video stays clean.** Boxes go out as metadata (ONVIF on the RTSP metadata track; here JSONL/MQTT), and the client draws them.
  - In-place drawing writes into a surface other readers may still hold (the tee lesson above).
  - It burns the boxes into recordings permanently.
  - The C++ repo removed its box overlay on `feat/drop-boxoverlay`.
- **Video must not wait for inference.** The C++ product infers one frame in `interval + 1` (the device config uses `interval=2`), and a tracker carries the boxes in between.
- **The first TensorRT engine build takes minutes**, after every image or model change. The camera streams without inference until the engine is ready (an `engine-warm` thread; the engine is cached on `/data`). In this repo, `Pipeline::build` fails when `platform.engine()` fails, so a camera that is still building its engine has **no video at all**.

What to change:

- Make `Overlay` optional, default off on Jetson.
- Split the runner: the video path (capture → encode → sink) runs at the sensor rate.
- Inference reads the same `Arc<dyn HwBuffer>` on a side path with a one-deep leaky hand-off, and joins its detections to the metadata by `seq`/`pts`. The side path only reads the frame, so sharing it is safe.
- An engine that is not ready yet means "video plus no detections", not a build error.

## L4. IMX296 on Argus: what the first device run will show (Medium, plan for it)

These are known, and none is in `design.md`'s risk table:

1. **Argus auto exposure swings the brightness.** The product caps gain at 16 (`gainrange="1 16"`) and can pin exposure (`exposuretimerange="N N"`). `capture_launch` sets neither. Add `exposure_us`, `gain_max` and `gain_min` to `SourceConfig` now, as a pinned launch-string test like the existing ones.
2. **White balance from the NITO does nothing.** The image patches `nvarguscamerasrc` with a `wbgains` property (meta-vc-camera `recipes-multimedia/gstreamer`), and the app runs its own AWB through it.
   - Expect a green cast until the sensor vendor's NITO arrives.
   - Detection accuracy on the device will differ from the sim. Pin the golden-set mAP gate to the tuned image, not to the first boot.
3. **The first frames after a start are dark**, while AE converges. Skip the first N frames, about 1 s, for any golden or mAP comparison and for "first frame" health.
4. **cam1 stops after 13–15 s under Argus** (a hardware timeout, cause open). The product ships it disabled. The "30 FPS per camera, 2 cameras" target is blocked by this until the hardware cause is found. Say so in the targets table.

## L5. NVMM: you already have the surface, and probably don't need `nvmmalloc` for M1 (Medium, simplification)

**Getting the surface.** Round 2's R2-1 is right that `GstDmaBufMemory` will not match. But the fix is smaller than "add the nvmmalloc allocator":

- On JetPack, `gst_buffer_map(READ)` of an NVMM buffer returns an `NvBufSurface*` in `map.data`, not pixels.
- That is exactly what the shipping product does: `orin-nx-camera-viber/embedded/Machines/Jetson/TensorRt/src/TensorRt/NvmmFramePreprocessor.cpp:106-113`.
- From the surface, read:
  - `surfaceList[0].bufferDesc`: the dmabuf fd;
  - `.layout`: authoritative, so stop hardcoding `Layout::BlockLinear` in `JetsonPlatform::source`;
  - `.pitch` and `.planeParams`: the plane metadata round 1 asked for (P1-7).
- This needs `unsafe`, and `platform-jetson` has `#![forbid(unsafe_code)]`. Put it in a small `nvbuf-sys` crate: bindgen over `nvbufsurface.h` from meta-tegra's `tegra-mmapi_39.2.1`. Keep a safe `NvmmSurface::from_gst(&gst::Buffer)` wrapper that checks `map.size() == size_of::<NvBufSurface>()`.

**Feeding NVENC.** If the encoder pushes **the same `gst::Buffer`** it got from the capture `appsink` into the encode `appsrc`, NVENC receives NVMM memory from Argus's own allocator. That is what happens inside a single `nvarguscamerasrc ! nvv4l2h264enc` pipeline. `nvmmalloc` is needed only once *you* allocate the NVMM pool, which is the later direct-libargus path.

This is a hypothesis to check first on the board:

- Does `nvv4l2h264enc` accept a buffer that came from another pipeline?
- Are there `NvVideoRegisterMemHandle` errors?
- Does `copied=0` hold?

If it holds, M1 loses its biggest dependency. Keep the PTS rebased to the encode pipeline's running time.

## L6. Encoder details the product already paid for (Medium)

Compare `crates/platform-jetson/src/launch.rs` with `orin-nx-camera-viber/embedded/Machines/Jetson/src/Jetson/JetsonPipelineElements.cpp` and its pinned `JetsonLaunchStringCheck.cpp`:

- **`nvv4l2h264enc` defaults to Baseline profile.** The product sets `profile=2` (Main). The Rust H.264 string doesn't, so add it and pin it in the test.
- **`idrinterval=30` is hardcoded.** The product derives the GOP from config and also sets `iframeinterval` when the GOP is long (`JetsonEncoderTuner`). Make it a config value.
- **The nvv4l2 encoders report "Failed to process frame" when a flush-start reaches them** (`EncoderTakesFlush() == false`). So a looping file source must not loop with a flushing seek; restart the clip at EOS instead. Keep this in mind for the hardware-CI "recorded clip" source.
- **The encode output is asynchronous.** `Encoder::encode(frame, &mut FnMut(EncodedPacket))` implies packets come out during the call. With `appsrc → nvv4l2 → appsink`, they come out later on another thread. Either block on the appsink, which couples video latency to the encoder's pipeline depth, or change the trait to a pull/poll for packets. Decide this before writing the NVENC stage.

## L7. Clocks are part of the latency budget (Medium)

The product's numbers are 3.05 ms inference and 3.9 ms capture to encoded on the NX. They came from a CUDA graph **plus pinned engine clocks through devfreq**, which also works on fused units (`camera-clocks.service`; the LPR image adds a VIC floor in `lpr-clocks.service`). Unpinned DVFS gives a different, noisier latency, and the < 100 ms p95 gate would measure the governor, not the code.

Make "clocks pinned" a precondition of the M1/M2 measurement, and list it in `design.md` next to the Nsight/EMC gates.

## L8. Running on the camera's Yocto image is not free (High if this replaces camera-streamer)

`design.md` targets the camera-viber Yocto image. That image has rules a new binary must meet, or it bricks the A/B update path:

- **boot-ok.** A boot counts as good only when `/run/camera-streamer/boot-ok` is written. meta-tegra's `nv_update_verifier` waits for it. If `edge-vision` replaces camera-streamer and writes no marker, every update rolls back after the bootloader's retries.
  - Write the marker only after a real health check: a frame from an enabled camera, not just "up 60 s". The C++ product got that wrong too (see its 2026-10-23 review, P1-9).
- **systemd watchdog.** The product uses `WatchdogSec=15`, and the watchdog pet must not sit behind anything that touches the disk.
- **Sandbox and camera-guard (BPF LSM).** The service runs as a non-root user under `ProtectSystem=strict`. camera-guard limits what it may exec, write and connect to; it ships in `audit` and is set to `enforce` per camera. A new binary needs its own entries in `ebpf/camera-guard/guard.h` and its own grants: the `video` group for `/dev/nvhost-*` and the Argus socket.
- **The disk stalls.** `/recordings` is a DRAM-less NVMe that stalls under sustained writes. `FileVideoSink` and `JsonlMetaSink` write synchronously on the processing thread (`crates/sinks/src/lib.rs`). On the device, that stalls the frame loop. Disk writes belong on their own bounded writer thread.
- **Port clash.** `config/orin-nx-yolo.toml` uses `rtsp_port = 8554`, the same as camera-streamer. Pick another port, or state that the two never run together.
- **Toolchain.**
  - The pinned OE-core ships **Rust 1.98.1** (`repos/openembedded-core/meta/recipes-devtools/rust/rust_1.98.1.bb`), so the MSRV bump to 1.92 for gstreamer-rs 0.25 is safe.
  - Align `tools/Dockerfile.cross` with 1.98 rather than a lower version, so cross builds and image builds see the same compiler.
  - Plan the recipe with OE's `cargo` class and the crate fetcher from `Cargo.lock`; offline fetching is mandatory in the image build.

## L9. The model's licence (Medium, product)

`config/orin-nx-yolo.toml` defaults to `models/yolov8n.onnx`. Ultralytics YOLOv8 weights are **AGPL-3.0**.

The shipping product deliberately uses YOLOX, which is Apache-2.0. Its model signer refuses a package without `license`/`provenance`, or with a non-commercial licence (`tools/cam model sign`).

This repo already has a YOLOX decoder. Make YOLOX the device default, keep YOLOv8 for the sim and tests only, and record the licence next to the model path.

---

## What carries over well

- **The sysfs NVENC/NVJPG probe** (`probe.rs`) is the product's `MediaEngineProbe` lesson, done right: the Nano image carries the plugins without the engine.
- **Pinned launch strings tested like code.** Keep doing it, and add the items in L4 and L6 to the pins.
- **Two of the product's rules are already in the design:** in-place analytics must sit before any tee, and strict zero-copy with an explicit opt-in for the Nano CPU copy.
- **`copied_frames` as a loud regression counter**, plus refusing to copy NVMM. That is the right instinct.

## Suggested order (M1 bring-up)

1. **First, on the board:**
   - L5 surface access through `gst_buffer_map`, plus the same-`gst::Buffer` NVENC hypothesis. It decides how big M1 is.
   - L2 Argus pool count and in-flight budget.
2. **Before writing the NVENC stage:** L6, the encoder trait's packet model, the profile and the GOP. L3, the split video and inference paths, with overlay optional.
3. **Before a soak test:** L1, EOS ambiguity, the once-per-boot recovery and the exit-75 unit. L4, the AE caps and skipping the first frames.
4. **Before installing on a camera image:** L8, boot-ok, the sandbox, camera-guard, the disk writer and the port. L9, the model licence.
5. **For the measurements:** L7, pin clocks.

---

## Acknowledgement — 2026-10-23 (implementing agent)

Read in full; nothing here is disputed. Applied immediately (PC-side):
L4.1 exposure/gain caps are now `SourceConfig` fields rendered into the
pinned capture string (`gain_max = 16.0` in the Orin profile); L6 profile=2
(Main) and config-driven GOP (`gop_frames`) in all encoder strings; L8 port
moved to 8555 and the cross image aligned to Rust 1.98; L9 YOLOX
(Apache-2.0) is the device default model with provenance recorded in the
profile, live `queue_depth = 2` per L2. The rest is recorded in design.md
as "M1 device constraints" — L1 (process-level Argus recovery, exit-75,
one process per camera default), L2 (in-flight surface budget), L3 (clean
video + metadata boxes, video never gated on inference, engine-warm
streaming), L5 (`gst_buffer_map` → `NvBufSurface*` via a future `nvbuf-sys`
crate; same-`gst::Buffer` NVENC hypothesis first on the board), L7
(clocks pinned as a measurement precondition), L8 (boot-ok, watchdog,
sandbox/camera-guard, bounded disk writer). None of it is claimed done;
each item closes on the board or in the M1 stage work it gates.
