# Rust code review: edge-vision workspace

**Date:** 2026-10-23
**Scope:** every crate's Rust source: `core`, `pipeline`, `postproc`, `sinks`, `platform-sim` (plus `onnx`), `platform-jetson`, `conformance` and `apps/edge-vision`. This is a code-level review. Design and device topics are in `REVIEW-ROUND3-ORIN-LEAD-2026-10-23.md`.

**Tree reviewed.** The working tree was being edited during the review. I copied it at 15:37 and reviewed that copy; it is the uncommitted work on top of `2d1c917`. Line numbers refer to that copy.

**Checks on the copy:**
- `cargo fmt --check` passes.
- `cargo clippy` fails, but only in `platform-jetson`, which has a half-written `format!` ("2 positional arguments ... no arguments were given"). That edit is in progress and is not counted as a finding.

**Reproduced.** Every finding marked **[R]** was reproduced with a throwaway test on the copy. Those tests are not in the repo.

**Not repeated here.** Rounds 1 and 2 already cover NVMM/dmabuf, bus errors, MSRV, appsink bounds, the deadlock, and the `FramePoll` split.

**Already fixed in the working tree.** The following round 1/2 items now look right, so they are not reopened:
- `EncodedPacket` is `Arc<[u8]>`.
- `Tensor` owns its buffer in a `Box`.
- `PostProcessor::decode(&mut self)` reuses NMS scratch space, with an allocation gate test.
- `HwFrame::meta` is private, and the fd check runs in release builds.
- `JsonlMetaSink::push_video` now ignores video.
- Preprocessor layout and colorspace are passed through to the preprocessor.

---

## P1: crashes and wrong output

### 1. A NaN box coordinate panics the YOLOX decoder, and the release build aborts **[R]**

**Where:** `crates/postproc/src/yolox.rs:119-120`

```rust
let x = ((x1 - pad_x) / scale / fw).clamp(0.0, 1.0);         // NaN stays NaN
let width = (((x2 - pad_x) / scale / fw) - x).clamp(0.0, 1.0 - x); // max = NaN -> panic
```

`f32::clamp` panics when either bound is NaN.

**When it happens:** one NaN in `boxes` is enough. FP16 overflow gives `inf - inf`, and an engine that went wrong after a CUDA error gives NaN too. The workspace sets `panic = "abort"` for release, so the whole process dies on the processing thread.

**Repro:**
```
boxes = [NaN, 10, 20, 20], objectness = 1.0, scores = 0.9
-> panicked at core/src/num/f32.rs: min > max, or either was NaN
```

**Fix:** never let a non-finite value reach `clamp`:
```rust
if !(x1.is_finite() && y1.is_finite() && x2.is_finite() && y2.is_finite()) { continue; }
let x = ((x1 - pad_x) / scale / fw).clamp(0.0, 1.0);
let x_end = ((x2 - pad_x) / scale / fw).clamp(0.0, 1.0);
let width = x_end - x;
```
This form matches what `yolo.rs` already does: clip both corners, then subtract. Add the repro as a unit test.

### 2. NaN detections pass both decoders' filters **[R]**

**YOLO, NaN coordinates** (`crates/postproc/src/yolo.rs:128`):
- `if width <= 0.0 || height <= 0.0 { continue; }` is false for NaN, so the box is kept.
- Repro: a `cx` of NaN gives `Detection { x: NaN, width: NaN, score: 0.9 }`.
- `serde_json` writes NaN as `null`, so every JSONL consumer receives a broken box.

**YOLOX, NaN objectness** (`crates/postproc/src/yolox.rs:108`):
- `if score < self.score_threshold { continue; }` is false for NaN, so the candidate is kept.
- Repro: objectness NaN gives 1 detection with `score: NaN`.
- `total_cmp` sorts positive NaN above every real score. That box then wins NMS and can suppress real boxes that overlap it.

**Fix:** write each filter so that NaN fails it:
- `if !(score >= self.score_threshold)`
- `if !(width > 0.0 && height > 0.0)`

Add one test per decoder with NaN and inf inputs.

### 3. Config typos are accepted silently **[R]**

**Where:** `crates/core/src/config.rs`. No struct has `#[serde(deny_unknown_fields)]`.

**Repro:** a config with `pool_sise = 2` and `enabeld = true` loads with no error. The pool silently keeps its default, and the encoder silently stays off, so the camera runs metadata-only. The project's goal 4 is "config-driven", so a typo has to be an error.

**Fix:** add `#[serde(deny_unknown_fields)]` to `Config`, `SourceConfig`, `ModelConfig`, `EncoderConfig`, `StreamConfig` and `PipelineConfig`. Add a test with a misspelled key. `toml` then reports the unknown key and its position.

### 4. The runner's error path leaks the pipeline state and hides the first error

**Where:** `crates/pipeline/src/lib.rs:243-296`, `Pipeline::run`.

1. **`stop()` is skipped.** When `source.next_frame(...)?` (line 256) returns `Err`, the closure returns early and `source.stop()?` (line 269) never runs. The source is dropped inside the thread, and a GStreamer source stays PLAYING until its `Drop`.
2. **The pipeline is left without its source.** `capture.join()...??` (line 295) returns before `self.source = source` (line 296), so `self.source` is still the `NoopSource` swapped in at the start. The next `run()` on the same `Pipeline` returns `Ok` at once, with 0 frames and EndOfStream.
3. **The real error is lost.** When processing fails (`last_err`) and capture then fails, the capture error wins and the processing error is dropped. The metrics gathered so far are dropped too.

**Fix:**
- Have the thread always return its source: `-> (Box<dyn FrameSource>, Result<(u64, u64)>)`.
- Call `stop()` on every path: a small guard, or `let r = loop_body(); let s = source.stop(); (source, r.and(s))`.
- Restore `self.source` before inspecting any result.
- Prefer `last_err` over the capture error.
- Return the partial metrics with the error, for example as a field of the error.

### 5. There is no way to stop `run()` from outside

`run()` ends only on EndOfStream, the frame limit, or an error. There is no stop handle, and the app installs no signal handler (`apps/edge-vision/src/main.rs` says "lands with the Jetson daemon work"). But `--features jetson` already builds the real platform. So SIGTERM from systemd, or Ctrl-C, kills the process with the capture pipeline still PLAYING, and buffered metadata is lost (see item 7).

**Fix:**
- Add `Pipeline::stop_handle() -> Arc<AtomicBool>`. The capture loop already checks `capture_stop`, so pass the same flag in.
- In the app, set it from `signal-hook` or `ctrlc` on SIGINT and SIGTERM.
- Add a test that stops an unbounded sim source from another thread within, say, 500 ms.

### 6. A stalled source counts timeouts forever

The capture loop does `FramePoll::Timeout => timeouts += 1` with no bound, and the counter is only visible after `run()` returns. Round 2 fixed this for GStreamer, which now polls bus errors. The runner still has no policy for any other source.

**Fix:** add `pipeline.max_idle` (for example 5 s without a frame). Return `Err` once it is exceeded, so systemd or the caller sees a dead camera rather than an idle process.

---

## P2: robustness, API shape, performance

### 7. `JsonlMetaSink` and `FileVideoSink` never flush
Both wrap a `BufWriter<File>` and rely on `Drop` (`crates/sinks/src/lib.rs:23-26`, `:60`). `Drop` swallows flush errors. An abort or kill loses up to 8 KiB, and a `tail -f` reader sees data in bursts.

Fix:
- Flush per frame for JSONL; the lines are small.
- Flush per keyframe or per second for video.
- Add a `Sink::finish(&mut self) -> Result<()>` that the runner calls on every exit path, so that errors surface.

### 8. `Detections` cannot be attributed or synchronised
`Detections` carries only `frame_seq`. With two cameras, nothing in the metadata says which camera a box belongs to. With video, nothing joins the metadata to a packet: `EncodedPacket` has `pts`, `Detections` has none.

Fix: add `camera_id` and `pts` to `Detections`, and have `clear(meta)` take the `FrameMeta`, so decoders cannot forget them.

### 9. Stringly-typed choices are validated in three places
`platform`, `source.kind`, `model.decoder` and `model.device` are `String`s. They are matched in:
- `Config::validate`,
- the app (`main.rs:96-110`, where `_ =>` silently falls back to the sim decoder),
- each `Platform::source`.

Fix: make `decoder` and `source.kind` enums with `#[serde(rename_all = "lowercase")]`. Serde then rejects bad values with a position, and every `match` is checked for exhaustiveness. Keep `device` a string, since it is backend-defined, but validate it in one place only (the builder already does).

### 10. Validation gaps in `Config::validate`
- `gain_min > gain_max`, negative gain, NaN gain, and `exposure_us = 0` are all accepted. They go straight into a launch string on the Jetson.
- `bitrate_kbps = 0` and `gop_frames = 0` are accepted. On the encoder side `bitrate_kbps.saturating_mul(1000)` silently caps at about 4.29 Gbit/s.
- `num_classes = 0` is accepted. The YOLO decoder then expects 4 channels and silently finds nothing.
- `self.pipeline.queue_depth + 3` (line 306) can overflow for a huge TOML value. It panics in debug and wraps in release. Use `saturating_add`, and check `queue_depth >= 1` *before* this.
- Line 349 has a lost `\` continuation, so the error text contains 18 spaces: `"...is disabled;                  enable the encoder..."`.
- The doc on `pool_size` (line 50) says "frames in flight + 2", but the rule enforced is `queue_depth + 3`.

### 11. `FrameSource::layout()` has a default
`stage.rs:88` returns `Layout::PitchLinear` unless overridden. A new backend that forgets to override it silently claims pitch-linear, which is the exact bug class the design wants to stop. Nothing checks that each frame's `meta.layout` equals `source.layout()`.

Fix: remove the default so every backend must state its layout, and add that check to `conformance::check_frame_order_and_timestamps`.

### 12. `PlatformCaps` has two fields for one concept
`devices` and `accelerators` both list "gpu", "dla0" and so on (`stage.rs:143-153`). The builder checks `devices`; nothing reads `accelerators`.

The sim also claims `devices: ["cpu", "gpu"]`, while the default `model.device` is `"gpu"`. So the sim accepts "gpu" and runs on the CPU.

Fix: keep one field, have the sim claim only "cpu", and make the default device come from the platform rather than from a hardcoded `"gpu"`.

### 13. The engine's input is not checked against the config
- The builder now reads `input_spec` but never compares its shape with `[1, 3, model.input_height, model.input_width]` or its dtype with `model.precision`.
- `OnnxEngine::load` builds its input binding from the config and never reads `session.inputs()`.

So a 416×416 model with a 640 config fails only at the first `infer`, with an ORT error, or worse, silently on a dynamic-shape model.

Fix: in `OnnxEngine::load`, read the declared input shape and refuse a mismatch. In `Pipeline::build`, check the spec and fail with `DomainMismatch { stage: "engine", ... }`.

### 14. `OnnxEngine::infer` allocates heavily on every frame
Per frame, `crates/platform-sim/src/onnx.rs`:
- copies the input with `.to_vec()`: 4.9 MB at 640²,
- copies every output with `.to_vec()`,
- builds new `Tensor`s and `String` names,
- and replaces `self.outputs`, so before the first frame `output_bindings()` has empty shapes.

It is sim-only, but the bench numbers include it, and it is the reference for how a backend should hold bindings.

Fix: use `TensorRef::from_array_view` over the input slice, keep preallocated output tensors (or ORT `IoBinding`), and copy into them in place.

### 15. Channel and thread details
- `channel::Receiver::recv_timeout` (`channel.rs:82-101`) waits the *full* timeout again after a spurious wakeup, so a wait can exceed `timeout`. Track a deadline and wait for the time remaining.
- The design names `crossbeam`; either use `crossbeam_channel` (bounded, plus `force_send` for drop-oldest) or record why there is a custom channel.
- The capture thread is unnamed (`std::thread::spawn`, `lib.rs:243`). Use `thread::Builder::new().name(format!("capture-cam{}", cfg.camera_id))`, which shows up in `top -H`, gdb, perf and panic messages, and handle the spawn error.

### 16. Post-processing tuning is hard-coded, and the candidates are sorted twice
- `iou_threshold = 0.45`, `max_nms = 300` and `top_k = 100` are not in `ModelConfig`.
- Over `max_nms`, each decoder fully sorts all candidates (`sort_unstable_by` plus `truncate`), and `nms` then sorts again. `select_nth_unstable_by(max_nms)` makes the first step O(n).

### 17. Smaller items
- The sim `File` source allocates `vec![0u8; frame_bytes]` every frame (`platform-sim/src/lib.rs`, `next_frame`). Reuse `self.scratch`.
- `conformance::check_clean_shutdown` asserts nothing beyond "no panic". Add a check that every pooled buffer came back: a pool `available()` count is enough for sim and GStreamer.
- `metrics.frames_captured` counts frames the capture thread *took*, and `--frames N` limits captures, not processed frames. With drops, `--frames 30` processes fewer than 30. Document this, or limit on processed frames.

---

## Process

1. **Metrics are visible only at the end.** `PipelineMetrics` is built locally in `run()` and returned when it ends, so `--prometheus` prints once at exit. That doesn't meet the "Observable" goal for a long-running service. Put the counters in an `Arc<…>` of atomics shared by the capture and processing threads, and serve them while the pipeline runs. Round 1's P2-14, on percentiles, builds on this.
2. **CI gaps** (`.github/workflows/ci.yml`):
   - Nothing builds `edge-vision --features jetson`, the feature combination the app actually ships with.
   - There is no MSRV (1.92) job.
   - There is no feature-powerset check. `cargo hack check --feature-powerset --workspace` would have caught round 1's U-1 build break.
   - The aarch64 job builds default features only.
3. **One test per finding.** Items 1–6 each have a natural regression test; add those with the fixes.

## What is good

- `#![forbid(unsafe_code)]` everywhere except the test allocator. It holds.
- Typed errors (`thiserror`), with `anyhow` only in the binary.
- `FramePoll` makes "timeout" and "end of stream" separate states.
- The steady-state allocation gate on the decoders is a real test.
- The conformance suite runs one set of checks against every backend, and catches broken stubs.
- The launch strings are pinned by tests.
- The builder refuses invalid configurations with errors that say what to change.

## Suggested order

1. Items 1–2: the NaN guards and their tests. These are crashes and wrong output today.
2. Item 3: `deny_unknown_fields`.
3. Items 4–6: the runner's error path, a stop handle with signal handling, and the idle limit.
4. Items 7–8: sink flushing, plus `camera_id`/`pts` in `Detections`.
5. The rest, then the CI additions.

## Resolution — 2026-10-23 (implementing agent)

All items addressed in this pass, verified on this PC: `cargo fmt
--check`, clippy `-D warnings` across default / `onnx` / `gst` /
`jetson,onnx`, workspace tests (57 passed, 1 pre-existing ignored),
`platform-sim --features onnx` (6), `platform-jetson --features
gst` (14), aarch64 check of the app with `onnx`, and a 30-frame
`--prometheus` smoke run. The reviewer's probe tests were rescued
from the scratchpad before it expired and are the basis of the
items 1–3 regression tests.

- **1–2 (NaN):** both decoders drop non-finite coordinates before
  any `clamp`/cast, and the score/size filters are written so NaN
  fails them (`is_nan || <`). Regression tests in `postproc` from
  the reviewer's probes.
- **3:** `#[serde(deny_unknown_fields)]` on all six config
  structs; the `pool_sise`/`enabeld` probe is a regression test.
- **4:** `Pipeline::run` returns `(PipelineMetrics, Result<()>)`;
  partial metrics survive errors (capture counts are returned on
  the error path too), the source is always restored — including
  on capture-thread spawn failure, via a hand-back slot — and the
  processing error wins. `stop()` runs on every capture path.
  Tests: stage-error return, run-twice, stall counts.
- **5:** `Pipeline::stop_handle()` exposes the `AtomicBool`; the
  app installs a Ctrl-C/SIGTERM handler. The flag is cleared at
  the start of each run so shutdown cannot poison a rerun.
- **6:** `pipeline.max_idle_ms` (default 5000; 0 disables) fails
  the run when the source goes silent; the sim's pacing now
  honours the caller's timeout so this is testable off-device.
- **7:** `Sink::finish()` added; the runner calls it on every
  exit path. JSONL flushes per line, the file video sink on
  keyframes; `MultiSink` finishes all children, first error wins.
- **8:** `Detections` carries `camera_id` and `pts`;
  `clear(&FrameMeta)` makes attribution unforgettable.
- **9:** `model.decoder` is a `DecoderKind` enum — serde rejects
  unknown values, the app match is exhaustive, the hand-rolled
  validation is gone.
- **10:** gain (finite, > 0, min ≤ max), `exposure_us = 0`,
  zero bitrate/GOP with the encoder enabled, and
  `num_classes = 0` are rejected; the pool floor uses
  `saturating_add`; the mangled error string is repaired.
- **11:** `FrameSource::layout()` has no default; forgetting it
  is a compile error in all five impls.
- **12:** `PlatformCaps::accelerators` deleted; `model.device`
  is `Option<String>` validated once, in the builder; the sim
  claims only `cpu`.
- **13:** the builder compares the engine's declared input
  shape/dtype with the config (`DomainMismatch { stage:
  "engine" }`); `OnnxEngine::load` reads `session.inputs()` and
  refuses a static-shape or non-F32 mismatch at load. Both have
  regression tests.
- **14:** `infer` passes a `TensorRef` view over the input
  binding (no input clone); outputs are preallocated from their
  declared shapes and overwritten in place.
- **15:** `recv_timeout` waits against one deadline; the
  capture thread is named (`capture-cam{id}`) with the spawn
  error handled; the custom channel is kept and `channel.rs`
  records why (eviction must return the frame to the sender).
- **16:** `iou_threshold`, `max_nms`, `top_k` are `ModelConfig`
  fields wired through decoder builders; the pre-NMS cap is
  `select_nth_unstable_by`, O(n).
- **17:** the sim file source reuses `self.scratch`;
  `check_clean_shutdown` asserts every pooled buffer came back
  via the new `FrameSource::pool_available()`; `--frames` is
  documented as limiting captures (help text + README).
- **Process 1:** counters live in an `Arc<LiveMetrics>` shared
  by both threads; `Pipeline::live_metrics()` hands it out and
  `--prometheus` prints live summaries during the run. Test
  observes progress mid-run from another thread.
- **Process 2:** CI now builds/clippy's the shipped
  `jetson,onnx` combination, runs `cargo hack check
  --feature-powerset`, has an MSRV (1.92) job, and cross-checks
  the app's `onnx` feature for aarch64. The `jetson` (GStreamer)
  cross-check still needs an arm64 pkg-config sysroot and is
  explicitly left to the device-sysroot work.
- **Process 3:** items 1–8 and 13 each landed with a regression
  test; the rest are covered by existing tests where noted.
