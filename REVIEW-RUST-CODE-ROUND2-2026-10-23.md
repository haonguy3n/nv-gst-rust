# Rust code review, round 2: after `e910ffc`

**Date:** 2026-10-23
**Scope:** committed `HEAD` (`6514176`; the code is in `e910ffc`), exported clean, plus the uncommitted C1 work in the tree at 17:15.
**No product code was changed by this review.**

## Checks (run on a clean export of HEAD)

| Command | Result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -D warnings` | pass |
| `cargo test --workspace` | 57 passed |
| clippy + test `-p platform-sim --features onnx` | pass, 6 tests |
| clippy + test `-p platform-jetson --features gst` | pass, 14 tests |
| clippy + build `-p edge-vision --features jetson,onnx` | pass |
| clippy `-p edge-core --features cpu-debug` | pass |
| Working tree (C1 in progress): clippy | **fails**: unused import `GpuStream` in `crates/conformance/src/lib.rs` |
| Working tree (C1 in progress): `cargo test --workspace` | 58 passed |

## The 17 items from round 1: verified fixed

| Item | Checked in code |
|---|---|
| 1. YOLOX NaN panic | A finiteness check runs before every `clamp` (`yolox.rs`). Regression test `nan_box_does_not_panic_and_nan_objectness_is_dropped`. |
| 2. NaN detections pass | `score.is_nan() \|\| score < thr` and `!(w > 0 && h > 0)` in both decoders, with tests for NaN, ±inf and NaN score. |
| 3. Config typos | `deny_unknown_fields` on all 6 structs. Test `misspelled_keys_are_rejected`. |
| 4. Runner error path | The source travels through a slot and is restored before any result is read. `stop()` runs on every path, the processing error wins, partial metrics are returned, and a failed spawn returns the source. |
| 5. External stop | `stop_handle()` plus `ctrlc` with `termination`, which covers SIGTERM. |
| 6. Stalled source | `pipeline.max_idle_ms` (default 5 s) turns a long silence into `Err`. |
| 7. Sink flush | `Sink::finish` runs on every exit path. JSONL flushes per line, video per keyframe, and `MultiSink` returns the first error. |
| 8. Attribution | `Detections` carries `camera_id` and `pts`; `clear(&FrameMeta)` sets both. |
| 9. String-typed config | `DecoderKind` enum; `device: Option<String>`, validated in the builder. |
| 10. Validation gaps | Gain, exposure, bitrate, GOP, `num_classes`, IoU, `max_nms`, `top_k` are checked; `saturating_add`; the stray spaces in the message are gone. |
| 11. Layout default | The default is removed; every source states its layout. |
| 12. Duplicate caps field | `accelerators` is gone, and the sim claims only `"cpu"`. |
| 13. Engine input check | The ONNX loader checks the declared input's dtype and shape, and the builder checks the spec. |
| 14. ONNX allocations | `TensorRef::from_array_view` on the input, and output buffers reused through `replace`. |
| 15. Channel and threads | One deadline in `recv_timeout`, with a test. The capture thread is named. The choice of a custom channel is now documented. |
| 16. Post-processing tuning | IoU, `max_nms` and `top_k` are in config; the pre-cap uses `select_nth_unstable_by`. |
| 17. Smaller items | The sim file source reuses its scratch buffer, and conformance now checks for pool leaks through `pool_available()`. |

CI gained the shipped `jetson,onnx` build, `cargo hack --feature-powerset`, an MSRV 1.92 job and an aarch64 `onnx` check. All good.

---

## New findings at HEAD

None of these is a crash. The first three change behaviour in the field.

### 1. A stop request can be lost (P2)
`run()` begins with `self.stop.store(false)` (`pipeline/src/lib.rs:291`), because the runner sets the same flag itself to stop capture (`:364`). So a SIGINT or SIGTERM that arrives after `ctrlc::set_handler` (`main.rs:137`) but before `run()` resets the flag is silently ignored. The same goes for a supervisor that calls stop between two runs. On the device, the window will include future warm-up and start work.

**Fix:** give the runner its own per-run flag for teardown (`let shutdown = Arc::new(AtomicBool::new(false))`), and have the capture loop stop on `external_stop || shutdown`. Never reset the external flag inside `run()`; reset it only through an explicit `Pipeline::clear_stop()`.

### 2. A stalled consumer is reported as "source stalled" (P2)
The idle timer in `capture_loop` counts any run of `FramePoll::Timeout`. But the sim and GStreamer sources also return `Timeout` when their **pool is exhausted**, and on Argus a held buffer starves the source. So when the *processing* side stops for more than 5 s (a sink blocked on a stalling disk, a slow first inference), the run ends with `source stalled: no frame for …`. The camera was fine; the consumer stopped.

**Fix:** make the message and the decision look at both sides. Before blaming the source, check whether the processing side moved (`frames_processed` from `LiveMetrics`, or whether the channel is full), and report "processing stalled for X; capture pool exhausted" when it didn't. Better still, have `FramePoll::Timeout` say why (`NoFrame` vs `PoolExhausted`).

### 3. `pts` serialises as `{"secs":…, "nanos":…}` in JSONL (P3)
`Detections::pts: Duration` uses serde's default `Duration` form, so every JSONL line now contains a nested object. A consumer joining metadata to video wants one number.

**Fix:** serialise it as `pts_ns: u64`, for example with `#[serde(rename = "pts_ns", serialize_with = "…")]`. Extend `jsonl_lines_are_flushed_and_attributed_before_drop` to assert the format; today it checks only `frame_seq` and `camera_id`.

### 4. A race can drop the last frame unprocessed (P3)
In the processing loop, `if rx.is_closed_and_empty() || capture.is_finished() { break; }` (`pipeline/src/lib.rs:349`) can break with one frame still queued:
1. `recv_timeout` gives up on an empty queue,
2. capture then sends its last frame and exits,
3. and `is_finished()` is now true.

That frame is neither processed nor counted as dropped. `is_closed_and_empty()` already covers the thread's exit, because the `Sender` is dropped with the thread, even on a panic.

**Fix:** remove `|| capture.is_finished()`.

### 5. `capture_timeouts` now counts normal pacing (P3)
The paced sim source now honours the poll timeout (`wait > timeout` → `Timeout`), which is correct for the trait. But with the runner's 100 ms poll, any source below 10 fps reports several "timeouts" per frame, while the metric's help text says "e.g. pool temporarily exhausted". Rename it to "polls without a frame", or count only pool exhaustion (with the `Timeout` reason from item 2).

### 6. Live metrics details (P3)
- `frames_dropped` is published only after a frame is processed (`sync_processing`), so drops are invisible exactly when processing stalls. `LiveMetrics` could hold the channel's own `Arc<AtomicU64>`.
- `reset()` at every `run()` makes the counters go back to 0. Prometheus treats that as a counter reset, which is fine, but say so in the doc comment.

### 7. Second Ctrl-C does nothing (P3)
If teardown hangs (a source's `stop()`, a blocked sink), a second Ctrl-C has no effect, and the only way out is SIGKILL. A common pattern: the first signal sets stop, the second calls `std::process::exit(130)`.

### 8. Small items (P3)
- `apps/edge-vision/src/main.rs:2` still says "Signal handling lands with the Jetson daemon work".
- At HEAD, the doc comment of `stop_handle` is attached to `live_metrics` (`pipeline/src/lib.rs:231-241`). The C1 diff in the tree fixes this.
- `OnnxEngine::infer` still collects a `Vec<usize>` shape per output per frame. It is tiny; compare it in place and allocate only when it changes.

---

## The C1 work in the tree (uncommitted)

`GpuStream` becomes an opaque `Arc<dyn Any + Send + Sync>` that the platform creates (`Platform::stream`), and the pipeline owns one per camera. This matches the plan's C1 and D7, and the downcast test is right.

Two notes before committing it:
- **clippy fails:** `crates/conformance/src/lib.rs` still imports `GpuStream`, which it no longer uses.
- **`Platform::stream` has a default that returns `noop()`.** That is the same pattern as the layout default removed in item 11: a Jetson backend that forgets to override it gets a no-op stream silently. Either make `stream()` required, as `layout()` now is, or have every Jetson stage refuse a no-op stream, `stream.backend::<CudaStream>().ok_or(...)`, with a test. Required is more consistent.

## Suggested order

1. Commit C1, with the import fix and the decision on the `stream()` default.
2. Items 1–2: the stop flag and stall attribution. Both decide how the service behaves under systemd on the device.
3. Items 3–4: the `pts` format and the race. One line each, plus a test.
4. The rest whenever the files are touched.
