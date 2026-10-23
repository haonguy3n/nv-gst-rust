# Round 2 review — after the P0 fixes

**Date:** 2026-10-23 (after commits `a05de3f` + `2d1c917`)
**Scope:** the new `platform-jetson` GStreamer source, the runner/config changes, the
design/README updates, CI, and the claim that the review is "done".
**No product code changed by this review.** This file is a review note.

## Verdict

Real progress. The two runner bugs are fixed with committed regression tests, the encoder
decision matrix is now testable, the `gst` feature compiles and its tests pass, and the
design has a useful "Implementation status" table.

But I would **not call it done**:

1. **The new GStreamer source does not handle NVIDIA NVMM.** It only recognizes upstream
   `GstDmaBufMemory`; `nvarguscamerasrc` on JetPack uses NVIDIA's NVMM/NvBufSurface
   allocator. Unless JetPack 7.2.1 exposes `GstDmaBufMemory` there (verify on the board),
   the Orin path either fails at `map_readable()` or silently copies block-linear bytes.
   The PC test uses `videotestsrc` system memory, so it cannot catch this.
2. **A post-start GStreamer error has no error path.** `next_frame` checks only
   `appsink.is_eos()`; there is no bus polling. A camera/decoder error can return
   `Timeout` forever, and the new `FramePoll::Timeout` policy keeps the run alive. A dead
   camera becomes an infinite silent loop.
3. **P0-5 is only partially fixed.** The design body still contains many statements that
   contradict the revision and the new status table (list below).
4. **The CPU-encode opt-in is fixed at the builder level but not end-to-end.**
   `JetsonPlatform::caps()` advertises `cpu_encode = true` while `encoder()` still returns
   `None`/`Err`, and `preprocessor()` fails before the builder reaches the encoder.

## What I re-ran

| Command | Result |
| --- | --- |
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| `cargo test --workspace` | pass |
| `cargo test -p platform-sim --features onnx` | pass (2 + 1 + 2 tests) |
| `cargo test -p platform-jetson --features gst` | pass (10 unit + 2 integration) |
| `cargo clippy -p edge-core --features cpu-debug --all-targets -- -D warnings` | pass |
| `cargo test -p edge-pipeline --test sim_pipeline` | pass (4, incl. P0-1/P0-2 regressions) |
| `cargo run -p edge-vision -- config/x86-dev.toml --frames 5` | pass (5 processed, 0 timeouts) |
| `cargo run -p edge-vision --features jetson -- config/orin-nx-yolo.toml` | fails honestly: `no element "nvarguscamerasrc"` |
| temp config `pool_size=6`, `queue_depth=4` | correctly rejected: needs `>= 7` |
| `cargo build --workspace --target aarch64-unknown-linux-gnu` | **not re-run**: no `aarch64-linux-gnu-gcc` on this host; CI config now has the linker env |
| on-device GStreamer / NVMM | **not run**: no Orin here; code inspection below |

The committed P0-1 and P0-2 tests are real and pass:
`stage_error_with_unbounded_source_returns_instead_of_deadlocking` and
`smallest_legal_pool_with_slow_sink_accounts_for_every_frame`.

## P0 fixes — status

| Review item | Status | Evidence |
| --- | --- | --- |
| P0-1 deadlock | **Fixed** | `AtomicBool` stop flag, `drop(rx)` before `join`, regression test passes |
| P0-2 `Ok(None)` conflation | **Fixed** | `FramePoll` used in sim/conformance/GStreamer; `capture_timeouts`; pool validation `>= queue + 3`; tests pass |
| P0-3 Nano opt-in | **Builder-level fixed; real path M1** | `cpu_encode` + matrix tests; fake platform builds; `JetsonPlatform` still has no CPU encoder and `preprocessor()` is `Unsupported` |
| P0-4 aarch64 CI | **Config looks fixed; unverified locally** | CI now sets `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER` + `CC_`/`AR_`; no cross gcc here to re-link |
| P0-5 design body | **Partial** | status table added; body still stale (below) |

## New findings

### R2-1 (High): the GStreamer source only wraps `GstDmaBufMemory`, not NVMM

**Where:** [`crates/platform-jetson/src/gst_source.rs:153-184`](crates/platform-jetson/src/gst_source.rs#L153-L184),
[`crates/platform-jetson/src/lib.rs:87-107`](crates/platform-jetson/src/lib.rs#L87-L107).

```rust
if let Some(dmabuf) = mem.downcast_memory_ref::<gstreamer_allocators::DmaBufMemory>() {
    ... return DmaBuf { fd, buffer, memory: self.memory };
}
// Fallback: copy into the shm pool (dev PCs only).
```

`nvarguscamerasrc`/`nvvidconv` on JetPack produce NVIDIA NVMM buffers, not upstream
`GstDmaBufMemory`; the reference `gst-nvmm-cpp` exists to add the NVMM allocator. The
capture launch string asks for `video/x-raw(memory:NVMM)`, so the first branch is not the
one the Orin will take. The fallback then:

- calls `buffer.map_readable()` on an NVMM buffer (may fail, because NVMM is not a normal
  CPU map), or
- copies the mapped bytes into `/dev/shm` and increments `copied_frames`.

If the map succeeds, the bytes are block-linear/padded, not a tight NV12 frame, and the
source metadata still says `Layout::PitchLinear`. The PC test cannot catch any of this: it
uses a system-memory `videotestsrc` and asserts `copied_frames == 5`.

**Fix:** add an NVMM branch that obtains the `NvBufSurface*` (via `gst-nvmm-cpp` /
`GstNvBufSurface` or `gst_buffer_get_nvds_bufsurface` in DeepStream), exposes the
`bufferDesc` dmabuf fd and plane parameters, and holds the `GstBuffer`/`GstSample` alive.
Until that exists, fail loudly when `self.memory == FrameMemory::Nvmm` and the memory is
not a known dmabuf; do not silently copy.

### R2-2 (High): no GStreamer bus error handling; a dead source loops as `Timeout`

**Where:** [`crates/platform-jetson/src/gst_source.rs:195-203`](crates/platform-jetson/src/gst_source.rs#L195-L203).

`next_frame` distinguishes only "sample", "appsink is EOS", and "timeout". There is no
`pipeline.bus().timed_pop_filtered(...)` for `Error`/`Eos`. A runtime error after
`start()` (camera unplug, decoder error, caps failure later) need not set
`appsink.is_eos()`; `try_pull_sample` then returns `None` on every call, which is reported
as `FramePoll::Timeout`. The new runner policy intentionally continues on `Timeout`, so
`run()` never returns: it just accumulates `capture_timeouts`.

`start()` does catch a synchronous state-change failure, and the `filesrc=/nonexistent`
case fails there — but that is not a general runtime error path.

**Fix:** poll the bus for `Error` and `Eos` (and post `EndOfStream` / return `Err`), and add
a bounded idle policy (e.g. max consecutive timeouts or max no-frame age) so a dead camera
surfaces as an error instead of an infinite loop. Add a test with a pipeline that starts
and then errors.

### R2-3 (High, doc): P0-5 is not fixed; the body still contradicts the revision

The new status table is good, but the body still has statements the code and the table
contradict. These are the ones I would fix before calling the design done:

| `design.md` line | Still says | Contradicted by |
| --- | --- | --- |
| [`69`](design.md#L69) | "Orin Nano targets are the same at one camera" | CPU-encode profile is ~1.1 cores, not the `<25%`/one-camera target |
| [`96`/`98`](design.md#L96-L98) | RTP copies, then "the socket send is the one copy no user-space design can avoid" | At least two copies; the second sentence contradicts the first |
| [`103`](design.md#L103) | builder rejects a layout mismatch at startup | builder hardcodes `PreConfig.layout = PitchLinear`; no stage layout checks |
| [`104`](design.md#L104) | encoder hands packets out via callback "instead of allocating a `Vec`" | `EncodedPacket.data: Vec<u8>` and `MultiSink` clones it |
| [`105`](design.md#L105) | "the encoder asserts" the same fd | `debug_assert_eq!` only; compiled out in release |
| [`107`](design.md#L107) | counting allocator fails the build above 0 | no such test |
| [`120`](design.md#L120) | Nano streams "MQTT / WebSocket metadata" | only JSONL/stdout sinks exist |
| [`183`](design.md#L183) | trait sketch `next_frame -> Result<Option<HwFrame>>` | actual API is `FramePoll` |
| [`282`](design.md#L282) | capture is "libargus ... pitch-linear" from M1 | M1 is `nvarguscamerasrc`; block-linear; status table says GStreamer |
| [`287`/`288`](design.md#L287-L288) | encode is V4L2 M2M; stream wraps "V4L2 mmap, no copy" | M1 encode is `nvv4l2h26xenc` via appsrc; RTP payloading copies |
| [`295`](design.md#L295) | pool "e.g. 6 buffers" | default 8; validation is `queue + 3` |
| [`297`](design.md#L297) | cross and native builds "both covered in CI" | CI has cross only; no Orin runner |
| [`322`](design.md#L322) | conformance covers golden images and mAP | `conformance` explicitly says it does not |
| [`326`/`347`](design.md#L326-L347) | M1 is direct Argus capture | revision and status table say GStreamer |
| [`349`](design.md#L349) | Nano "no CPU encode, no frame copy" | revision decision 3 is the explicit CPU opt-in |
| [`334-339`](design.md#L334-L339) | first-week checklist all unchecked | work is done and committed |

The status table does not override the body; an agent reading the body will still implement
the stale design. Either rewrite these sections or mark each as "planned / not current".

### R2-4 (Medium): `cpu_encode = true` is advertised without a CPU encoder

**Where:** [`crates/platform-jetson/src/lib.rs:80-82`](crates/platform-jetson/src/lib.rs#L80-L82),
[`crates/platform-jetson/src/lib.rs:132-140`](crates/platform-jetson/src/lib.rs#L132-L140).

`caps()` says a CPU path exists, but `encoder()` ignores `cfg.hardware` and returns
`Err(Unsupported)` when NVENC is present, or `Ok(None)` when it is not. The builder can
therefore accept `relaxed + hardware=false` for a platform that cannot supply it; it only
fails later with "returned no encoder" — after `preprocessor()`/`engine()` would already
fail on the device. The fake-platform matrix does not exercise the real platform.

**Fix:** set `cpu_encode` only when the CPU encoder stage actually exists (feature/flag), or
return a typed `Unsupported` from `caps()`/`encoder()` that the builder turns into a clear
error. The status table should say "CPU path not implemented", not just "M1".

### R2-5 (Medium): MSRV mismatch — `gst` needs Rust 1.92, workspace says 1.85

**Where:** `Cargo.toml` (`rust-version = "1.85"`), `tools/Dockerfile.cross` (`rust:1.85`).

`gstreamer`, `gstreamer-sys`, and `gstreamer-allocators` 0.25.4 declare
`rust-version = "1.92"` (checked in the registry `Cargo.toml`). The default workspace build
on 1.85 still works because `gst` is optional, but
`cargo build -p edge-vision --features jetson` on the stated MSRV fails inside
`gstreamer-sys`. CI uses `stable` (1.99) so it will not see this.

**Fix:** bump the workspace MSRV and the cross Docker image to 1.92, or pin gstreamer 0.24
(whose MSRV fits 1.85). Add an MSRV CI job if 1.85 is a promise.

### R2-6 (Medium): GStreamer fallback is not safe on a device

**Where:** [`crates/platform-jetson/src/gst_source.rs:164-183`](crates/platform-jetson/src/gst_source.rs#L164-L183).

- `buffer.map_readable()` then `write_all(map.as_slice())` writes the raw map; if the map
  includes padding or is block-linear, the file is not a tight NV12 frame (and `write_all`
  can extend it beyond the `frame_bytes` it was sized for). There is no
  `GstFrameBuffer::read_bytes` or plane metadata, so a future stage can only reopen the fd
  and guess the layout. The PC test never reads pixels, so it proves the fd contract, not
  that the copied bytes are usable.
- Pool exhaustion returns `Err("gst source fallback pool exhausted")`, not
  `FramePoll::Timeout`; a slow consumer stops the pipeline instead of dropping a frame.
- `JetsonPlatform::source` constructs the source with `FrameMemory::Nvmm`, while
  `GstFrameBuffer::Copied::memory()` returns `DmaBuf`. The builder checks only
  `source.memory()`, so the declared domain is wrong for the frames actually produced.
- The integration test constructs the source with `FrameMemory::DmaBuf`, so it does not
  catch this mismatch.

**Fix:** require `map.size() == frame_bytes` (or read caps/pitch), return `Timeout` on pool
exhaustion, and add a conformance check that `frame.buf.memory()` matches
`source.memory()`.

### R2-7 (Medium): the GStreamer tests are PC-only and one asserts the wrong thing on a Jetson

**Where:** [`crates/platform-jetson/tests/gst_source.rs`](crates/platform-jetson/tests/gst_source.rs).

- `missing_element_fails_at_construction_with_a_clear_error` asserts that
  `nvarguscamerasrc` is missing. On the Orin, it exists, so this test fails when someone
  runs the feature tests on the device.
- `pulls_ordered_frames_from_a_launch_string` assumes `videotestsrc`/`videoconvert` are in
  the Yocto image; a minimal Jetson image may not have them.
- Neither test exercises NVMM, the `Copied`-vs-`Nvmm` mismatch, bus errors, or fallback
  stride.

**Fix:** gate the missing-element assertion on element availability/target, skip the
pattern test when the element is absent, and add the device-only checks as `#[ignore]` or
feature-gated tests.

### R2-8 (Low): `FramePoll::Timeout` documentation and metrics disagree

**Where:** [`crates/core/src/stage.rs:53-60`](crates/core/src/stage.rs#L53-L60),
[`crates/pipeline/src/metrics.rs:47-58`](crates/pipeline/src/metrics.rs#L47-L58).

`FramePoll::Timeout` says the frame "is counted as dropped by the pipeline", but the
pipeline increments only `capture_timeouts`; `frames_dropped` remains channel evictions.
The Prometheus help for `edge_frames_dropped_total` still says "or pool exhaustion", which
is no longer true. Decide the semantics and make the doc/help match.

### R2-9 (Low): README/status wording still overclaims

- [`README.md:50-54`](README.md#L50-L54): "like the Orin Nano profile — it streams
  detections only and never hands frames to a CPU encoder" is true only for `strict`;
  revision 3 explicitly allows `relaxed + hardware=false`. "the builder refuses an encoder
  request on any platform without hardware encode" is no longer true for a platform with
  `cpu_encode`.
- The "Implementation status" table says GStreamer capture has "dmabuf wrap" verified by
  PC tests; the PC test only verifies the system-memory fallback (R2-1).
- `design.md:98` still says output tensors are "around 1 MB"; YOLOv8 640×640/80-class is
  2.82 MB F32. `design.md:322` still claims mAP/golden conformance. `design.md:295` still
  says pool 6.
- The review file's header says "Not changed: no product code was modified" but the
  appended "Resolution" changed code later; split the external review from the agent's
  resolution, or update the header.

## What is good

- **P0-1** is a real fix, not a paper one: stop flag, receiver drop, bounded join, and a
  regression test that would have hung before.
- **P0-2** is real: `FramePoll`, timeout counting, pool validation, conformance update,
  and a slow-sink regression test.
- **P0-3** builder/config matrix is clear and testable; the fake-platform tests cover the
  four combinations.
- **P0-4** CI env is the right fix; the previous failure was exactly the missing linker.
- `platform-jetson --features gst` compiles and its PC tests pass; the missing-element
  error is clear; the `gst-capture` example is useful.
- The implementation-status table is a good idea and is honest about zero-copy not being
  verified on device.

## Suggested order

1. **R2-1/R2-2**: NVMM wrapping + bus errors/idle timeout. These are the M1 blockers; a
   GStreamer source that copies or loops forever is worse than one that errors.
2. **R2-3**: rewrite the design body (or delete the stale sections). The status table is
   not enough.
3. **R2-4/R2-5**: CPU capability honesty and MSRV.
4. **R2-6/R2-7**: fallback safety and device-gated tests.
5. **R2-8/R2-9**: docs/metrics/README.

Everything in P1-6…P2-19 from the first review remains open and is acknowledged in the
resolution; this round does not re-litigate it.

## Verification notes

- GStreamer 1.28.8 is installed on this host; CI uses the distro's 1.24, which
  gstreamer-rs 0.25 supports (its `system-deps` minimum is 1.14).
- `gstreamer-allocators-1.0.pc` is in `libgstreamer-plugins-base1.0-dev` on Debian/Ubuntu,
  so the CI package list is plausible; I did not find a missing `bad` dev package.
- `aarch64-linux-gnu-gcc` is not installed here, so the cross link could not be re-run;
  the CI job installs it and sets the linker env.
- No Orin is attached, so the NVMM/bus behavior is a code-level finding, not a device
  measurement.

---

## Resolution — 2026-10-23 (implementing agent)

Fixed in the round-2 pass, all verified on this PC (workspace 45 tests,
clippy clean across default/onnx/gst/jetson feature sets):

- **R2-1:** an NVMM-declared source now returns a loud `Err` ("refusing to
  copy") when a pulled buffer is not dmabuf memory; the silent block-linear
  copy is gone. The real NvBufSurface branch remains M1 device work, exactly
  as the review frames it. Regression test:
  `nvmm_pipeline_without_dmabuf_refuses_instead_of_copying`.
- **R2-2:** `next_frame` drains bus `Error`/`Eos` before and after each pull;
  a runtime pipeline error surfaces as `Err` with element + debug text.
  Regression test: `runtime_pipeline_error_surfaces_instead_of_timing_out_forever`.
  Not added: a pipeline-level idle-timeout policy — consecutive capture
  timeouts are now counted and exported, and the cap stays an operator
  decision for M1 rather than a hardcoded constant.
- **R2-3:** the listed stale body lines were rewritten (Nano target budget,
  bitstream copies incl. RTP payloading and the correct 2.82 MB tensor size,
  builder/layout wording, write-into and same-fd as implemented, counting
  allocator scoped to decoders, `FramePoll` in the trait sketch, capture
  table, pool sizing, CI runner honesty, conformance wording, M1 summary,
  Nano risk row) and the first-week checklist now shows its true state.
- **R2-4:** `JetsonPlatform::caps().cpu_encode` is `false` until an encoder
  stage exists; caps no longer advertise what `encoder()` cannot supply.
- **R2-5:** workspace `rust-version` raised 1.85 → 1.92 (verified:
  `gstreamer-sys 0.25.4` declares 1.92) and `Dockerfile.cross` moved to
  `rust:1.92-bookworm`; the aarch64 workspace build was re-verified in the
  rebuilt image. The bump surfaced two MSRV-gated clippy lints, also fixed.
- **R2-6:** fallback copies require `map.size() ==` one tight NV12 frame;
  pool exhaustion drops the frame as `Timeout`; the NVMM guard (R2-1) makes
  the `Copied`/domain mismatch unreachable on an NVMM source, and the
  conformance suite now asserts `frame.buf.memory() == source.memory()`.
- **R2-7:** GStreamer tests gate on `ElementFactory::find` (skip when the
  element is absent — or present, for the missing-element test).
- **R2-8:** one semantics everywhere: timeouts counted as timeouts,
  `frames_dropped` is post-capture eviction only (trait doc + Prometheus
  help aligned).
- **R2-9:** README CPU-encode paragraph and the design status-table
  GStreamer row corrected; the split-history note lives here rather than in
  the round-1 file's header.

Also done in this pass (round-1 leftovers): P1-6 shared `Arc<[u8]>` packet
payload, P1-7 source-declared layout into `PreConfig` (plane/pitch
descriptor deferred to M1 NVMM params, contract documented), P1-8 bounded
appsink (`max-buffers=2 drop=true`) and appsrc (`block=true
max-buffers=4`), P1-10 precision/device config validated against caps +
`FrameMemory::Other(&'static str)`, P1-11 `Box<dyn TensorBuffer>`, P1-12
`decode(&mut self)` with reusable scratch, pre-NMS cap and a
counting-allocator steady-state test (which caught and fixed a per-decode
stable-sort allocation), P1-13 private `HwFrame::meta` + release-mode fd
check, colorspace matrices (BT.601/709 × limited/full), transposed YOLO
heads, corner-clipped box sizes, engine cache key (args + arch),
`metadata_only` defaulting to true with the incoherent combination
rejected, JSONL sink ignoring video in fan-outs, unique sim pool names,
spec-driven buffer budget, u64 metrics averages, unused `Error::Timeout`
removed.

Still open and not claimed anywhere: NVMM/NvBufSurface capture and NVENC
dmabuf proof on JetPack 7 (M1, needs the board), the full plane descriptor,
pipeline-level zero-allocation gate, latency percentiles and multi-camera
(P2-14), an RTSP sink to make `rtsp_port` live (P2-15), and committed
end-to-end detector tests replacing the manual runs (P2-16 remainder).
