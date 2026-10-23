//! Conformance suite: trait-level checks every `platform-*` backend
//! must pass before it counts as a port (see design.md, "Porting
//! guide"). A backend runs it from its own test target:
//!
//! ```ignore
//! #[test]
//! fn conformance() {
//!     edge_conformance::check_all(&MyPlatform, &source_cfg(), &model_cfg());
//! }
//! ```
//!
//! Covered: honest caps, frame order and monotonic timestamps,
//! frame metadata matching the config, the same-fd contract, pool
//! exhaustion dropping instead of deadlocking (and recovering when
//! buffers are returned), the preprocessor writing into the
//! engine's input binding, repeatable inference, the encoder
//! option matching the advertised caps, and clean shutdown.
//!
//! Not covered here (needs a model, a clip and hardware): golden
//! pre-processing values beyond "written and finite" — use
//! [`assert_slice_close`] in backend-specific tests for that — and
//! detection mAP against the ONNX reference.

#![forbid(unsafe_code)]

use edge_core::{
    EncoderConfig, FramePoll, HwFrame, ModelConfig, Platform, PreConfig, SourceConfig,
};

/// Unwrap a poll that must yield a frame.
fn expect_frame(poll: FramePoll, platform: &dyn Platform, context: &str) -> HwFrame {
    match poll {
        FramePoll::Frame(frame) => frame,
        FramePoll::Timeout(reason) => {
            panic!("{}: {context}: timed out ({reason:?})", platform.name())
        }
        FramePoll::EndOfStream => panic!("{}: {context}: source ended early", platform.name()),
    }
}
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(2);

/// Assert two float slices match element-wise within `tol`.
/// For backend-specific golden-image pre-processing tests.
pub fn assert_slice_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "slice length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: actual {a} vs expected {e} (tol {tol})"
        );
    }
}

/// Run every conformance check against `platform`.
pub fn check_all(platform: &dyn Platform, source_cfg: &SourceConfig, model_cfg: &ModelConfig) {
    check_caps_are_honest(platform, source_cfg);
    check_frame_order_and_timestamps(platform, source_cfg);
    check_pool_exhaustion_drops_and_recovers(platform, source_cfg);
    check_preprocess_and_inference(platform, source_cfg, model_cfg);
    check_encoder_option_matches_caps(platform);
    check_clean_shutdown(platform, source_cfg);
}

pub fn check_caps_are_honest(platform: &dyn Platform, source_cfg: &SourceConfig) {
    let caps = platform.caps();
    assert!(
        source_cfg.width <= caps.max_width && source_cfg.height <= caps.max_height,
        "{}: config {}x{} exceeds advertised max {}x{}",
        platform.name(),
        source_cfg.width,
        source_cfg.height,
        caps.max_width,
        caps.max_height
    );
    if caps.has_hw_encoder {
        assert!(
            !caps.codecs.is_empty(),
            "{}: advertises a HW encoder but no codecs",
            platform.name()
        );
    }
}

pub fn check_frame_order_and_timestamps(platform: &dyn Platform, cfg: &SourceConfig) {
    let caps = platform.caps();
    let mut cfg = cfg.clone();
    cfg.frames = None; // this check decides how many frames to pull
    let mut source = platform
        .source(&cfg)
        .unwrap_or_else(|e| panic!("{}: creating source: {e}", platform.name()));
    assert_eq!(
        source.memory(),
        caps.frame_memory,
        "{}: source memory domain differs from caps",
        platform.name()
    );
    source
        .start()
        .unwrap_or_else(|e| panic!("{}: source start: {e}", platform.name()));

    let mut last_seq: Option<u64> = None;
    let mut last_pts = Duration::ZERO;
    for _ in 0..5 {
        let frame = expect_frame(
            source
                .next_frame(TIMEOUT)
                .unwrap_or_else(|e| panic!("{}: next_frame: {e}", platform.name())),
            platform,
            "next_frame",
        );
        assert_eq!(
            frame.meta().width,
            cfg.width,
            "{}: frame width",
            platform.name()
        );
        assert_eq!(
            frame.meta().height,
            cfg.height,
            "{}: frame height",
            platform.name()
        );
        assert_eq!(frame.meta().camera_id, cfg.camera_id);
        assert_eq!(
            frame.meta().dmabuf_fd,
            frame.buf.dmabuf_fd(),
            "{}: FrameMeta fd must be the buffer's fd (same-fd contract)",
            platform.name()
        );
        assert!(
            frame.buf.dmabuf_fd() >= 0,
            "{}: dmabuf fd must be valid",
            platform.name()
        );
        assert_eq!(
            frame.buf.memory(),
            caps.frame_memory,
            "{}: produced frame's memory domain must match the declared source domain",
            platform.name()
        );
        if let Some(prev) = last_seq {
            assert!(
                frame.meta().seq > prev,
                "{}: frame seq must strictly increase ({} after {})",
                platform.name(),
                frame.meta().seq,
                prev
            );
            assert!(
                frame.meta().pts >= last_pts,
                "{}: frame pts must be monotonic",
                platform.name()
            );
        }
        last_seq = Some(frame.meta().seq);
        last_pts = frame.meta().pts;
    }
    source
        .stop()
        .unwrap_or_else(|e| panic!("{}: source stop: {e}", platform.name()));
}

pub fn check_pool_exhaustion_drops_and_recovers(platform: &dyn Platform, cfg: &SourceConfig) {
    let mut cfg = cfg.clone();
    cfg.frames = None;
    let mut source = platform
        .source(&cfg)
        .unwrap_or_else(|e| panic!("{}: creating source: {e}", platform.name()));
    source
        .start()
        .unwrap_or_else(|e| panic!("{}: source start: {e}", platform.name()));

    // Hold every pooled buffer. The next call must return promptly
    // (a dropped frame) instead of blocking the sensor forever.
    let mut held = Vec::new();
    for i in 0..cfg.pool_size {
        let frame = expect_frame(
            source
                .next_frame(TIMEOUT)
                .unwrap_or_else(|e| panic!("{}: next_frame {i}: {e}", platform.name())),
            platform,
            "pool fill",
        );
        held.push(frame);
    }
    let started = Instant::now();
    let extra = source
        .next_frame(Duration::from_millis(100))
        .unwrap_or_else(|e| panic!("{}: exhausted next_frame: {e}", platform.name()));
    assert!(
        matches!(extra, FramePoll::Timeout(_)),
        "{}: exhausted pool must report Timeout (never EndOfStream, never buffer {})",
        platform.name(),
        cfg.pool_size + 1
    );
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{}: exhausted pool must not deadlock",
        platform.name()
    );

    // Returning the buffers (by drop) must replenish the pool:
    // no leaked buffers after a full cycle.
    drop(held);
    let mut recovered = 0;
    for _ in 0..cfg.pool_size {
        if matches!(
            source
                .next_frame(TIMEOUT)
                .unwrap_or_else(|e| panic!("{}: recovery next_frame: {e}", platform.name())),
            FramePoll::Frame(_)
        ) {
            recovered += 1;
        }
    }
    assert_eq!(
        recovered,
        cfg.pool_size,
        "{}: pool must fully replenish after frames are dropped",
        platform.name()
    );
    source
        .stop()
        .unwrap_or_else(|e| panic!("{}: source stop: {e}", platform.name()));
}

pub fn check_preprocess_and_inference(
    platform: &dyn Platform,
    source_cfg: &SourceConfig,
    model_cfg: &ModelConfig,
) {
    let mut source = platform
        .source(source_cfg)
        .unwrap_or_else(|e| panic!("{}: creating source: {e}", platform.name()));
    let mut engine = platform
        .engine(model_cfg)
        .unwrap_or_else(|e| panic!("{}: creating engine: {e}", platform.name()))
        // Conformance measures the ready engine; warming (plan
        // build) may legitimately take a while on device.
        .wait_ready(Duration::from_secs(600))
        .unwrap_or_else(|e| panic!("{}: engine warmup: {e}", platform.name()));
    let mut pre = platform
        .preprocessor(&PreConfig {
            src_width: source_cfg.width,
            src_height: source_cfg.height,
            dst_width: model_cfg.input_width,
            dst_height: model_cfg.input_height,
            layout: source.layout(),
            color: model_cfg.input_color,
            colorspace: model_cfg.input_colorspace,
            normalize: model_cfg.input_normalize,
            pad_value: model_cfg.pad_value,
        })
        .unwrap_or_else(|e| panic!("{}: creating preprocessor: {e}", platform.name()));
    let stream = platform
        .stream(source_cfg.camera_id)
        .unwrap_or_else(|e| panic!("{}: creating stream: {e}", platform.name()));

    source
        .start()
        .unwrap_or_else(|e| panic!("{}: source start: {e}", platform.name()));
    let frame = expect_frame(
        source
            .next_frame(TIMEOUT)
            .unwrap_or_else(|e| panic!("{}: next_frame: {e}", platform.name())),
        platform,
        "preprocess check",
    );

    // Inference must be repeatable on the same engine instance.
    for round in 0..2 {
        {
            let inputs = engine.input_bindings();
            assert!(
                !inputs.is_empty(),
                "{}: engine exposes no input bindings",
                platform.name()
            );
            let input = &mut inputs[0];
            let expected_elems =
                3 * model_cfg.input_width as usize * model_cfg.input_height as usize;
            assert_eq!(
                input.spec.num_elements(),
                expected_elems,
                "{}: input binding must match model config (3xHxW)",
                platform.name()
            );
            pre.run(&frame, input, &stream)
                .unwrap_or_else(|e| panic!("{}: preprocessor round {round}: {e}", platform.name()));
        }
        // Where the input is host-visible, the preprocessor must
        // actually have written finite, non-trivial values into the
        // binding (write-into, never return).
        if let Some(values) = engine.input_bindings()[0].as_f32_slice() {
            assert!(
                values.iter().all(|v| v.is_finite()),
                "{}: preprocessor wrote non-finite values",
                platform.name()
            );
            assert!(
                values.iter().any(|&v| v != 0.0),
                "{}: preprocessor left the input binding all zeros",
                platform.name()
            );
        }
        engine
            .infer(&stream)
            .unwrap_or_else(|e| panic!("{}: infer round {round}: {e}", platform.name()));
        assert!(
            !engine.output_bindings().is_empty(),
            "{}: engine exposes no output bindings after infer",
            platform.name()
        );
    }
    source
        .stop()
        .unwrap_or_else(|e| panic!("{}: source stop: {e}", platform.name()));
}

pub fn check_encoder_option_matches_caps(platform: &dyn Platform) {
    let caps = platform.caps();
    let mut cfg = EncoderConfig {
        enabled: true,
        ..EncoderConfig::default()
    };
    if let Some(&codec) = caps.codecs.first() {
        cfg.codec = codec;
    }
    let encoder = platform
        .encoder(&cfg)
        .unwrap_or_else(|e| panic!("{}: encoder(): {e}", platform.name()));
    assert_eq!(
        encoder.is_some(),
        caps.has_hw_encoder,
        "{}: encoder() must return Some exactly when caps advertise a HW encoder",
        platform.name()
    );
}

pub fn check_clean_shutdown(platform: &dyn Platform, cfg: &SourceConfig) {
    let mut source = platform
        .source(cfg)
        .unwrap_or_else(|e| panic!("{}: creating source: {e}", platform.name()));
    source
        .start()
        .unwrap_or_else(|e| panic!("{}: source start: {e}", platform.name()));
    let _ = source
        .next_frame(TIMEOUT)
        .unwrap_or_else(|e| panic!("{}: next_frame: {e}", platform.name()));
    source
        .stop()
        .unwrap_or_else(|e| panic!("{}: source stop: {e}", platform.name()));
    // The frame above was dropped: every pooled buffer must be
    // back. A leak here starves the next start.
    if let Some(available) = source.pool_available() {
        assert_eq!(
            available,
            cfg.pool_size,
            "{}: pool leak across start/stop ({available} of {} buffers back)",
            platform.name(),
            cfg.pool_size
        );
    }
    // Dropping a stopped source (and any frames above) must not hang
    // or panic; reaching this line with `drop` is the assertion.
    drop(source);
}
