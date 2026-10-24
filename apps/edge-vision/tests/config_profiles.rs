//! The shipped profiles must at least parse and validate.

use edge_core::{Codec, Config, ZeroCopyMode};
use std::path::PathBuf;

fn profile(name: &str) -> Config {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../config")
        .join(name);
    Config::load(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn x86_sim_profile_is_metadata_only() {
    let cfg = profile("x86-dev.toml");
    assert!(cfg.stream.metadata_only);
    assert!(!cfg.encoder.enabled);
}

#[test]
fn orin_nano_profile_is_relaxed_cpu_h264() {
    let cfg = profile("orin-nano-yolo.toml");
    assert_eq!(cfg.zero_copy, ZeroCopyMode::Relaxed);
    assert!(cfg.encoder.enabled);
    assert!(!cfg.encoder.hardware);
    assert_eq!(cfg.encoder.codec, Codec::H264);
    assert!(!cfg.stream.metadata_only);
    // No at-start gain/exposure ranges on the Nano (black-frame risk).
    assert!(cfg.source.exposure_us.is_none());
    assert!(cfg.source.gain_min.is_none() && cfg.source.gain_max.is_none());
}

#[test]
fn orin_nx_profile_is_strict_hardware_h264() {
    let cfg = profile("orin-nx-yolo.toml");
    assert_eq!(cfg.zero_copy, ZeroCopyMode::Strict);
    assert!(cfg.encoder.hardware);
    assert_eq!(cfg.pipeline.max_idle_ms, 5000);
}
