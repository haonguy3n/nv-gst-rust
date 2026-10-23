//! The simulator is a backend like any other: it must pass the
//! shared conformance suite. A hardware backend's job in the port
//! is making this same call green on the board.

use edge_core::{ModelConfig, SourceConfig};
use platform_sim::SimPlatform;

fn source_cfg() -> SourceConfig {
    SourceConfig {
        kind: "synthetic".into(),
        path: None,
        camera_id: 0,
        width: 64,
        height: 64,
        // Unpaced: conformance pulls several pools' worth of
        // frames and must not wait on simulated sensor timing.
        fps: 30,
        exposure_us: None,
        gain_min: None,
        gain_max: None,
        paced: false,
        pool_size: 4,
        frames: None,
    }
}

fn model_cfg() -> ModelConfig {
    ModelConfig {
        path: None,
        decoder: Default::default(),
        input_color: Default::default(),
        input_normalize: Default::default(),
        pad_value: 114,
        input_colorspace: Default::default(),
        precision: Default::default(),
        device: None,
        input_width: 32,
        input_height: 32,
        num_classes: 80,
        score_threshold: 0.25,
        iou_threshold: 0.45,
        max_nms: 300,
        top_k: 100,
    }
}

#[test]
fn sim_passes_conformance_suite() {
    edge_conformance::check_all(&SimPlatform, &source_cfg(), &model_cfg());
}
