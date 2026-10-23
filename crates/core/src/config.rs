use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One TOML file picks the platform, camera, model, encoder and outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// `"sim"` always available; `"jetson"` requires a build with that
    /// platform feature, and fails with a clear error otherwise.
    pub platform: String,
    #[serde(default)]
    pub zero_copy: ZeroCopyMode,
    pub source: SourceConfig,
    pub model: ModelConfig,
    #[serde(default)]
    pub encoder: EncoderConfig,
    #[serde(default)]
    pub stream: StreamConfig,
    #[serde(default)]
    pub pipeline: PipelineConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ZeroCopyMode {
    /// Refuse any profile that would hand frames to the CPU.
    #[default]
    Strict,
    /// Development escape hatch. Still never inserts silent converters.
    Relaxed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// `"synthetic"` | `"file"` (sim), `"argus"` (jetson), `"v4l2"` (nxp).
    pub kind: String,
    /// Raw NV12 file for `kind = "file"`.
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub camera_id: u8,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Preallocated buffer pool; must be >= frames in flight + 2.
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
    /// Stop after this many frames (sim/CI). `None` runs forever.
    #[serde(default)]
    pub frames: Option<u64>,
}

fn default_pool_size() -> usize {
    6
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// ONNX model on sim; `.engine` built on-device on Jetson.
    #[serde(default)]
    pub path: Option<PathBuf>,
    pub input_width: u32,
    pub input_height: u32,
    #[serde(default = "default_num_classes")]
    pub num_classes: u32,
    #[serde(default = "default_score_threshold")]
    pub score_threshold: f32,
}

fn default_num_classes() -> u32 {
    80
}
fn default_score_threshold() -> f32 {
    0.25
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Codec {
    H264,
    H265,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_codec")]
    pub codec: Codec,
    /// Hardware encoder only. `zero_copy = "strict"` refuses `false`.
    #[serde(default = "default_true")]
    pub hardware: bool,
    #[serde(default = "default_bitrate")]
    pub bitrate_kbps: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            codec: Codec::H264,
            hardware: true,
            bitrate_kbps: default_bitrate(),
        }
    }
}

fn default_codec() -> Codec {
    Codec::H264
}
fn default_true() -> bool {
    true
}
fn default_bitrate() -> u32 {
    8_000
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamConfig {
    /// Required on SoCs without a HW encoder (Orin Nano): detections
    /// go out over MQTT/WebSocket, frames never touch the CPU.
    #[serde(default)]
    pub metadata_only: bool,
    #[serde(default = "default_rtsp_port")]
    pub rtsp_port: u16,
}

fn default_rtsp_port() -> u16 {
    8554
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Bounded channel depth between capture and processing.
    #[serde(default = "default_queue_depth")]
    pub queue_depth: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            queue_depth: default_queue_depth(),
        }
    }
}

fn default_queue_depth() -> usize {
    4
}

impl Config {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_toml_str(&text)
    }

    pub fn from_toml_str(text: &str) -> Result<Self> {
        let cfg: Config = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.platform.is_empty() {
            return Err(Error::invalid("platform", "must not be empty"));
        }
        let s = &self.source;
        if s.width == 0 || s.height == 0 {
            return Err(Error::invalid("source.width/height", "must be > 0"));
        }
        if s.width % 2 != 0 || s.height % 2 != 0 {
            return Err(Error::invalid(
                "source.width/height",
                "must be even for NV12",
            ));
        }
        if !(1..=240).contains(&s.fps) {
            return Err(Error::invalid("source.fps", "must be within 1..=240"));
        }
        if s.pool_size < 3 {
            return Err(Error::invalid(
                "source.pool_size",
                "must be >= frames in flight + 2 (>= 3)",
            ));
        }
        if s.kind == "file" && s.path.is_none() {
            return Err(Error::invalid(
                "source.path",
                "required when source.kind = \"file\"",
            ));
        }
        let m = &self.model;
        if m.input_width == 0 || m.input_height == 0 {
            return Err(Error::invalid("model.input_width/height", "must be > 0"));
        }
        if !(0.0..=1.0).contains(&m.score_threshold) {
            return Err(Error::invalid(
                "model.score_threshold",
                "must be within 0.0..=1.0",
            ));
        }
        if self.zero_copy == ZeroCopyMode::Strict && self.encoder.enabled && !self.encoder.hardware
        {
            return Err(Error::invalid(
                "encoder.hardware",
                "zero_copy = \"strict\" refuses a CPU encoder; \
                 use a SoC with a hardware encoder or stream metadata only",
            ));
        }
        if self.encoder.enabled && self.stream.metadata_only {
            return Err(Error::invalid(
                "stream.metadata_only",
                "conflicts with encoder.enabled = true",
            ));
        }
        if self.pipeline.queue_depth == 0 {
            return Err(Error::invalid("pipeline.queue_depth", "must be >= 1"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
platform = "sim"
[source]
kind = "synthetic"
width = 640
height = 480
fps = 30
[model]
input_width = 320
input_height = 320
"#;

    #[test]
    fn parses_minimal_config_with_defaults() {
        let cfg = Config::from_toml_str(VALID).expect("valid config");
        assert_eq!(cfg.source.pool_size, 6);
        assert_eq!(cfg.zero_copy, ZeroCopyMode::Strict);
        assert_eq!(cfg.pipeline.queue_depth, 4);
    }

    #[test]
    fn rejects_odd_dimensions_for_nv12() {
        let bad = VALID.replace("width = 640", "width = 641");
        assert!(Config::from_toml_str(&bad).is_err());
    }

    #[test]
    fn strict_mode_refuses_cpu_encoder() {
        let bad = format!("{VALID}\n[encoder]\nenabled = true\nhardware = false\n");
        let err = Config::from_toml_str(&bad).unwrap_err().to_string();
        assert!(err.contains("strict"), "unexpected error: {err}");
    }

    #[test]
    fn file_source_requires_path() {
        let bad = VALID.replace("kind = \"synthetic\"", "kind = \"file\"");
        assert!(Config::from_toml_str(&bad).is_err());
    }
}
