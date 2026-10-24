use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// One TOML file picks the platform, camera, model, encoder and outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// Simulator only: pace frame production at `fps` like a real
    /// sensor. Set `false` for throughput benches and burst tests.
    /// Hardware sources ignore this (the sensor sets the pace).
    #[serde(default = "default_true")]
    pub paced: bool,
    /// Simulated backend's preallocated buffer pool; the minimum is
    /// `pipeline.queue_depth + 3`. On Jetson the Argus pool is
    /// owned by `nvarguscamerasrc` and this value is not used.
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
    /// Argus capture pool size on a Jetson. When set, the builder
    /// validates the live in-flight surface budget (appsink, channel,
    /// processing, appsrc and encoder, plus one for the CPU-encode
    /// copy). Left unset until the board's count is known.
    #[serde(default)]
    pub argus_pool: Option<usize>,
    /// Pinned exposure time in microseconds (Argus
    /// `exposuretimerange`). `None` lets auto-exposure run; AE
    /// brightness swings were a real device problem, so the caps
    /// are config, not launch-string folklore.
    #[serde(default)]
    pub exposure_us: Option<u32>,
    /// Analog gain bounds (Argus `gainrange`). The reference
    /// product caps gain at 16 to stop AE brightness swings.
    #[serde(default)]
    pub gain_min: Option<f32>,
    #[serde(default)]
    pub gain_max: Option<f32>,
    /// Stop after this many frames (sim/CI). `None` runs forever.
    #[serde(default)]
    pub frames: Option<u64>,
}

fn default_pool_size() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// ONNX model on sim; `.engine` built on-device on Jetson.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Output decoder. Serde rejects unknown values with a
    /// position, and every match on it is exhaustive.
    #[serde(default)]
    pub decoder: DecoderKind,
    /// Channel order the model expects (YOLOv8: RGB, YOLOX: BGR).
    #[serde(default)]
    pub input_color: InputColor,
    /// Pixel scaling the model expects (YOLOv8: 0..1, YOLOX: 0..255).
    #[serde(default)]
    pub input_normalize: InputNormalize,
    /// Letterbox padding value in byte domain (114 in both families).
    #[serde(default = "default_pad_value")]
    pub pad_value: u8,
    /// YUV->RGB matrix/range the frames arrive in; must match the
    /// training pipeline or accuracy drifts silently.
    #[serde(default)]
    pub input_colorspace: InputColorspace,
    /// Inference precision to build/select (TensorRT FP16/INT8 on
    /// device; FP32 in sim). Validated against platform caps.
    #[serde(default)]
    pub precision: Precision,
    /// Accelerator to run on: backend-defined name ("gpu", "dla0",
    /// "cpu", ...), validated against platform caps in the
    /// builder (the one place). `None` takes the platform's
    /// first declared device.
    #[serde(default)]
    pub device: Option<String>,
    pub input_width: u32,
    pub input_height: u32,
    #[serde(default = "default_num_classes")]
    pub num_classes: u32,
    #[serde(default = "default_score_threshold")]
    pub score_threshold: f32,
    /// NMS IoU threshold.
    #[serde(default = "default_iou_threshold")]
    pub iou_threshold: f32,
    /// Cap on candidates entering NMS (bounds its O(n^2)).
    #[serde(default = "default_max_nms")]
    pub max_nms: usize,
    /// Cap on detections kept after NMS.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DecoderKind {
    /// Deterministic sim engine rows.
    #[default]
    Sim,
    /// YOLOv8/v11 raw head, `[1, 4 + classes, N]` (or transposed).
    Yolo,
    /// YOLOX three outputs: boxes xyxy, objectness, scores.
    Yolox,
}
fn default_pad_value() -> u8 {
    114
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InputColor {
    #[default]
    Rgb,
    Bgr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum InputNormalize {
    /// Pixels scaled to 0.0..=1.0.
    #[default]
    Unit,
    /// Pixels left in 0.0..=255.0.
    None,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    Fp32,
    Fp16,
    Int8,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp32 => f.write_str("fp32"),
            Self::Fp16 => f.write_str("fp16"),
            Self::Int8 => f.write_str("int8"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum InputColorspace {
    /// BT.601 limited range (most cameras/ISP defaults).
    #[default]
    Bt601Limited,
    Bt709Limited,
    Bt601Full,
    Bt709Full,
}

fn default_num_classes() -> u32 {
    80
}
fn default_score_threshold() -> f32 {
    0.25
}

fn default_iou_threshold() -> f32 {
    0.45
}

fn default_max_nms() -> usize {
    300
}

fn default_top_k() -> usize {
    100
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Codec {
    H264,
    H265,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Keyframe interval in frames (GOP length).
    #[serde(default = "default_gop_frames")]
    pub gop_frames: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            codec: Codec::H264,
            hardware: true,
            bitrate_kbps: default_bitrate(),
            gop_frames: default_gop_frames(),
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
fn default_gop_frames() -> u32 {
    30
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamConfig {
    /// Detections only, no video. Defaults to true: video is what
    /// a config asks for explicitly (with an enabled encoder), so
    /// an omitted field can never silently promise video that
    /// will not arrive.
    #[serde(default = "default_true")]
    pub metadata_only: bool,
    /// Also write detections to this file as JSON lines (one line
    /// per frame). Useful for metadata-only profiles and debugging.
    #[serde(default)]
    pub meta_file: Option<PathBuf>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            metadata_only: true,
            meta_file: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineConfig {
    /// Bounded channel depth between capture and processing.
    #[serde(default = "default_queue_depth")]
    pub queue_depth: usize,
    /// Fail the run when no frame arrives for this long. A dead
    /// camera must be an error a supervisor can act on, not an
    /// idle process counting timeouts forever. 0 disables.
    #[serde(default = "default_max_idle_ms")]
    pub max_idle_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            queue_depth: default_queue_depth(),
            max_idle_ms: default_max_idle_ms(),
        }
    }
}

fn default_max_idle_ms() -> u64 {
    5_000
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
        if !s.width.is_multiple_of(2) || !s.height.is_multiple_of(2) {
            return Err(Error::invalid(
                "source.width/height",
                "must be even for NV12",
            ));
        }
        if !(1..=240).contains(&s.fps) {
            return Err(Error::invalid("source.fps", "must be within 1..=240"));
        }
        // The pool must cover the channel queue plus capture and
        // in-flight processing, or pool exhaustion masquerades as
        // a starved (and formerly truncated) run.
        let min_pool = self.pipeline.queue_depth.saturating_add(3);
        if s.pool_size < min_pool {
            return Err(Error::invalid(
                "source.pool_size",
                format!(
                    "must be >= pipeline.queue_depth + 3 (>= {min_pool} here), got {}",
                    s.pool_size
                ),
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
        if !self.encoder.enabled && !self.stream.metadata_only {
            return Err(Error::invalid(
                "stream.metadata_only",
                "false promises video but the encoder is disabled; \
                 enable the encoder or set metadata_only = true",
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
        for (name, gain) in [
            ("source.gain_min", s.gain_min),
            ("source.gain_max", s.gain_max),
        ] {
            if let Some(g) = gain {
                if !g.is_finite() || g <= 0.0 {
                    return Err(Error::invalid(name, "must be finite and > 0"));
                }
            }
        }
        if let (Some(lo), Some(hi)) = (s.gain_min, s.gain_max) {
            if lo > hi {
                return Err(Error::invalid("source.gain_min", "must be <= gain_max"));
            }
        }
        if s.exposure_us == Some(0) {
            return Err(Error::invalid("source.exposure_us", "must be > 0 when set"));
        }
        if self.encoder.enabled {
            if self.encoder.bitrate_kbps == 0 {
                return Err(Error::invalid(
                    "encoder.bitrate_kbps",
                    "must be > 0 when the encoder is enabled",
                ));
            }
            if self.encoder.gop_frames == 0 {
                return Err(Error::invalid(
                    "encoder.gop_frames",
                    "must be > 0 when the encoder is enabled",
                ));
            }
        }
        if m.num_classes == 0 {
            return Err(Error::invalid("model.num_classes", "must be > 0"));
        }
        if !(0.0..=1.0).contains(&m.iou_threshold) {
            return Err(Error::invalid(
                "model.iou_threshold",
                "must be within 0.0..=1.0",
            ));
        }
        if m.max_nms == 0 {
            return Err(Error::invalid("model.max_nms", "must be > 0"));
        }
        if m.top_k == 0 {
            return Err(Error::invalid("model.top_k", "must be > 0"));
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
    fn misspelled_keys_are_rejected() {
        // Regression (review item 3, from the reviewer's probe):
        // pool_sise/enabeld used to load silently with defaults.
        let typo = VALID
            .replace("fps = 30", "fps = 30\npool_sise = 2")
            .replace("[model]", "[encoder]\nenabeld = true\n[model]");
        let err = Config::from_toml_str(&typo).unwrap_err().to_string();
        assert!(
            err.contains("pool_sise") || err.contains("enabeld"),
            "error must name the unknown key: {err}"
        );
    }

    /// Item 9: the decoder is an enum, so an unknown value is a
    /// parse error naming the variants, not a silent fallback.
    #[test]
    fn unknown_decoder_value_is_rejected() {
        let bad = VALID.replace("[model]", "[model]\ndecoder = \"yollo\"");
        let err = Config::from_toml_str(&bad).unwrap_err().to_string();
        assert!(
            err.contains("yollo") || err.contains("unknown variant"),
            "{err}"
        );
    }

    #[test]
    fn parses_minimal_config_with_defaults() {
        let cfg = Config::from_toml_str(VALID).expect("valid config");
        assert_eq!(cfg.source.pool_size, 8);
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
