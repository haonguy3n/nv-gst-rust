//! Platform-agnostic core of the edge-vision pipeline.
//!
//! The pipeline only knows the traits in [`stage`] and the opaque
//! [`HwFrame`] handle. Vendor code lives in `platform-*` crates.
//! See `design.md` at the workspace root for the zero-copy contract
//! these types are written to enforce.

#![forbid(unsafe_code)]

pub mod config;
pub mod detection;
pub mod error;
pub mod frame;
pub mod letterbox;
pub mod stage;
pub mod tensor;

pub use config::{
    Codec, Config, DecoderKind, EncoderConfig, InputColor, InputColorspace, InputNormalize,
    ModelConfig, PipelineConfig, Precision, SourceConfig, StreamConfig, ZeroCopyMode,
};
pub use detection::{Detection, Detections};
pub use error::{Error, Result};
pub use frame::{FrameMemory, FrameMeta, HwBuffer, HwFrame, Layout, PixelFormat};
pub use letterbox::{normalize_factor, order_channels, Letterbox};
pub use stage::{
    EncodedPacket, Encoder, EngineState, EngineWarmup, FramePoll, FrameSource, GpuStream,
    InferenceEngine, Overlay, Platform, PlatformCaps, PostProcessor, PreConfig, Preprocessor, Sink,
};
pub use tensor::{DType, Tensor, TensorBuffer, TensorSpec};
