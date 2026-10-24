//! Platform-agnostic core of the edge-vision pipeline.
//!
//! The pipeline only knows the traits in [`stage`] and the opaque
//! [`HwFrame`] handle. Vendor code lives in `platform-*` crates.
//! See `docs/design.md` for the zero-copy contract these types are
//! written to enforce.

#![forbid(unsafe_code)]

// `cpu-debug` is a development-only escape hatch: release builds must
// never contain a path that maps hardware pixels to the CPU. A feature
// accidentally enabled by a dev-dependency fails this build.
#[cfg(all(feature = "cpu-debug", not(debug_assertions)))]
compile_error!("feature `cpu-debug` must never be enabled in a release build");

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
pub use frame::{
    FrameMemory, FrameMeta, HwBuffer, HwFrame, Layout, PixelFormat, PlaneDesc, PlaneLayout,
};
pub use letterbox::{normalize_factor, order_channels, Letterbox};
pub use stage::{
    EncodedPacket, Encoder, EngineState, EngineWarmup, FramePoll, FrameSource, GpuStream,
    InferenceEngine, Overlay, Platform, PlatformCaps, PostProcessor, PreConfig, Preprocessor, Sink,
    TimeoutReason,
};
pub use tensor::{DType, Tensor, TensorBuffer, TensorSpec};
