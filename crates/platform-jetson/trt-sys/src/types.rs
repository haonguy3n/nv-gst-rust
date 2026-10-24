//! Plain data shared with the wrappers; no FFI, so it builds and is
//! tested on a PC.

use std::fmt;
use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

/// A TensorRT/CUDA failure with the shim's message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// Tensor element type, as the shim codes it (see `trt_shim.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I8,
    I32,
    Bool,
    U8,
    I64,
    Bf16,
    Other,
}

impl DType {
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::I8,
            3 => Self::I32,
            4 => Self::Bool,
            5 => Self::U8,
            6 => Self::I64,
            7 => Self::Bf16,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    Input,
    Output,
}

impl IoMode {
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Input),
            1 => Some(Self::Output),
            _ => None,
        }
    }
}

/// One I/O tensor of an engine. `shape` is the engine's: `-1` marks a
/// dynamic dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoDesc {
    pub name: String,
    pub mode: IoMode,
    pub dtype: DType,
    pub shape: Vec<i64>,
}

/// Device 0 as the plan key needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub sm: (u32, u32),
    /// `cudaRuntimeGetVersion`, for example 13020.
    pub cuda_runtime: u32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub fp16: bool,
    pub workspace_bytes: u64,
    /// Shared across models: later builds reuse earlier tactic timings.
    pub timing_cache: Option<PathBuf>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            fp16: false,
            workspace_bytes: 256 << 20,
            timing_cache: None,
        }
    }
}

/// TensorRT's build progress. Nested phases carry their parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEvent<'a> {
    PhaseStart {
        phase: &'a str,
        parent: Option<&'a str>,
        steps: i32,
    },
    Step {
        phase: &'a str,
        step: i32,
    },
    PhaseFinish {
        phase: &'a str,
    },
}

/// TensorRT's severity (0 internal error .. 4 verbose) as a `log` level.
/// Info and verbose stay below `info`: a build prints hundreds of lines.
pub fn severity_to_level(severity: i32) -> log::Level {
    match severity {
        0 | 1 => log::Level::Error,
        2 => log::Level::Warn,
        3 => log::Level::Debug,
        _ => log::Level::Trace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_codes_match_the_shim_header() {
        assert_eq!(DType::from_code(0), DType::F32);
        assert_eq!(DType::from_code(1), DType::F16);
        assert_eq!(DType::from_code(6), DType::I64);
        assert_eq!(DType::from_code(99), DType::Other);
        assert_eq!(DType::from_code(-1), DType::Other);
    }

    #[test]
    fn io_mode_codes() {
        assert_eq!(IoMode::from_code(0), Some(IoMode::Input));
        assert_eq!(IoMode::from_code(1), Some(IoMode::Output));
        assert_eq!(IoMode::from_code(-1), None);
    }

    #[test]
    fn severities_map_to_log_levels() {
        assert_eq!(severity_to_level(0), log::Level::Error);
        assert_eq!(severity_to_level(1), log::Level::Error);
        assert_eq!(severity_to_level(2), log::Level::Warn);
        assert_eq!(severity_to_level(3), log::Level::Debug);
        assert_eq!(severity_to_level(4), log::Level::Trace);
    }

    #[test]
    fn default_build_uses_a_bounded_workspace() {
        let o = BuildOptions::default();
        assert_eq!(o.workspace_bytes, 256 * 1024 * 1024);
        assert!(!o.fp16);
    }
}
