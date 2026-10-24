//! Error type of the create/map/transform API.

use crate::SurfaceError;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NvbufError {
    /// A C call returned a non-zero code.
    Call { func: &'static str, code: i32 },
    /// The surface the driver gave back is not what the caller asked for.
    Geometry(String),
    /// The input bytes are not an NvBufSurface.
    Surface(SurfaceError),
}

impl fmt::Display for NvbufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Call { func, code } => write!(f, "{func} failed with code {code}"),
            Self::Geometry(what) => write!(f, "bad surface geometry: {what}"),
            Self::Surface(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for NvbufError {}

impl From<SurfaceError> for NvbufError {
    fn from(e: SurfaceError) -> Self {
        Self::Surface(e)
    }
}
