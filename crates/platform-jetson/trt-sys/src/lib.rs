//! TensorRT and the CUDA runtime for the Jetson inference path.
//!
//! Role: Adapter. A C++ shim (`shim/trt_shim.cpp`, a C ABI that catches
//! every exception) makes TensorRT's C++ API callable, and this crate
//! wraps it in safe Rust types. Every `unsafe` of the inference path
//! lives here (decision D8); `platform-jetson` forbids it.
//!
//! The types in `types` are plain Rust and always build. The wrappers
//! (`Runtime`, `Engine`, `Context`, `Stream`, `MappedBuffer`, ...) exist
//! only with the `shim` feature, which needs the TensorRT and CUDA
//! development files (the Yocto sysroot); see `build.rs` for the
//! environment variables.
//!
//! Ownership order, enforced by `Arc` links so a wrong drop order cannot
//! be written: `Context` holds its `Engine`, `Engine` holds its `Runtime`.
//! A `Graph` must be dropped before the `Context` it was captured from
//! (the owner declares it first), and mapped buffers must outlive both.

#![deny(unsafe_op_in_unsafe_fn)]

mod types;
pub use types::{
    severity_to_level, BuildOptions, DType, DeviceInfo, Error, IoDesc, IoMode, ProgressEvent,
    Result,
};

#[cfg(feature = "cuda-preprocess")]
mod cuda_preprocess;
#[cfg(feature = "shim")]
mod ffi;
#[cfg(feature = "shim")]
mod shim;
#[cfg(feature = "cuda-preprocess")]
pub use cuda_preprocess::RgbaTensorWriter;
#[cfg(feature = "shim")]
pub use shim::{
    build_plan, device_info, set_device_for_thread, version, Context, Engine, Event, Graph,
    MappedBuffer, Plan, Runtime, Stream,
};
