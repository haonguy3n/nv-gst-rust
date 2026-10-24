//! The decisions of the TensorRT path that need no GPU: what a model
//! file is, where plans live, which engine I/O is acceptable, and when
//! the CUDA graph is captured. Role: pure policy, so the PC tests all
//! of it; `trt.rs` and `warmup.rs` only apply it to the device.

use edge_core::{DType, Error, Precision, Result, TensorSpec};
use std::path::{Path, PathBuf};
use trt_sys::{IoDesc, IoMode};

/// Environment variable that moves the plan cache (tests, dev boxes).
pub const CACHE_ENV: &str = "EDGE_VISION_CACHE";

/// TensorRT's workspace limit. The default is all device memory, which on
/// the Orin is all RAM.
pub const WORKSPACE_BYTES: u64 = 256 << 20;

/// What `model.path` names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// An ONNX model: built on first use, cached by `plan_cache`.
    Onnx(PathBuf),
    /// A plan someone built already (`trtexec`): loaded as it is.
    Plan(PathBuf),
}

impl ModelSource {
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("onnx") => Ok(Self::Onnx(path.to_path_buf())),
            Some("engine" | "plan") => Ok(Self::Plan(path.to_path_buf())),
            _ => Err(Error::invalid(
                "model.path",
                format!(
                    "{}: expected .onnx (built and cached on the device) or .engine/.plan (a prebuilt TensorRT plan)",
                    path.display()
                ),
            )),
        }
    }
}

/// The plan cache directory: `EDGE_VISION_CACHE` when set and not empty,
/// else the systemd `CacheDirectory`.
pub fn cache_dir(env_value: Option<&str>) -> PathBuf {
    match env_value {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(crate::plan_cache::DEFAULT_CACHE_DIR),
    }
}

/// The builder settings the plan key must cover besides the precision.
pub fn builder_flags(precision: Precision) -> Result<String> {
    match precision {
        Precision::Fp32 | Precision::Fp16 => Ok(format!(
            "ws={}MiB;batch=1;profile=static",
            WORKSPACE_BYTES >> 20
        )),
        Precision::Int8 => Err(Error::Unsupported(
            "int8 needs a calibration cache in the plan key; use fp16 or fp32".into(),
        )),
    }
}

/// The shared timing cache: one per TensorRT version, for every model.
pub fn timing_cache_path(cache: &Path, trt_version: i32) -> PathBuf {
    cache.join(format!("timing.trt{trt_version}.cache"))
}

/// First line of `<root>/etc/nv_tegra_release` (the BSP in the plan key).
pub fn read_l4t(root: &Path) -> String {
    std::fs::read_to_string(root.join("etc/nv_tegra_release"))
        .ok()
        .and_then(|text| text.lines().next().map(|l| l.trim().to_string()))
        .filter(|l| !l.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Checks the engine's I/O set before anything is allocated: one input,
/// at least one output.
pub fn check_io_set(ios: &[IoDesc]) -> Result<()> {
    let inputs = ios.iter().filter(|io| io.mode == IoMode::Input).count();
    let outputs = ios.len() - inputs;
    if inputs != 1 {
        return Err(Error::Unsupported(format!(
            "the engine has {inputs} inputs; exactly one image input is supported"
        )));
    }
    if outputs == 0 {
        return Err(Error::Unsupported("the engine has no output".into()));
    }
    Ok(())
}

/// The shape to set on an input whose batch is dynamic (`-1` in dimension
/// 0), pinned to 1. `None` for a fully static input. Any other dynamic
/// dimension is refused: the plan build refuses it too.
pub fn input_shape_fix(io: &IoDesc) -> Result<Option<Vec<i64>>> {
    if io.shape.is_empty() {
        return Err(Error::Unsupported(format!(
            "input `{}` is a scalar",
            io.name
        )));
    }
    let mut fixed = io.shape.clone();
    for (i, d) in fixed.iter_mut().enumerate() {
        if *d < 0 && i != 0 {
            return Err(Error::Unsupported(format!(
                "input `{}` dimension {i} is dynamic; only the batch may be",
                io.name
            )));
        }
    }
    if fixed[0] < 0 {
        fixed[0] = 1;
        Ok(Some(fixed))
    } else {
        Ok(None)
    }
}

/// The `TensorSpec` of one I/O tensor from the context's shape (dynamic
/// batch resolved). Refuses a non-F32 tensor and any non-positive
/// dimension, with the tensor named.
pub fn tensor_spec(name: &str, dtype: trt_sys::DType, shape: &[i64]) -> Result<TensorSpec> {
    if dtype != trt_sys::DType::F32 {
        return Err(Error::Unsupported(format!(
            "tensor `{name}` is {dtype:?}; only F32 I/O is supported (an FP16 plan keeps F32 I/O)"
        )));
    }
    if shape.is_empty() || shape.iter().any(|d| *d <= 0) {
        return Err(Error::Unsupported(format!(
            "tensor `{name}` has no fixed shape ({shape:?}); dynamic shapes are not supported"
        )));
    }
    let dims = shape.iter().map(|d| *d as usize).collect();
    Ok(TensorSpec::new(name, dims, DType::F32))
}

/// What `infer` does next about the CUDA graph (T10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferAction {
    /// `enqueueV3`, then synchronize.
    Enqueue,
    /// Enqueue and synchronize; the first good inference is the warm-up
    /// pass, so only now capture the graph.
    EnqueueThenCapture,
    /// Launch the captured graph.
    Replay,
}

/// The graph fallback state machine: captured after the first good
/// inference, replayed after, and once a capture fails, never retried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphPolicy {
    Disabled,
    Pending,
    Captured,
    Failed,
}

impl GraphPolicy {
    pub fn new(enabled: bool) -> Self {
        if enabled {
            Self::Pending
        } else {
            Self::Disabled
        }
    }

    pub fn action(self) -> InferAction {
        match self {
            Self::Pending => InferAction::EnqueueThenCapture,
            Self::Captured => InferAction::Replay,
            Self::Disabled | Self::Failed => InferAction::Enqueue,
        }
    }

    /// Records the capture attempt that `EnqueueThenCapture` asked for.
    pub fn capture_done(self, ok: bool) -> Self {
        match (self, ok) {
            (Self::Pending, true) => Self::Captured,
            (Self::Pending, false) => Self::Failed,
            (other, _) => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io(name: &str, mode: IoMode, dtype: trt_sys::DType, shape: &[i64]) -> IoDesc {
        IoDesc {
            name: name.into(),
            mode,
            dtype,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn model_source_by_extension() {
        let onnx = ModelSource::from_path(Path::new("/m/yolox.ONNX")).expect("onnx");
        assert_eq!(onnx, ModelSource::Onnx("/m/yolox.ONNX".into()));
        for p in ["/m/a.engine", "/m/a.plan"] {
            assert_eq!(
                ModelSource::from_path(Path::new(p)).expect("plan"),
                ModelSource::Plan(p.into())
            );
        }
        let err = ModelSource::from_path(Path::new("/m/a.pt")).expect_err("unknown");
        assert!(err.to_string().contains("model.path"), "{err}");
        assert!(ModelSource::from_path(Path::new("/m/noext")).is_err());
    }

    #[test]
    fn cache_dir_env_wins_when_not_empty() {
        assert_eq!(cache_dir(Some("/x")), PathBuf::from("/x"));
        assert_eq!(cache_dir(Some("")), PathBuf::from("/var/cache/edge-vision"));
        assert_eq!(cache_dir(None), PathBuf::from("/var/cache/edge-vision"));
    }

    #[test]
    fn int8_is_refused_with_the_reason() {
        let err = builder_flags(Precision::Int8).expect_err("int8");
        assert!(err.to_string().contains("calibration"), "{err}");
        assert_eq!(
            builder_flags(Precision::Fp16).expect("fp16"),
            builder_flags(Precision::Fp32).expect("fp32")
        );
    }

    #[test]
    fn timing_cache_is_per_trt_version() {
        assert_eq!(
            timing_cache_path(Path::new("/c"), 10162),
            PathBuf::from("/c/timing.trt10162.cache")
        );
    }

    #[test]
    fn l4t_line_or_unknown() {
        let dir = std::env::temp_dir().join(format!("edge-l4t-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("etc")).expect("mkdir");
        assert_eq!(read_l4t(&dir), "unknown");
        std::fs::write(
            dir.join("etc/nv_tegra_release"),
            "# R39 (release), REVISION: 2.1\nBOARD: x\n",
        )
        .expect("write");
        assert_eq!(read_l4t(&dir), "# R39 (release), REVISION: 2.1");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn io_set_needs_one_input_and_an_output() {
        let i = io("in", IoMode::Input, trt_sys::DType::F32, &[1, 3, 8, 8]);
        let o = io("out", IoMode::Output, trt_sys::DType::F32, &[1, 4]);
        assert!(check_io_set(&[i.clone(), o.clone()]).is_ok());
        assert!(check_io_set(std::slice::from_ref(&i)).is_err());
        assert!(check_io_set(std::slice::from_ref(&o)).is_err());
        assert!(check_io_set(&[i.clone(), i, o]).is_err());
    }

    #[test]
    fn dynamic_batch_is_pinned_to_one() {
        let dynamic = io("in", IoMode::Input, trt_sys::DType::F32, &[-1, 3, 416, 416]);
        assert_eq!(
            input_shape_fix(&dynamic).expect("fix"),
            Some(vec![1, 3, 416, 416])
        );
        let fixed = io("in", IoMode::Input, trt_sys::DType::F32, &[1, 3, 416, 416]);
        assert_eq!(input_shape_fix(&fixed).expect("static"), None);
        let bad = io("in", IoMode::Input, trt_sys::DType::F32, &[-1, 3, -1, 416]);
        let err = input_shape_fix(&bad).expect_err("dynamic height");
        assert!(err.to_string().contains("dimension 2"), "{err}");
        assert!(input_shape_fix(&io("s", IoMode::Input, trt_sys::DType::F32, &[])).is_err());
    }

    #[test]
    fn tensor_spec_refuses_non_f32_and_dynamic_shapes() {
        let spec = tensor_spec("o1", trt_sys::DType::F32, &[1, 2730, 4]).expect("spec");
        assert_eq!(spec.shape, vec![1, 2730, 4]);
        assert_eq!(spec.dtype, DType::F32);
        assert_eq!(spec.byte_len(), 2730 * 4 * 4);
        let err = tensor_spec("o1", trt_sys::DType::F16, &[1, 4]).expect_err("f16");
        assert!(
            err.to_string().contains("o1") && err.to_string().contains("F16"),
            "{err}"
        );
        assert!(tensor_spec("o", trt_sys::DType::F32, &[1, -1]).is_err());
        assert!(tensor_spec("o", trt_sys::DType::F32, &[0, 4]).is_err());
        assert!(tensor_spec("o", trt_sys::DType::F32, &[]).is_err());
        assert!(tensor_spec("o", trt_sys::DType::I32, &[1, 4]).is_err());
    }

    #[test]
    fn graph_is_captured_once_after_the_first_inference() {
        let mut p = GraphPolicy::new(true);
        assert_eq!(p.action(), InferAction::EnqueueThenCapture);
        p = p.capture_done(true);
        assert_eq!(p, GraphPolicy::Captured);
        assert_eq!(p.action(), InferAction::Replay);
        // a later report cannot undo a capture
        assert_eq!(p.capture_done(false), GraphPolicy::Captured);
    }

    #[test]
    fn failed_capture_falls_back_to_enqueue_for_good() {
        let p = GraphPolicy::new(true).capture_done(false);
        assert_eq!(p, GraphPolicy::Failed);
        assert_eq!(p.action(), InferAction::Enqueue);
        assert_eq!(p.capture_done(true), GraphPolicy::Failed);
        assert_eq!(GraphPolicy::new(false).action(), InferAction::Enqueue);
    }
}
