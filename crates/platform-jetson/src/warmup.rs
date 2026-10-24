//! Engine warm-up: load or build the plan on the `engine-warm` thread.
//! Role: Controller of the background preparation (D5, D6). `engine()`
//! returns `EngineState::Warming` at once, so video never waits for a
//! plan that takes minutes to build; a failure surfaces from the poll.
//!
//! Order: a prebuilt `.engine`/`.plan` loads as it is. An `.onnx` model
//! looks its plan up in the cache by `plan_cache::plan_key`; a miss
//! builds it in-process (progress in the log), writes it atomically,
//! prunes older plans of the model, then loads it. A cached plan that
//! does not load is rebuilt once. The whole preparation runs under the
//! process-wide build lock (`engine_job::build_lock`).

use crate::build_progress::BuildProgress;
use crate::cuda::platform_err;
use crate::engine_job::{build_lock, BackgroundJob, THREAD_NAME};
use crate::plan_cache::{model_id, plan_key, PlanCache, PlanKeyInputs};
use crate::trt::TrtEngine;
use crate::trt_logic::{
    builder_flags, cache_dir, read_l4t, timing_cache_path, ModelSource, CACHE_ENV, WORKSPACE_BYTES,
};
use crate::JetsonPlatform;
use edge_core::{
    EngineState, EngineWarmup, Error, InferenceEngine, ModelConfig, Precision, Result,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Everything the warm-up thread needs; plain data, built on the caller's thread.
#[derive(Debug, Clone)]
pub struct WarmupRequest {
    pub source: ModelSource,
    pub precision: Precision,
    pub cache_dir: PathBuf,
    /// `/proc/device-tree/compatible` of the module (plan key).
    pub soc: String,
    /// First line of `/etc/nv_tegra_release` (plan key).
    pub l4t: String,
    pub use_graph: bool,
}

/// The `EngineWarmup` the runner polls once per frame.
pub struct TrtWarmup {
    job: BackgroundJob<Box<dyn InferenceEngine>>,
}

impl TrtWarmup {
    pub fn spawn(request: WarmupRequest) -> Result<Self> {
        let job = BackgroundJob::spawn(THREAD_NAME, move |cancel| prepare(&request, &cancel))?;
        Ok(Self { job })
    }
}

impl EngineWarmup for TrtWarmup {
    fn poll_ready(&mut self) -> Result<Option<Box<dyn InferenceEngine>>> {
        self.job.poll()
    }
}

impl JetsonPlatform {
    /// `Platform::engine` under the `trt` feature.
    pub(crate) fn trt_engine(&self, cfg: &ModelConfig) -> Result<EngineState> {
        let path = cfg.path.as_ref().ok_or_else(|| {
            Error::invalid(
                "model.path",
                "required: an .onnx model or a prebuilt .engine/.plan",
            )
        })?;
        match cfg.device.as_deref() {
            None | Some("gpu") => {}
            Some(other) => {
                return Err(Error::Unsupported(format!(
                    "model.device `{other}`: only \"gpu\" runs on JetPack 7"
                )))
            }
        }
        builder_flags(cfg.precision)?; // refuses int8 before a thread starts
        let request = WarmupRequest {
            source: ModelSource::from_path(path)?,
            precision: cfg.precision,
            cache_dir: cache_dir(std::env::var(CACHE_ENV).ok().as_deref()),
            soc: self
                .module
                .compatible
                .clone()
                .unwrap_or_else(|| "unknown".into()),
            l4t: read_l4t(&self.root),
            use_graph: true,
        };
        Ok(EngineState::Warming(Box::new(TrtWarmup::spawn(request)?)))
    }
}

/// Where one model's plan lives in the cache, and what builds it.
struct OnnxPlan {
    cache: PlanCache,
    model_id: String,
    key: String,
    onnx: PathBuf,
    fp16: bool,
    timing_cache: PathBuf,
}

fn prepare(request: &WarmupRequest, cancel: &AtomicBool) -> Result<Box<dyn InferenceEngine>> {
    let runtime = trt_sys::Runtime::new().map_err(platform_err)?;
    let _lock = build_lock();
    let (plan, rebuild_with) = match &request.source {
        ModelSource::Plan(path) => (std::fs::read(path)?, None),
        ModelSource::Onnx(path) => {
            let location = locate(request, path)?;
            let (bytes, from_cache) = load_or_build(&location, cancel)?;
            (bytes, from_cache.then_some(location))
        }
    };
    match TrtEngine::from_plan(&runtime, &plan, request.use_graph) {
        Ok(engine) => Ok(Box::new(engine)),
        Err(e) => match rebuild_with {
            Some(location) => {
                log::warn!("tensorrt: the cached plan does not load ({e}); rebuilding it");
                let fresh = build_and_store(&location, cancel)?;
                Ok(Box::new(TrtEngine::from_plan(
                    &runtime,
                    &fresh,
                    request.use_graph,
                )?))
            }
            None => Err(e),
        },
    }
}

fn locate(request: &WarmupRequest, onnx: &Path) -> Result<OnnxPlan> {
    let bytes = std::fs::read(onnx)
        .map_err(|e| Error::Platform(format!("cannot read {}: {e}", onnx.display())))?;
    let device = trt_sys::device_info().map_err(platform_err)?;
    let version = trt_sys::version();
    let flags = builder_flags(request.precision)?;
    let key = plan_key(&PlanKeyInputs {
        onnx_bytes: &bytes,
        trt_version: u32::try_from(version).unwrap_or(0),
        precision: request.precision,
        sm: device.sm,
        soc: &request.soc,
        l4t: &request.l4t,
        cuda: device.cuda_runtime,
        gpu_name: &device.name,
        builder_flags: &flags,
    });
    Ok(OnnxPlan {
        cache: PlanCache::new(&request.cache_dir),
        model_id: model_id(&bytes),
        key,
        onnx: onnx.to_path_buf(),
        fp16: request.precision == Precision::Fp16,
        timing_cache: timing_cache_path(&request.cache_dir, version),
    })
}

/// The plan bytes, and whether they came from the cache.
fn load_or_build(plan: &OnnxPlan, cancel: &AtomicBool) -> Result<(Vec<u8>, bool)> {
    if let Some(bytes) = plan.cache.load(&plan.model_id, &plan.key) {
        log::info!(
            "tensorrt: plan for {} loaded from {}",
            plan.onnx.display(),
            plan.cache.path_for(&plan.model_id, &plan.key).display()
        );
        return Ok((bytes, true));
    }
    Ok((build_and_store(plan, cancel)?, false))
}

fn build_and_store(plan: &OnnxPlan, cancel: &AtomicBool) -> Result<Vec<u8>> {
    log::info!(
        "tensorrt: building a {} plan for {} (minutes, once; cached in {})",
        if plan.fp16 { "fp16" } else { "fp32" },
        plan.onnx.display(),
        plan.cache.model_dir(&plan.model_id).display()
    );
    // The timing cache sits next to the plans; best effort if the folder is read-only.
    if let Some(dir) = plan.timing_cache.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let options = trt_sys::BuildOptions {
        fp16: plan.fp16,
        workspace_bytes: WORKSPACE_BYTES,
        timing_cache: Some(plan.timing_cache.clone()),
    };
    let (started, mut progress) = (Instant::now(), BuildProgress::default());
    let built = trt_sys::build_plan(&plan.onnx, &options, &mut |event| {
        if let Some(line) = progress.line(event, started.elapsed()) {
            log::log!(line.level, "{}", line.text);
        }
        !cancel.load(Ordering::Acquire)
    })
    .map_err(platform_err)?;
    let bytes = built.as_bytes().to_vec();
    log::info!(
        "tensorrt: plan built in {:.0} s, {} bytes",
        started.elapsed().as_secs_f64(),
        bytes.len()
    );
    match plan.cache.store(&plan.model_id, &plan.key, &bytes) {
        Ok(()) => {
            if let Err(e) = plan.cache.prune_except(&plan.model_id, &plan.key) {
                log::warn!("tensorrt: cannot prune old plans: {e}");
            }
        }
        Err(e) => log::warn!("tensorrt: cannot cache the plan ({e}); the next start rebuilds it"),
    }
    Ok(bytes)
}
