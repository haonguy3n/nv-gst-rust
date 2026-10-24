//! TensorRT plan cache: pure Rust, fully tested on the PC.
//! The shim (`trt-sys`, task T6) supplies built plan bytes and
//! consumes loaded ones; everything about *where* a plan lives
//! and *whether the copy on disk is intact* is decided here.
//!
//! Layout: `<cache>/<model_id>/<key>.plan`. The model namespace means
//! pruning one model's old plans cannot evict another model's.
//!
//! On-disk format per entry: an 8-byte magic, a little-endian
//! u64 payload length, the payload's SHA-256, then the plan
//! bytes. A file that fails any of those checks (truncated by a
//! power cut mid-write, corrupted, foreign) loads as missing,
//! so the caller rebuilds instead of handing TensorRT garbage.
//! Writes go to a temp file in the same directory, fsync, rename
//! over the target, then fsync the directory: a crash can leave
//! the old plan or the new one, never a torn one.

use edge_core::{Precision, Result};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Where the camera image keeps built plans (systemd
/// `CacheDirectory=edge-vision`).
pub const DEFAULT_CACHE_DIR: &str = "/var/cache/edge-vision";

const MAGIC: &[u8; 8] = b"edgeplan";
const HEADER_LEN: usize = 8 + 8 + 32;

/// Inputs that make a built plan unique. Any change (new model,
/// new TensorRT, different precision or builder flags, different
/// GPU) must produce a different key, because a plan is only
/// valid for the exact combination that built it.
#[derive(Debug, Clone)]
pub struct PlanKeyInputs<'a> {
    pub onnx_bytes: &'a [u8],
    /// `getInferLibVersion()` from the shim.
    pub trt_version: u32,
    pub precision: Precision,
    /// GPU compute capability, e.g. (8, 7) for Orin.
    pub sm: (u32, u32),
    /// Module identity from `/proc/device-tree/compatible`
    /// (e.g. `nvidia,p3767-0000`). NX and Nano share an SM count but
    /// not clocks or memory bandwidth.
    pub soc: &'a str,
    /// `/etc/nv_tegra_release` (BSP).
    pub l4t: &'a str,
    /// CUDA runtime version (`cudaRuntimeGetVersion`).
    pub cuda: u32,
    /// GPU name from `cudaGetDeviceProperties` (`Orin`, ...).
    pub gpu_name: &'a str,
    /// Serialised builder flags (workspace, DLA core, ...).
    pub builder_flags: &'a str,
}

/// Hex SHA-256 over all key inputs, with separators so fields
/// cannot bleed into each other.
pub fn plan_key(inputs: &PlanKeyInputs<'_>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(inputs.onnx_bytes);
    hasher.update(b"|trt=");
    hasher.update(inputs.trt_version.to_le_bytes());
    hasher.update(b"|precision=");
    hasher.update(inputs.precision.to_string());
    hasher.update(b"|sm=");
    hasher.update(inputs.sm.0.to_le_bytes());
    hasher.update(b".");
    hasher.update(inputs.sm.1.to_le_bytes());
    hasher.update(b"|soc=");
    hasher.update(inputs.soc);
    hasher.update(b"|l4t=");
    hasher.update(inputs.l4t);
    hasher.update(b"|cuda=");
    hasher.update(inputs.cuda.to_le_bytes());
    hasher.update(b"|gpu=");
    hasher.update(inputs.gpu_name);
    hasher.update(b"|flags=");
    hasher.update(inputs.builder_flags);
    hex(&hasher.finalize())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Stable per-model cache namespace. Plans live under
/// `<cache>/<model_id>/`, so pruning one model can never evict another
/// model's multi-minute build.
pub fn model_id(onnx_bytes: &[u8]) -> String {
    let digest = Sha256::digest(onnx_bytes);
    hex(&digest[..8])
}

pub struct PlanCache {
    dir: PathBuf,
}

impl PlanCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn default_dir() -> Self {
        Self::new(DEFAULT_CACHE_DIR)
    }

    pub fn model_dir(&self, model_id: &str) -> PathBuf {
        self.dir.join(model_id)
    }

    pub fn path_for(&self, model_id: &str, key: &str) -> PathBuf {
        self.model_dir(model_id).join(format!("{key}.plan"))
    }

    /// The cached plan bytes, or `None` when missing or not
    /// intact (both mean: rebuild).
    pub fn load(&self, model_id: &str, key: &str) -> Option<Vec<u8>> {
        let raw = std::fs::read(self.path_for(model_id, key)).ok()?;
        if raw.len() < HEADER_LEN || &raw[..8] != MAGIC {
            return None;
        }
        let len = u64::from_le_bytes(raw[8..16].try_into().ok()?) as usize;
        let payload = raw.get(HEADER_LEN..)?;
        if payload.len() != len {
            return None;
        }
        let digest = Sha256::digest(payload);
        if digest.as_slice() != &raw[16..48] {
            return None;
        }
        Some(payload.to_vec())
    }

    /// Atomically store a built plan under `key` for one model.
    pub fn store(&self, model_id: &str, key: &str, plan: &[u8]) -> Result<()> {
        let dir = self.model_dir(model_id);
        std::fs::create_dir_all(&dir)?;
        // create_new: two builders in one process (or two processes)
        // must never share a temporary path. The sequence makes even
        // same-process collisions impossible.
        let tmp = dir.join(format!(
            ".{key}.tmp-{}-{}",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let write_tmp = |path: &Path| -> std::io::Result<()> {
            let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
            file.write_all(MAGIC)?;
            file.write_all(&(plan.len() as u64).to_le_bytes())?;
            file.write_all(&Sha256::digest(plan))?;
            file.write_all(plan)?;
            file.sync_all()
        };
        if let Err(e) = write_tmp(&tmp) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
        std::fs::rename(&tmp, self.path_for(model_id, key))?;
        File::open(&dir)?.sync_all()?;
        Ok(())
    }

    /// Remove every cached plan of `model_id` except `keep`. Plans for
    /// other models are untouched. A missing model directory removes
    /// nothing.
    pub fn prune_except(&self, model_id: &str, keep: &str) -> Result<usize> {
        let keep_name = format!("{keep}.plan");
        let mut removed = 0;
        let entries = match std::fs::read_dir(self.model_dir(model_id)) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(e.into()),
        };
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".plan") && name != keep_name {
                std::fs::remove_file(entry.path())?;
                removed += 1;
            }
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cache(name: &str) -> PlanCache {
        let dir =
            std::env::temp_dir().join(format!("edge-plan-cache-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        PlanCache::new(dir)
    }

    fn inputs<'a>(onnx: &'a [u8]) -> PlanKeyInputs<'a> {
        PlanKeyInputs {
            onnx_bytes: onnx,
            trt_version: 10016,
            precision: Precision::Fp16,
            sm: (8, 7),
            soc: "nvidia,p3767-0000",
            l4t: "R39.2.1",
            cuda: 13020,
            gpu_name: "Orin",
            builder_flags: "ws=1024",
        }
    }

    #[test]
    fn key_changes_with_every_input() {
        let base = plan_key(&inputs(b"model-a"));
        assert_eq!(base.len(), 64);
        assert_ne!(base, plan_key(&inputs(b"model-b")));
        let mut other = inputs(b"model-a");
        other.trt_version = 10015;
        assert_ne!(base, plan_key(&other));
        let mut other = inputs(b"model-a");
        other.precision = Precision::Fp32;
        assert_ne!(base, plan_key(&other));
        let mut other = inputs(b"model-a");
        other.sm = (8, 9);
        assert_ne!(base, plan_key(&other));
        let mut other = inputs(b"model-a");
        other.soc = "nvidia,p3767-0003";
        assert_ne!(base, plan_key(&other));
        let mut other = inputs(b"model-a");
        other.cuda = 13021;
        assert_ne!(base, plan_key(&other));
        let mut other = inputs(b"model-a");
        other.gpu_name = "Orin Nano";
        assert_ne!(base, plan_key(&other));
        let mut other = inputs(b"model-a");
        other.builder_flags = "ws=2048";
        assert_ne!(base, plan_key(&other));
        assert_eq!(base, plan_key(&inputs(b"model-a")), "key is deterministic");
    }

    #[test]
    fn store_then_load_round_trips() {
        let cache = temp_cache("roundtrip");
        let model = model_id(b"m");
        let key = plan_key(&inputs(b"m"));
        assert_eq!(cache.load(&model, &key), None, "missing before store");
        cache.store(&model, &key, b"plan-bytes").unwrap();
        assert_eq!(
            cache.load(&model, &key).as_deref(),
            Some(b"plan-bytes".as_slice())
        );
    }

    #[test]
    fn truncated_or_corrupt_plan_loads_as_missing() {
        let cache = temp_cache("corrupt");
        let model = model_id(b"m");
        let key = plan_key(&inputs(b"m"));
        cache.store(&model, &key, b"plan-bytes").unwrap();
        let path = cache.path_for(&model, &key);

        // Truncated (power cut mid-write on a non-atomic writer).
        let raw = std::fs::read(&path).unwrap();
        std::fs::write(&path, &raw[..raw.len() - 3]).unwrap();
        assert_eq!(cache.load(&model, &key), None);

        // Corrupted payload (checksum mismatch).
        cache.store(&model, &key, b"plan-bytes").unwrap();
        let mut raw = std::fs::read(&path).unwrap();
        *raw.last_mut().unwrap() ^= 0xff;
        std::fs::write(&path, raw).unwrap();
        assert_eq!(cache.load(&model, &key), None);
    }

    #[test]
    fn prune_removes_other_keys_of_the_same_model_only() {
        let cache = temp_cache("prune");
        let model = model_id(b"model-a");
        let other_model = model_id(b"model-b");
        cache.store(&model, "aaa", b"1").unwrap();
        cache.store(&model, "bbb", b"2").unwrap();
        cache.store(&model, "ccc", b"3").unwrap();
        cache.store(&other_model, "ddd", b"4").unwrap();
        assert_eq!(cache.prune_except(&model, "bbb").unwrap(), 2);
        assert_eq!(cache.load(&model, "bbb").as_deref(), Some(b"2".as_slice()));
        assert_eq!(cache.load(&model, "aaa"), None);
        assert_eq!(
            cache.load(&other_model, "ddd").as_deref(),
            Some(b"4".as_slice()),
            "pruning one model must not evict another model's plan"
        );
        assert_eq!(
            cache.prune_except(&model, "bbb").unwrap(),
            0,
            "second prune is a no-op"
        );
    }
}
