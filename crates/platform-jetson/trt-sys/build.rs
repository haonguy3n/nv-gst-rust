//! Compiles `shim/trt_shim.cpp` and links TensorRT, the ONNX parser and
//! the CUDA runtime, only with the `shim` feature.
//!
//! Environment (all optional; each one is a `rerun-if-env-changed`):
//! - `TRT_SYS_INCLUDE`: directory of `NvInfer.h` and `NvOnnxParser.h`
//!   (default: `<sysroot>/usr/include`, then the compiler's `/usr/include`).
//! - `TRT_SYS_CUDA_INCLUDE`: directory of `cuda_runtime_api.h` (default:
//!   `<sysroot>/usr/local/cuda*/include`, else `/usr/local/cuda*/include`).
//! - `TRT_SYS_LIB`: directory of `libnvinfer.so` and `libnvonnxparser.so`
//!   (default: `<sysroot>/usr/lib` when a sysroot is set).
//! - `TRT_SYS_CUDA_LIB`: directory of `libcudart.so` (default: like the
//!   CUDA include directory, `lib64` then `lib`).
//! - `SYSROOT` or `PKG_CONFIG_SYSROOT_DIR`: the target sysroot.
//! - `TRT_SYS_SKIP_SHIM=1`: type-check the Rust side on a PC without the
//!   headers; the result cannot link.

use std::env;
use std::path::{Path, PathBuf};

const ENV_VARS: &[&str] = &[
    "TRT_SYS_INCLUDE",
    "TRT_SYS_CUDA_INCLUDE",
    "TRT_SYS_LIB",
    "TRT_SYS_CUDA_LIB",
    "TRT_SYS_SKIP_SHIM",
    "SYSROOT",
    "PKG_CONFIG_SYSROOT_DIR",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/trt_shim.cpp");
    println!("cargo:rerun-if-changed=shim/trt_shim.h");
    for var in ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if env::var_os("CARGO_FEATURE_SHIM").is_none() {
        return;
    }
    if env::var_os("TRT_SYS_SKIP_SHIM").is_some() {
        println!("cargo:warning=trt-sys: TRT_SYS_SKIP_SHIM is set: the C++ shim is neither compiled nor linked");
        return;
    }
    let sysroot = env::var_os("SYSROOT")
        .or_else(|| env::var_os("PKG_CONFIG_SYSROOT_DIR"))
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty());
    let trt_include = find_trt_include(sysroot.as_deref());
    let cuda_include = find_cuda_include(sysroot.as_deref());
    compile(trt_include.as_deref(), &cuda_include);
    link(sysroot.as_deref(), &cuda_include);
}

fn fail(message: &str) -> ! {
    eprintln!("error: trt-sys: {message}");
    panic!("trt-sys: {message}");
}

fn rooted(sysroot: Option<&Path>, path: &str) -> PathBuf {
    match sysroot {
        Some(root) => root.join(path.trim_start_matches('/')),
        None => PathBuf::from(path),
    }
}

/// `Some(dir)` when the headers are in a directory the compiler needs
/// told about (`TRT_SYS_INCLUDE`); `None` when they are in the sysroot's
/// or the system's `usr/include`, which the compiler searches itself
/// (the Yocto SDK's compiler carries `--sysroot`; `-isystem` on a
/// standard directory would break `#include_next` in libstdc++).
fn find_trt_include(sysroot: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = env::var_os("TRT_SYS_INCLUDE").map(PathBuf::from) {
        if dir.join("NvInfer.h").is_file() && dir.join("NvOnnxParser.h").is_file() {
            return Some(dir);
        }
        fail(&format!(
            "TRT_SYS_INCLUDE={} has no NvInfer.h / NvOnnxParser.h",
            dir.display()
        ));
    }
    let mut candidates = vec![rooted(sysroot, "/usr/include")];
    if sysroot.is_none() {
        candidates.push(PathBuf::from("/usr/include/aarch64-linux-gnu"));
    }
    for dir in candidates {
        if dir.join("NvInfer.h").is_file() && dir.join("NvOnnxParser.h").is_file() {
            return (dir.ends_with("aarch64-linux-gnu")).then_some(dir);
        }
    }
    fail(
        "TensorRT headers (NvInfer.h, NvOnnxParser.h) not found. Set TRT_SYS_INCLUDE to their \
         directory, or SYSROOT to the target sysroot (the Yocto SDK's \
         $SDKTARGETSYSROOT). The `trt` feature only builds against the Jetson image's \
         TensorRT 10.16 / CUDA 13.2 development files; on a PC without them build \
         without `trt`, or set TRT_SYS_SKIP_SHIM=1 to type-check the Rust side only",
    )
}

/// `<root>/usr/local/cuda`, else the newest `cuda-*` next to it.
fn cuda_home(sysroot: Option<&Path>) -> Option<PathBuf> {
    let base = rooted(sysroot, "/usr/local");
    let plain = base.join("cuda");
    if plain.join("include").is_dir() {
        return Some(plain);
    }
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&base)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("cuda-"))
                && p.join("include").is_dir()
        })
        .collect();
    versions.sort();
    versions.pop()
}

fn find_cuda_include(sysroot: Option<&Path>) -> PathBuf {
    if let Some(dir) = env::var_os("TRT_SYS_CUDA_INCLUDE").map(PathBuf::from) {
        if dir.join("cuda_runtime_api.h").is_file() {
            return dir;
        }
        fail(&format!(
            "TRT_SYS_CUDA_INCLUDE={} has no cuda_runtime_api.h",
            dir.display()
        ));
    }
    if let Some(home) = cuda_home(sysroot) {
        let dir = home.join("include");
        if dir.join("cuda_runtime_api.h").is_file() {
            return dir;
        }
    }
    fail(
        "CUDA headers (cuda_runtime_api.h) not found. Set TRT_SYS_CUDA_INCLUDE to the CUDA \
         include directory (for example <sysroot>/usr/local/cuda-13.2/include), or SYSROOT to \
         the target sysroot",
    )
}

fn compile(trt_include: Option<&Path>, cuda_include: &Path) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("shim/trt_shim.cpp")
        .warnings(true)
        .include("shim")
        // -isystem: NVIDIA's headers carry deprecation warnings that are not ours.
        .flag("-isystem")
        .flag(cuda_include.to_string_lossy().as_ref());
    if let Some(dir) = trt_include {
        build.flag("-isystem").flag(dir.to_string_lossy().as_ref());
    }
    build.compile("trt_shim");
}

fn link(sysroot: Option<&Path>, cuda_include: &Path) {
    let trt_lib = env::var_os("TRT_SYS_LIB")
        .map(PathBuf::from)
        .or_else(|| sysroot.map(|root| root.join("usr/lib")));
    if let Some(dir) = trt_lib {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    let cuda_lib = env::var_os("TRT_SYS_CUDA_LIB")
        .map(PathBuf::from)
        .or_else(|| {
            let home = cuda_include.parent()?;
            ["lib64", "lib"]
                .iter()
                .map(|d| home.join(d))
                .find(|d| d.join("libcudart.so").exists())
        });
    if let Some(dir) = cuda_lib {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=nvonnxparser");
    println!("cargo:rustc-link-lib=cudart");
}
