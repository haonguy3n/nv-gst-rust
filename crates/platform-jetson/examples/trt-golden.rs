//! Golden-set runner (T11): the PC's letterboxed input tensors through the app's TensorRT path.
//!
//! The plan is prepared by `TrtWarmup` exactly as the app does (cache, FP16 build, CUDA graph),
//! each `<in>/input/<id>.bin` (raw f32 LE, NCHW) is copied into the input binding, and every
//! output is written to `<out>/trt/<id>.<output>.bin`. Compare on the PC with
//! `tools/golden/compare.py`; the procedure is in `tools/golden/README.md`.
//!
//!     cargo run --release -p platform-jetson --features trt --example trt-golden -- \
//!         --onnx /opt/edge-vision/models/yolox-s.onnx --cache /var/cache/edge-vision \
//!         --in /data/golden --out /data/golden

use edge_core::{EngineWarmup, Error, GpuStream, InferenceEngine, Precision, Result};
use platform_jetson::cuda::new_gpu_stream;
use platform_jetson::probe::probe_module_identity;
use platform_jetson::trt_logic::{read_l4t, ModelSource};
use platform_jetson::warmup::{TrtWarmup, WarmupRequest};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

struct Args {
    onnx: PathBuf,
    cache: PathBuf,
    input: PathBuf,
    out: PathBuf,
    limit: Option<usize>,
}

fn parse_args() -> std::result::Result<Args, String> {
    let (mut onnx, mut cache, mut input, mut out, mut limit) = (None, None, None, None, None);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--onnx" => onnx = Some(PathBuf::from(value)),
            "--cache" => cache = Some(PathBuf::from(value)),
            "--in" => input = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--limit" => limit = Some(value.parse().map_err(|_| "--limit: not a number")?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let need = |v: Option<PathBuf>, n: &str| v.ok_or(format!("{n} is required"));
    Ok(Args {
        onnx: need(onnx, "--onnx")?,
        cache: need(cache, "--cache")?,
        input: need(input, "--in")?,
        out: need(out, "--out")?,
        limit,
    })
}

fn err(what: impl std::fmt::Display) -> Error {
    Error::Platform(what.to_string())
}

/// The engine as the app gets it: built or loaded by the warm-up thread.
fn load_engine(args: &Args) -> Result<Box<dyn InferenceEngine>> {
    let root = Path::new("/");
    let request = WarmupRequest {
        source: ModelSource::from_path(&args.onnx)?,
        precision: Precision::Fp16,
        cache_dir: args.cache.clone(),
        soc: probe_module_identity(root)
            .compatible
            .unwrap_or_else(|| "unknown".into()),
        l4t: read_l4t(root),
        use_graph: true,
    };
    let mut warmup = TrtWarmup::spawn(request)?;
    let started = Instant::now();
    loop {
        if let Some(engine) = warmup.poll_ready()? {
            eprintln!(
                "engine ready after {:.1} s",
                started.elapsed().as_secs_f64()
            );
            return Ok(engine);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Input file stems (the image ids), sorted by number.
fn input_ids(dir: &Path) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    for entry in std::fs::read_dir(dir.join("input"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "bin") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                ids.push(stem.to_string());
            }
        }
    }
    ids.sort_by_key(|s| s.parse::<u64>().unwrap_or(u64::MAX));
    Ok(ids)
}

fn fill_input(engine: &mut dyn InferenceEngine, path: &Path) -> Result<()> {
    let bytes = std::fs::read(path)?;
    let input = engine
        .input_bindings()
        .first_mut()
        .ok_or_else(|| err("the engine has no input"))?;
    let dst = input
        .as_f32_slice_mut()
        .ok_or_else(|| err("the input binding is not host-visible"))?;
    if bytes.len() != dst.len() * 4 {
        return Err(err(format!(
            "{}: {} bytes, the input binding takes {}",
            path.display(),
            bytes.len(),
            dst.len() * 4
        )));
    }
    let (words, _) = bytes.as_chunks::<4>();
    for (value, word) in dst.iter_mut().zip(words) {
        *value = f32::from_le_bytes(*word);
    }
    Ok(())
}

fn write_outputs(engine: &dyn InferenceEngine, dir: &Path, id: &str) -> Result<()> {
    for tensor in engine.output_bindings() {
        let data = tensor
            .as_f32_slice()
            .ok_or_else(|| err("an output binding is not host-visible"))?;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("{id}.{}.bin", tensor.spec.name)), bytes)?;
    }
    Ok(())
}

fn run(args: &Args) -> Result<()> {
    let ids = input_ids(&args.input)?;
    let ids = &ids[..args.limit.map_or(ids.len(), |n| n.min(ids.len()))];
    let first = ids.first().ok_or_else(|| err("no input/*.bin found"))?;
    let out_dir = args.out.join("trt");
    std::fs::create_dir_all(&out_dir)?;
    let mut engine = load_engine(args)?;
    let stream: GpuStream = new_gpu_stream()?;
    // The first inference captures the CUDA graph; run it once so every recorded one replays it.
    fill_input(
        engine.as_mut(),
        &args.input.join(format!("input/{first}.bin")),
    )?;
    engine.infer(&stream)?;
    let mut times = Vec::with_capacity(ids.len());
    for id in ids {
        fill_input(engine.as_mut(), &args.input.join(format!("input/{id}.bin")))?;
        let started = Instant::now();
        engine.infer(&stream)?;
        let ms = started.elapsed().as_secs_f64() * 1e3;
        write_outputs(engine.as_ref(), &out_dir, id)?;
        println!("{id}: infer {ms:.2} ms");
        times.push(ms);
    }
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "{} images: mean {mean:.2} ms, median {:.2} ms, max {:.2} ms; outputs in {}",
        times.len(),
        times[times.len() / 2],
        times[times.len() - 1],
        out_dir.display()
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "{e}\nusage: trt-golden --onnx <model.onnx> --cache <dir> --in <golden dir> --out <dir> [--limit N]"
            );
            return std::process::ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("trt-golden: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
