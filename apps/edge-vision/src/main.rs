//! Edge-vision binary: parse config, pick the platform, run the
//! pipeline, report metrics. SIGINT/SIGTERM ask the pipeline to
//! stop; a second signal exits immediately.

#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use edge_core::{Config, Detections, EncodedPacket, Platform, PostProcessor, Sink};
use edge_pipeline::Pipeline;
use platform_sim::{SimPlatform, SimPostProcessor};
use std::path::PathBuf;

/// Prints detections for the first few frames, then just counts.
struct StdoutSink {
    verbose_frames: u64,
}

impl Sink for StdoutSink {
    fn push_video(&mut self, pkt: EncodedPacket) -> edge_core::Result<()> {
        println!(
            "video packet: {} bytes keyframe={}",
            pkt.data.len(),
            pkt.keyframe
        );
        Ok(())
    }

    fn push_meta(&mut self, det: &Detections) -> edge_core::Result<()> {
        if det.frame_seq < self.verbose_frames {
            for d in &det.items {
                println!(
                    "frame {} detection class={} score={:.2} box=({:.2},{:.2} {:.2}x{:.2})",
                    det.frame_seq, d.class_id, d.score, d.x, d.y, d.width, d.height
                );
            }
        }
        Ok(())
    }
}

fn parse_args() -> Result<(PathBuf, Option<u64>, bool)> {
    let mut config = PathBuf::from("config/x86-dev.toml");
    let mut frames: Option<u64> = None;
    let mut prometheus = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => {
                let value = args.next().context("--frames needs a value")?;
                frames = Some(value.parse().context("--frames must be a number")?);
            }
            "--prometheus" => prometheus = true,
            "-h" | "--help" => {
                println!("usage: edge-vision [CONFIG.toml] [--frames N] [--prometheus]");
                println!("  --frames N   stop after N frames are *captured*;");
                println!("               with drops under load, fewer are processed");
                std::process::exit(0);
            }
            _ if arg.starts_with("--frames=") => {
                frames = Some(
                    arg.trim_start_matches("--frames=")
                        .parse()
                        .context("--frames must be a number")?,
                );
            }
            _ => config = PathBuf::from(arg),
        }
    }
    Ok((config, frames, prometheus))
}

fn main() -> Result<()> {
    let (config_path, frames, prometheus) = parse_args()?;
    let config =
        Config::load(&config_path).with_context(|| format!("loading {}", config_path.display()))?;

    let platform: Box<dyn Platform> = match config.platform.as_str() {
        "sim" => Box::new(SimPlatform),
        "jetson" => {
            #[cfg(feature = "jetson")]
            {
                Box::new(platform_jetson::JetsonPlatform::new(std::path::Path::new(
                    "/",
                )))
            }
            #[cfg(not(feature = "jetson"))]
            {
                bail!(
                    "platform `jetson` needs a build with --features jetson \
                     (probe and launch strings live in platform-jetson; the \
                     device runtime lands in M1); use platform = \"sim\" on x86"
                )
            }
        }
        other => {
            bail!("unknown platform `{other}` (known: sim, jetson; jetson needs --features jetson)")
        }
    };

    let postprocessor: Box<dyn PostProcessor> = match config.model.decoder {
        edge_core::DecoderKind::Yolox => Box::new(
            edge_postproc::YoloxPostProcessor::new(
                config.model.input_width,
                config.model.input_height,
                config.model.num_classes,
                config.model.score_threshold,
            )
            .with_iou_threshold(config.model.iou_threshold)
            .with_max_nms(config.model.max_nms)
            .with_top_k(config.model.top_k),
        ),
        edge_core::DecoderKind::Yolo => Box::new(
            edge_postproc::YoloPostProcessor::new(
                config.model.input_width,
                config.model.input_height,
                config.model.num_classes,
                config.model.score_threshold,
            )
            .with_iou_threshold(config.model.iou_threshold)
            .with_max_nms(config.model.max_nms)
            .with_top_k(config.model.top_k),
        ),
        edge_core::DecoderKind::Sim => {
            Box::new(SimPostProcessor::new(config.model.score_threshold))
        }
    };
    let sink: Box<dyn Sink> = match &config.stream.meta_file {
        Some(path) => Box::new(edge_sinks::MultiSink::new(vec![
            Box::new(StdoutSink { verbose_frames: 3 }) as Box<dyn Sink>,
            Box::new(edge_sinks::JsonlMetaSink::create(path)?) as Box<dyn Sink>,
        ])),
        None => Box::new(StdoutSink { verbose_frames: 3 }),
    };
    let mut pipeline = Pipeline::build(&*platform, config, postprocessor, sink)?;

    println!("platform: {}", platform.name());
    println!("{}", pipeline.buffer_budget());
    let stop = pipeline.stop_handle();
    ctrlc::set_handler(move || {
        use std::sync::atomic::Ordering;
        // First signal: stop cleanly. Second: teardown is stuck
        // (a hung stop(), a blocked sink) — exit now instead of
        // leaving SIGKILL as the only way out. 130 = 128 + SIGINT.
        if stop.swap(true, Ordering::Relaxed) {
            std::process::exit(130);
        }
    })
    .context("installing signal handler")?;
    // With --prometheus, also serve the counters while the run is
    // in progress (stderr, so stdout keeps the final exposition).
    let reporter = if prometheus {
        let live = pipeline.live_metrics();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_flag = std::sync::Arc::clone(&done);
        let handle = std::thread::Builder::new()
            .name("metrics-reporter".into())
            .spawn(move || {
                // Sleep in slices so shutdown is not delayed by
                // up to a whole reporting interval.
                let mut slept = std::time::Duration::ZERO;
                while !done_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    slept += std::time::Duration::from_millis(100);
                    if slept >= std::time::Duration::from_secs(5) {
                        slept = std::time::Duration::ZERO;
                        eprintln!("[live] {}", live.snapshot().summary());
                    }
                }
            })
            .context("spawning metrics reporter")?;
        Some((done, handle))
    } else {
        None
    };
    let (metrics, result) = pipeline.run(frames);
    if let Some((done, handle)) = reporter {
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = handle.join();
    }
    println!("done: {}", metrics.summary());
    if prometheus {
        print!("{}", metrics.render_prometheus());
    }
    result?;
    Ok(())
}
