//! Edge-vision binary: parse config, pick the platform, run the
//! pipeline, report metrics. SIGINT/SIGTERM ask the pipeline to
//! stop; a second signal exits immediately.

#![forbid(unsafe_code)]

mod recovery;

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

/// The web UI's counters, from the same numbers as the `[live]` line.
fn stats_json(m: &edge_pipeline::PipelineMetrics, drops: &edge_sinks::AsyncDrops) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let ms = |d: Option<std::time::Duration>| {
        d.map_or_else(
            || "null".to_string(),
            |d| format!("{:.1}", d.as_secs_f64() * 1000.0),
        )
    };
    let encode_push_ms = if m.frames_processed == 0 {
        0.0
    } else {
        m.encode_push_total.as_secs_f64() * 1000.0 / m.frames_processed as f64
    };
    format!(
        "{{\"type\":\"stats\",\"captured\":{},\"processed\":{},\"dropped\":{},\
         \"enc_dropped\":{},\"timeouts\":{},\"detections\":{},\"p50_ms\":{},\"p95_ms\":{},\
         \"p99_ms\":{},\"encode_push_ms\":{:.2},\"sink_drop_video\":{},\"sink_drop_meta\":{}}}",
        m.frames_captured,
        m.frames_processed,
        m.frames_dropped,
        m.encode_dropped,
        m.capture_timeouts,
        m.detections,
        ms(m.latency.p50()),
        ms(m.latency.p95()),
        ms(m.latency.p99()),
        encode_push_ms,
        drops.video.load(Relaxed),
        drops.meta.load(Relaxed),
    )
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
                let platform = platform_jetson::JetsonPlatform::new(std::path::Path::new("/"));
                println!("module: {:?}", platform.module_identity());
                Box::new(platform)
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
    let mut sinks: Vec<Box<dyn Sink>> = vec![Box::new(StdoutSink { verbose_frames: 3 })];
    if let Some(path) = &config.stream.meta_file {
        sinks.push(Box::new(edge_sinks::JsonlMetaSink::create(path)?));
    }
    if let Some(addr) = config.stream.video_tcp {
        sinks
            .push(Box::new(edge_sinks::TcpH264Sink::bind(addr).with_context(
                || format!("binding stream.video_tcp {addr}"),
            )?));
        println!("video: raw H.264 on tcp://{addr}");
    }
    let web_ui = match config.stream.web_ui {
        Some(addr) => {
            let ui = edge_sinks::WebUi::bind(addr)
                .with_context(|| format!("binding stream.web_ui {addr}"))?;
            sinks.push(Box::new(ui.sink()));
            println!("video: web UI at http://{}/", ui.local_addr());
            Some(std::sync::Arc::new(ui))
        }
        None => None,
    };
    if let Some(port) = config.stream.rtsp_port {
        #[cfg(feature = "rtsp")]
        {
            let rtsp = edge_sinks::rtsp::RtspSink::new(port, config.encoder.codec)
                .with_context(|| format!("starting RTSP on port {port}"))?;
            println!("video: RTSP at {}", rtsp.url());
            sinks.push(Box::new(rtsp));
        }
        #[cfg(not(feature = "rtsp"))]
        {
            let _ = port;
            bail!("stream.rtsp_port needs a build with --features rtsp");
        }
    }
    let inner: Box<dyn Sink> = if sinks.len() == 1 {
        sinks.remove(0)
    } else {
        Box::new(edge_sinks::MultiSink::new(sinks))
    };
    // Disk/terminal writes happen on a bounded writer thread, never on
    // the frame-processing thread (N20).
    let async_sink = edge_sinks::AsyncSink::new(inner, 64);
    let drops = async_sink.drops();
    let sink: Box<dyn Sink> = Box::new(async_sink);
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
    // With --prometheus, also print the counters while the run is in
    // progress (stderr, so stdout keeps the final exposition); the web
    // UI gets them about once per second.
    let reporter = if prometheus || web_ui.is_some() {
        let live = pipeline.live_metrics();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done_flag = std::sync::Arc::clone(&done);
        let ui = web_ui.clone();
        let drops = drops.clone();
        let handle = std::thread::Builder::new()
            .name("metrics-reporter".into())
            .spawn(move || {
                // Sleep in slices so shutdown is not delayed by
                // up to a whole reporting interval.
                let mut ticks = 0u32;
                while !done_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    ticks += 1;
                    if ticks.is_multiple_of(10) {
                        if let Some(ui) = &ui {
                            ui.publish_stats(&stats_json(&live.snapshot(), &drops));
                        }
                    }
                    if prometheus && ticks.is_multiple_of(50) {
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
    {
        use std::sync::atomic::Ordering::Relaxed;
        println!(
            "sink drops: video={} meta={}",
            drops.video.load(Relaxed),
            drops.meta.load(Relaxed)
        );
    }
    if prometheus {
        print!("{}", metrics.render_prometheus());
    }

    if let Err(edge_core::Error::CaptureServiceLost(reason)) = &result {
        match recovery::CaptureRecoveryPolicy::from_env().on_capture_loss() {
            recovery::RecoveryAction::RestartDaemonAndExit75 => {
                eprintln!(
                    "capture service lost ({reason}); queuing nvargus-daemon restart, exit 75"
                );
                if let Err(e) = recovery::request_daemon_restart() {
                    eprintln!("warning: could not restart nvargus-daemon: {e}");
                }
                std::process::exit(75);
            }
            recovery::RecoveryAction::LogAndFail => {
                eprintln!(
                    "capture service lost again ({reason}) after one recovery this boot; failing"
                );
                std::process::exit(1);
            }
        }
    }
    result?;
    Ok(())
}
