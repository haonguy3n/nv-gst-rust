//! Edge-vision binary: parse config, pick the platform, run the
//! pipeline, report metrics. Signal handling lands with the Jetson
//! daemon work; the sim profile is bounded by `source.frames`.

#![forbid(unsafe_code)]

use anyhow::{bail, Context, Result};
use edge_core::{Config, Detections, EncodedPacket, Platform, Sink};
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

fn parse_args() -> Result<(PathBuf, Option<u64>)> {
    let mut config = PathBuf::from("config/x86-dev.toml");
    let mut frames: Option<u64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => {
                let value = args.next().context("--frames needs a value")?;
                frames = Some(value.parse().context("--frames must be a number")?);
            }
            "-h" | "--help" => {
                println!("usage: edge-vision [CONFIG.toml] [--frames N]");
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
    Ok((config, frames))
}

fn main() -> Result<()> {
    let (config_path, frames) = parse_args()?;
    let config =
        Config::load(&config_path).with_context(|| format!("loading {}", config_path.display()))?;

    let platform: Box<dyn Platform> = match config.platform.as_str() {
        "sim" => Box::new(SimPlatform),
        "jetson" => bail!(
            "platform `jetson` is not in this build: the Jetson backend \
             (direct Argus capture) lands in M1; use platform = \"sim\" on x86"
        ),
        other => bail!("unknown platform `{other}` (known: sim)"),
    };

    let threshold = config.model.score_threshold;
    let mut pipeline = Pipeline::build(
        &*platform,
        config,
        Box::new(SimPostProcessor::new(threshold)),
        Box::new(StdoutSink { verbose_frames: 3 }),
    )?;

    println!("platform: {}", platform.name());
    println!("{}", pipeline.buffer_budget());
    let metrics = pipeline.run(frames)?;
    println!("done: {}", metrics.summary());
    Ok(())
}
