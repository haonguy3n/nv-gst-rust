//! M1 bring-up tool: capture N frames through the GStreamer
//! launch string and report what arrived.
//!
//! On the Jetson (after flashing):
//!     cargo run -p platform-jetson --features gst --example gst-capture -- 30
//! On a dev PC (test pattern instead of the sensor):
//!     cargo run -p platform-jetson --features gst --example gst-capture -- 30 --test-pattern
//!
//! The line that matters on the device is `copied=0`: any other
//! value means frames left NVMM on their way in.

use edge_core::{FrameMemory, FramePoll, FrameSource, Layout, SourceConfig};
use platform_jetson::gst_source::GstLaunchSource;
use platform_jetson::launch::capture_launch;
use std::time::Duration;

fn main() -> std::process::ExitCode {
    let mut frames: u64 = 30;
    let mut test_pattern = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--test-pattern" => test_pattern = true,
            other => frames = other.parse().expect("frames must be a number"),
        }
    }

    let cfg = SourceConfig {
        kind: "argus".into(),
        path: None,
        camera_id: 0,
        width: 1440,
        height: 1080,
        fps: 30,
        exposure_us: None,
        gain_min: None,
        gain_max: None,
        paced: true,
        pool_size: 6,
        argus_pool: None,
        frames: None,
    };
    let (launch, memory, layout) = if test_pattern {
        (
            "videotestsrc is-live=true ! videoconvert \
             ! video/x-raw,format=NV12,width=1440,height=1080,framerate=30/1 \
             ! appsink name=frames max-buffers=1 drop=true sync=false"
                .to_string(),
            FrameMemory::DmaBuf,
            Layout::PitchLinear,
        )
    } else {
        (capture_launch(&cfg), FrameMemory::Nvmm, Layout::BlockLinear)
    };
    println!("launch: {launch}");

    // Never panic after `start()`: an abort leaves the Argus session
    // behind (the first Nano run saw Error 8 in the next client).
    let mut source = match GstLaunchSource::new(&launch, cfg, memory, layout) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("gst-capture: build source: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Err(e) = source.start() {
        eprintln!("gst-capture: start capture: {e}");
        return std::process::ExitCode::FAILURE;
    }
    let mut code = std::process::ExitCode::SUCCESS;
    for _ in 0..frames {
        match source.next_frame(Duration::from_secs(2)) {
            Ok(FramePoll::Frame(frame)) => println!(
                "seq={} pts={:?} fd={} {}x{} layout={} planes={:?}",
                frame.meta().seq,
                frame.meta().pts,
                frame.meta().dmabuf_fd,
                frame.meta().width,
                frame.meta().height,
                frame.meta().layout,
                frame.meta().planes
            ),
            Ok(FramePoll::Timeout(reason)) => println!("timeout ({reason:?})"),
            Ok(FramePoll::EndOfStream) => {
                println!("end of stream");
                break;
            }
            Err(e) => {
                eprintln!("gst-capture: capture failed: {e}");
                code = std::process::ExitCode::FAILURE;
                break;
            }
        }
    }
    if let Err(e) = source.stop() {
        eprintln!("gst-capture: stop capture: {e}");
        code = std::process::ExitCode::FAILURE;
    }
    println!(
        "copied={} (must be 0 on the Jetson)",
        source.copied_frames()
    );
    code
}
