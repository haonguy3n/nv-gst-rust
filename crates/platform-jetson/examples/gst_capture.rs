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

fn main() {
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
        frames: None,
    };
    let (launch, memory, layout) = if test_pattern {
        (
            "videotestsrc is-live=true ! videoconvert \
             ! video/x-raw,format=NV12,width=1440,height=1080,framerate=30/1 \
             ! appsink name=frames max-buffers=2 drop=true sync=false"
                .to_string(),
            FrameMemory::DmaBuf,
            Layout::PitchLinear,
        )
    } else {
        (capture_launch(&cfg), FrameMemory::Nvmm, Layout::BlockLinear)
    };
    println!("launch: {launch}");

    let mut source =
        GstLaunchSource::new(&launch, cfg, memory, layout).expect("build capture source");
    source.start().expect("start capture");
    for _ in 0..frames {
        match source.next_frame(Duration::from_secs(2)) {
            Ok(FramePoll::Frame(frame)) => println!(
                "seq={} pts={:?} fd={} {}x{}",
                frame.meta().seq,
                frame.meta().pts,
                frame.meta().dmabuf_fd,
                frame.meta().width,
                frame.meta().height
            ),
            Ok(FramePoll::Timeout) => println!("timeout (no frame due)"),
            Ok(FramePoll::EndOfStream) => {
                println!("end of stream");
                break;
            }
            Err(e) => panic!("capture failed: {e}"),
        }
    }
    source.stop().expect("stop capture");
    println!(
        "copied={} (must be 0 on the Jetson)",
        source.copied_frames()
    );
}
