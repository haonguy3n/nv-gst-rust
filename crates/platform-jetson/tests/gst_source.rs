#![cfg(feature = "gst")]
//! Capture through the GStreamer runtime on a dev PC: the same
//! code path the Jetson takes, fed by a test pattern instead of
//! nvarguscamerasrc. PC frames are system memory, so every frame
//! takes the counted copy fallback — on the device that counter
//! must stay at zero.
//!
//! Tests that depend on specific elements skip cleanly when the
//! element is absent (minimal device image) or present (the
//! missing-element test on an actual Jetson).

use edge_core::{FrameMemory, FramePoll, FrameSource, Layout, SourceConfig};
use platform_jetson::gst_source::GstLaunchSource;
use std::time::Duration;

fn element_exists(name: &str) -> bool {
    gstreamer::init().expect("gst init");
    gstreamer::ElementFactory::find(name).is_some()
}

fn cfg(kind: &str) -> SourceConfig {
    SourceConfig {
        kind: kind.into(),
        path: None,
        camera_id: 0,
        width: 64,
        height: 64,
        fps: 30,
        exposure_us: None,
        gain_min: None,
        gain_max: None,
        paced: true,
        pool_size: 6,
        frames: None,
    }
}

const PATTERN_LAUNCH: &str = "videotestsrc is-live=true \
    ! videoconvert ! video/x-raw,format=NV12,width=64,height=64,framerate=30/1 \
    ! appsink name=frames max-buffers=2 drop=true sync=false";

#[test]
fn pulls_ordered_frames_from_a_launch_string() {
    if !element_exists("videotestsrc") || !element_exists("videoconvert") {
        eprintln!("skipping: test-pattern elements not in this GStreamer install");
        return;
    }
    let mut source = GstLaunchSource::new(
        PATTERN_LAUNCH,
        cfg("test"),
        FrameMemory::DmaBuf,
        Layout::PitchLinear,
    )
    .expect("build gst source");
    source.start().expect("start");

    let mut last_pts = Duration::ZERO;
    for expected in 0..5u64 {
        let FramePoll::Frame(frame) = source
            .next_frame(Duration::from_secs(2))
            .expect("next_frame")
        else {
            panic!("expected a frame");
        };
        assert_eq!(frame.meta().seq, expected);
        assert!(frame.meta().pts >= last_pts, "pts must be monotonic");
        assert_eq!(frame.meta().dmabuf_fd, frame.buf.dmabuf_fd());
        assert!(frame.buf.dmabuf_fd() >= 0);
        assert_eq!(frame.buf.memory(), FrameMemory::DmaBuf);
        last_pts = frame.meta().pts;
    }
    source.stop().expect("stop");
    assert_eq!(
        source.copied_frames(),
        5,
        "PC system-memory frames take the counted copy fallback"
    );
}

#[test]
fn missing_element_fails_at_construction_with_a_clear_error() {
    if element_exists("nvarguscamerasrc") {
        eprintln!("skipping: nvarguscamerasrc exists here (device?)");
        return;
    }
    match GstLaunchSource::new(
        "nvarguscamerasrc ! appsink name=frames",
        cfg("argus"),
        FrameMemory::Nvmm,
        Layout::BlockLinear,
    ) {
        Ok(_) => panic!("nvarguscamerasrc must not exist on this machine"),
        Err(e) => assert!(!e.to_string().is_empty()),
    }
}

/// R2-1: an NVMM-declared source whose buffers are not dmabuf
/// memory must fail loudly, never silently copy.
#[test]
fn nvmm_pipeline_without_dmabuf_refuses_instead_of_copying() {
    if !element_exists("videotestsrc") {
        eprintln!("skipping: videotestsrc not in this GStreamer install");
        return;
    }
    let mut source = GstLaunchSource::new(
        PATTERN_LAUNCH,
        cfg("argus"),
        FrameMemory::Nvmm,
        Layout::BlockLinear,
    )
    .expect("build gst source");
    source.start().expect("start");
    match source.next_frame(Duration::from_secs(2)) {
        Err(e) => assert!(
            e.to_string().contains("refusing to copy"),
            "error must name the refusal: {e}"
        ),
        Ok(_) => panic!("NVMM source must not accept system-memory frames"),
    }
    let _ = source.stop();
    assert_eq!(source.copied_frames(), 0);
}

/// R2-2: a pipeline that errors after construction must surface
/// the bus error, not loop as Timeout forever.
#[test]
fn runtime_pipeline_error_surfaces_instead_of_timing_out_forever() {
    if !element_exists("filesrc") || !element_exists("decodebin") {
        eprintln!("skipping: filesrc/decodebin not in this GStreamer install");
        return;
    }
    let launch = "filesrc location=/nonexistent/definitely-missing.mp4 \
        ! decodebin ! videoconvert ! video/x-raw,format=NV12,width=64,height=64 \
        ! appsink name=frames max-buffers=2 drop=true sync=false";
    let mut source = match GstLaunchSource::new(
        launch,
        cfg("file"),
        FrameMemory::DmaBuf,
        Layout::PitchLinear,
    ) {
        Ok(s) => s,
        Err(_) => return, // failing at construction is also honest
    };
    // Either start() or a bounded number of polls must produce the error.
    if source.start().is_err() {
        return;
    }
    for _ in 0..50 {
        match source.next_frame(Duration::from_millis(100)) {
            Err(e) => {
                assert!(e.to_string().contains("pipeline error"), "{e}");
                return;
            }
            Ok(FramePoll::EndOfStream) => return, // also an honest terminal state
            Ok(_) => {}
        }
    }
    panic!("dead pipeline looped as Timeout without surfacing its bus error");
}
