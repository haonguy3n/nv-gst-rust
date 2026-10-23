#![cfg(feature = "gst")]
//! Capture through the GStreamer runtime on a dev PC: the same
//! code path the Jetson takes, fed by a test pattern instead of
//! nvarguscamerasrc. PC frames are system memory, so every frame
//! takes the counted copy fallback — on the device that counter
//! must stay at zero.

use edge_core::{FrameMemory, FramePoll, FrameSource, SourceConfig};
use platform_jetson::gst_source::GstLaunchSource;
use std::time::Duration;

#[test]
fn pulls_ordered_frames_from_a_launch_string() {
    let cfg = SourceConfig {
        kind: "test".into(),
        path: None,
        camera_id: 0,
        width: 64,
        height: 64,
        fps: 30,
        paced: true,
        pool_size: 6,
        frames: None,
    };
    let launch = "videotestsrc is-live=true \
        ! videoconvert ! video/x-raw,format=NV12,width=64,height=64,framerate=30/1 \
        ! appsink name=frames max-buffers=6 drop=false sync=false";
    let mut source =
        GstLaunchSource::new(launch, cfg, FrameMemory::DmaBuf).expect("build gst source");
    source.start().expect("start");

    let mut last_pts = Duration::ZERO;
    for expected in 0..5u64 {
        let FramePoll::Frame(frame) = source
            .next_frame(Duration::from_secs(2))
            .expect("next_frame")
        else {
            panic!("expected a frame");
        };
        assert_eq!(frame.meta.seq, expected);
        assert!(frame.meta.pts >= last_pts, "pts must be monotonic");
        assert_eq!(frame.meta.dmabuf_fd, frame.buf.dmabuf_fd());
        assert!(frame.buf.dmabuf_fd() >= 0);
        last_pts = frame.meta.pts;
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
    let cfg = SourceConfig {
        kind: "argus".into(),
        path: None,
        camera_id: 0,
        width: 64,
        height: 64,
        fps: 30,
        paced: true,
        pool_size: 3,
        frames: None,
    };
    // nvarguscamerasrc exists only on the Jetson; on a PC the
    // launch must fail here, not at first frame.
    match GstLaunchSource::new(
        "nvarguscamerasrc ! appsink name=frames",
        cfg,
        FrameMemory::Nvmm,
    ) {
        Ok(_) => panic!("nvarguscamerasrc must not exist on this machine"),
        Err(e) => assert!(!e.to_string().is_empty()),
    }
}
