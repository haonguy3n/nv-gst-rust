#![cfg(feature = "gst")]
//! CPU H.264 encoder path (Orin Nano). The production launch string
//! needs `nvvidconv`/NVMM, which a dev PC does not have, so the encode
//! test uses a system-memory launch string of the same shape. It
//! proves the packet plumbing (appsrc → x264enc → h264parse →
//! appsink → EncodedPacket), not device zero-copy.

use edge_core::{
    Codec, EncodedPacket, Encoder, EncoderConfig, FrameMemory, FrameMeta, HwFrame, Layout,
    PixelFormat,
};
use platform_jetson::gst_encoder::{check_encoder_elements, software_h264_available, GstEncoder};
use platform_jetson::gst_source::GstFrameBuffer;
use platform_jetson::probe::EncoderKind;
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 64;
const H: u32 = 64;

fn test_launch() -> &'static str {
    "appsrc name=framesrc is-live=true format=time max-buffers=2 \
     caps=video/x-raw,format=NV12,width=64,height=64,framerate=30/1 \
     ! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast \
       key-int-max=5 bitrate=500 \
     ! h264parse config-interval=-1 \
     ! video/x-h264,stream-format=byte-stream,alignment=au \
     ! appsink name=enc max-buffers=8 drop=false sync=false"
}

fn frame(seq: u64) -> HwFrame {
    let mut buffer = gstframe::buffer(seq);
    let buf = GstFrameBuffer::DmaBuf {
        fd: -1,
        buffer: buffer.clone(),
        memory: FrameMemory::DmaBuf,
    };
    let meta = FrameMeta {
        seq,
        pts: Duration::from_millis(33 * seq),
        width: W,
        height: H,
        format: PixelFormat::Nv12,
        layout: Layout::PitchLinear,
        planes: PixelFormat::Nv12.plane_layout(W, H),
        camera_id: 0,
        dmabuf_fd: -1,
    };
    let _ = buffer.get_mut();
    HwFrame::new(meta, Arc::new(buf))
}

mod gstframe {
    use gstreamer as gst;
    pub fn buffer(seq: u64) -> gst::Buffer {
        let mut buffer =
            gst::Buffer::from_slice(vec![128u8; (crate::W as usize * crate::H as usize * 3) / 2]);
        if let Some(buffer) = buffer.get_mut() {
            buffer.set_pts(gst::ClockTime::from_nseconds(seq * 33_000_000));
        }
        buffer
    }
}

#[test]
fn missing_x264_lists_the_package() {
    let err = check_encoder_elements(EncoderKind::Cpu, Codec::H264, |e| e != "x264enc")
        .expect_err("missing x264enc must be reported");
    assert!(
        err.to_string().contains("gstreamer1.0-plugins-ugly-x264"),
        "{err}"
    );
}

#[test]
fn cpu_x264_encoder_emits_h264_packets() {
    if !software_h264_available() {
        eprintln!("skipping: x264enc is not installed here");
        return;
    }
    let cfg = EncoderConfig {
        enabled: true,
        codec: Codec::H264,
        hardware: false,
        bitrate_kbps: 500,
        gop_frames: 5,
    };
    let mut encoder = GstEncoder::new_with_launch(EncoderKind::Cpu, &cfg, W, H, test_launch())
        .expect("build CPU encoder");
    let mut packets: Vec<EncodedPacket> = Vec::new();
    for seq in 0..10 {
        encoder
            .encode(frame(seq), &mut |pkt| packets.push(pkt))
            .expect("encode frame");
    }
    encoder
        .finish(&mut |pkt| packets.push(pkt))
        .expect("flush encoder");
    assert!(!packets.is_empty(), "x264 must emit at least one AU");
    assert!(
        packets.iter().any(|p| p.keyframe),
        "the first access unit must be a keyframe"
    );
    assert!(
        packets.iter().all(|p| !p.data.is_empty()),
        "no empty access units"
    );
    assert!(
        packets
            .iter()
            .any(|p| { p.data.starts_with(&[0, 0, 0, 1]) || p.data.starts_with(&[0, 0, 1]) }),
        "byte-stream output must carry an Annex-B start code"
    );
}

/// E2: the runner hands the encoder back after `finish` so a second
/// `run()` can use it. An ended appsrc cannot be pushed to, so finish
/// must tear the pipeline down.
#[test]
fn cpu_encoder_can_be_reused_after_finish() {
    if !software_h264_available() {
        eprintln!("skipping: x264enc is not installed here");
        return;
    }
    let cfg = EncoderConfig {
        enabled: true,
        codec: Codec::H264,
        hardware: false,
        bitrate_kbps: 500,
        gop_frames: 5,
    };
    let mut encoder = GstEncoder::new_with_launch(EncoderKind::Cpu, &cfg, W, H, test_launch())
        .expect("build CPU encoder");
    for round in 0..2 {
        let mut packets = Vec::new();
        for seq in 0..5 {
            encoder
                .encode(frame(seq), &mut |pkt| packets.push(pkt))
                .unwrap_or_else(|e| panic!("round {round} encode: {e}"));
        }
        encoder
            .finish(&mut |pkt| packets.push(pkt))
            .unwrap_or_else(|e| panic!("round {round} finish: {e}"));
        assert!(!packets.is_empty(), "round {round} emitted no packets");
    }
}

/// E1: a downstream caps mismatch must not block the encoder thread
/// forever, and the pipeline's bus error must reach the caller within
/// a second (the Nano ran 60 s on a not-negotiated encoder, silently).
#[test]
fn downstream_caps_mismatch_does_not_block_forever() {
    if !software_h264_available() {
        eprintln!("skipping: x264enc is not installed here");
        return;
    }
    let cfg = EncoderConfig {
        enabled: true,
        codec: Codec::H264,
        hardware: false,
        bitrate_kbps: 500,
        gop_frames: 5,
    };
    let launch = "appsrc name=framesrc is-live=true format=time block=false max-buffers=2 \
         caps=video/x-raw,format=NV12,width=64,height=64,framerate=30/1 \
         ! video/x-raw,format=I420 \
         ! x264enc tune=zerolatency speed-preset=ultrafast \
         ! h264parse config-interval=-1 \
         ! video/x-h264,stream-format=byte-stream,alignment=au \
         ! appsink name=enc max-buffers=8 drop=false sync=false";
    let mut encoder =
        GstEncoder::new_with_launch(EncoderKind::Cpu, &cfg, W, H, launch).expect("build encoder");
    let started = std::time::Instant::now();
    let mut failure = None;
    for seq in 0..60 {
        if let Err(e) = encoder.encode(frame(seq), &mut |_| {}) {
            failure = Some(e);
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "caps mismatch was not reported within 1 s"
        );
        std::thread::sleep(Duration::from_millis(33));
    }
    let err = failure.expect("a caps mismatch must surface as an encode error");
    assert!(err.to_string().contains("encoder pipeline error"), "{err}");
}

/// E1b: with a slow (not failed) element downstream, the leaky appsrc
/// must evict the oldest frame instead of queueing every push. The
/// sleeping `identity` holds the streaming thread, so the queue fills.
#[test]
fn slow_downstream_makes_the_appsrc_drop_not_queue() {
    let gst_version = gstreamer::version();
    if gst_version.1 < 20 {
        eprintln!("skipping: appsrc leaky-type needs GStreamer 1.20");
        return;
    }
    let cfg = EncoderConfig {
        enabled: true,
        codec: Codec::H264,
        hardware: false,
        bitrate_kbps: 500,
        gop_frames: 5,
    };
    let launch = "appsrc name=framesrc is-live=true format=time block=false \
         max-buffers=1 leaky-type=downstream \
         caps=video/x-raw,format=NV12,width=64,height=64,framerate=30/1 \
         ! identity sleep-time=200000 \
         ! appsink name=enc max-buffers=8 drop=false sync=false";
    let mut encoder =
        GstEncoder::new_with_launch(EncoderKind::Cpu, &cfg, W, H, launch).expect("build encoder");
    for seq in 0..30 {
        encoder
            .encode(frame(seq), &mut |_| {})
            .expect("push never fails");
    }
    assert!(
        encoder.dropped_frames() >= 20,
        "30 pushes against a 200 ms element must mostly leak, dropped {}",
        encoder.dropped_frames()
    );
}

/// Output packets carry the capture frame's PTS, not the encoder's shifted one
/// (x264enc moves timestamps up by 1000 h). The first frames start at a
/// non-zero PTS with a sub-second part, like a live capture does (a
/// whole-second start once hid a constant shift error, P1).
#[test]
fn packet_pts_equal_the_pushed_frames_pts() {
    if !software_h264_available() {
        eprintln!("skipping: x264enc is not installed here");
        return;
    }
    let cfg = EncoderConfig {
        enabled: true,
        codec: Codec::H264,
        hardware: false,
        bitrate_kbps: 500,
        gop_frames: 5,
    };
    let mut encoder = GstEncoder::new_with_launch(EncoderKind::Cpu, &cfg, W, H, test_launch())
        .expect("build CPU encoder");
    let base = 12_011; // 396.363 s
    let mut packets: Vec<EncodedPacket> = Vec::new();
    for seq in base..base + 10 {
        encoder
            .encode(frame(seq), &mut |pkt| packets.push(pkt))
            .expect("encode frame");
        // The appsrc is leaky (one buffer): pace the pushes so none is evicted.
        std::thread::sleep(Duration::from_millis(40));
    }
    encoder
        .finish(&mut |pkt| packets.push(pkt))
        .expect("flush encoder");
    let expected: Vec<Duration> = (base..base + 10)
        .map(|s| Duration::from_nanos(s * 33_000_000))
        .collect();
    let got: Vec<Duration> = packets.iter().map(|p| p.pts).collect();
    assert_eq!(got, expected, "packet pts must equal the frames' pts");
}
