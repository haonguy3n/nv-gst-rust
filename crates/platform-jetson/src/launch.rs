//! GStreamer launch strings for the M1 (hybrid) Jetson path,
//! built from typed values and pinned by unit tests — the same
//! practice as the reference products' launch-string checks: the
//! string IS the pipeline contract, so it is reviewed like code.
//!
//! Two pipelines meet in our Rust stages:
//!
//! - capture: `nvarguscamerasrc` writes NV12 into NVMM; an
//!   `appsink` hands those GstBuffers to the pipeline as `HwFrame`s
//!   wrapping the same NvBufSurface (dmabuf fd via the GstNvmm
//!   allocator; see the `nvmmalloc` element in gst-nvmm-cpp, which
//!   our Jetson image will need for the encode side).
//! - encode: an `appsrc` takes the same NVMM frames back into
//!   `nvv4l2h264enc`/`nvv4l2h265enc` (or, opt-in only, a VIC copy
//!   to system-memory I420 and a CPU encoder on Nano-class SoCs).
//!
//! Direct libargus capture replaces the first string later, behind
//! the same `FrameSource` trait; nothing else changes.

use crate::probe::EncoderKind;
use edge_core::{Codec, EncoderConfig, SourceConfig};

fn raw_caps(memory: &str, format: &str, width: u32, height: u32) -> String {
    format!("video/x-raw{memory},format={format},width={width},height={height}")
}

/// Capture side: Argus ISP -> NV12 in NVMM -> appsink.
pub fn capture_launch(cfg: &SourceConfig) -> String {
    format!(
        "nvarguscamerasrc name=camsrc sensor-id={} \
         ! {},framerate={}/1 \
         ! appsink name=frames max-buffers={} drop=false sync=false",
        cfg.camera_id,
        raw_caps("(memory:NVMM)", "NV12", cfg.width, cfg.height),
        cfg.fps.max(1),
        cfg.pool_size,
    )
}

/// File source on the device (recorded clips in hardware CI):
/// decoded and converted into NVMM before the appsink, so every
/// downstream stage sees the same caps as a live sensor.
pub fn file_capture_launch(cfg: &SourceConfig, location: &str) -> String {
    format!(
        "filesrc location={location} ! decodebin ! nvvidconv \
         ! {} ! appsink name=frames max-buffers={} drop=false sync=false",
        raw_caps("(memory:NVMM)", "NV12", cfg.width, cfg.height),
        cfg.pool_size,
    )
}

/// Encode side: appsrc (NVMM in) -> encoder -> byte-stream appsink.
pub fn encode_launch(kind: EncoderKind, cfg: &EncoderConfig, width: u32, height: u32) -> String {
    let bitrate_bps = cfg.bitrate_kbps.saturating_mul(1000);
    let src = format!(
        "appsrc name=framesrc is-live=true format=time \
         caps={}",
        raw_caps("(memory:NVMM)", "NV12", width, height)
    );
    match (kind, cfg.codec) {
        (EncoderKind::Nvenc, Codec::H264) => format!(
            "{src} ! nvv4l2h264enc bitrate={bitrate_bps} insert-sps-pps=true \
             idrinterval=30 maxperf-enable=true \
             ! h264parse config-interval=-1 \
             ! video/x-h264,stream-format=byte-stream,alignment=au \
             ! appsink name=enc sync=false"
        ),
        (EncoderKind::Nvenc, Codec::H265) => format!(
            "{src} ! nvv4l2h265enc bitrate={bitrate_bps} insert-sps-pps=true \
             idrinterval=30 maxperf-enable=true \
             ! h265parse config-interval=-1 \
             ! video/x-h265,stream-format=byte-stream,alignment=au \
             ! appsink name=enc sync=false"
        ),
        // Opt-in only (zero_copy = "relaxed"): the one deliberate
        // copy out of NVMM, immediately before the CPU encoder.
        (EncoderKind::Cpu, Codec::H264) => format!(
            "{src} ! nvvidconv ! {} \
             ! x264enc tune=zerolatency key-int-max=30 bitrate={} speed-preset=ultrafast \
             ! h264parse config-interval=-1 \
             ! video/x-h264,stream-format=byte-stream,alignment=au \
             ! appsink name=enc sync=false",
            raw_caps("", "I420", width, height),
            cfg.bitrate_kbps,
        ),
        (EncoderKind::Cpu, Codec::H265) => format!(
            "{src} ! nvvidconv ! {} \
             ! x265enc tune=zerolatency key-int-max=30 bitrate={} speed-preset=ultrafast \
             ! h265parse config-interval=-1 \
             ! video/x-h265,stream-format=byte-stream,alignment=au \
             ! appsink name=enc sync=false",
            raw_caps("", "I420", width, height),
            cfg.bitrate_kbps,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_cfg() -> SourceConfig {
        SourceConfig {
            kind: "argus".into(),
            path: None,
            camera_id: 1,
            width: 1440,
            height: 1080,
            fps: 30,
            paced: true,
            pool_size: 6,
            frames: None,
        }
    }

    #[test]
    fn capture_launch_is_pinned() {
        assert_eq!(
            capture_launch(&source_cfg()),
            "nvarguscamerasrc name=camsrc sensor-id=1 \
             ! video/x-raw(memory:NVMM),format=NV12,width=1440,height=1080,framerate=30/1 \
             ! appsink name=frames max-buffers=6 drop=false sync=false"
        );
    }

    #[test]
    fn nvenc_h265_launch_is_pinned() {
        let cfg = EncoderConfig {
            codec: Codec::H265,
            bitrate_kbps: 8000,
            ..EncoderConfig::default()
        };
        let launch = encode_launch(EncoderKind::Nvenc, &cfg, 1440, 1080);
        assert!(launch.contains("nvv4l2h265enc bitrate=8000000"), "{launch}");
        assert!(
            launch.contains("video/x-raw(memory:NVMM),format=NV12"),
            "{launch}"
        );
        assert!(
            !launch.contains("nvvidconv"),
            "HW path must not copy: {launch}"
        );
    }

    #[test]
    fn cpu_encode_launch_has_exactly_one_copy_out_of_nvmm() {
        let cfg = EncoderConfig::default();
        let launch = encode_launch(EncoderKind::Cpu, &cfg, 1440, 1080);
        assert_eq!(launch.matches("nvvidconv").count(), 1, "{launch}");
        assert!(launch.contains("x264enc"), "{launch}");
        assert!(launch.contains("format=I420"), "{launch}");
    }
}
