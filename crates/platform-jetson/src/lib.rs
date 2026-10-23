//! Jetson backend (Orin NX / Nano, JetPack 7 / L4T r39.2.1).
//!
//! M1 takes the hybrid path proven in the owner's camera products:
//! GStreamer pipelines (Argus capture into NVMM, NVENC encode) meet
//! our Rust stages at `appsink`/`appsrc`, and direct libargus
//! capture can later replace the capture string behind the same
//! `FrameSource` trait. This crate currently holds the parts that
//! are testable without a device — the media-engine probe and the
//! launch-string builders — plus honest caps derived from the
//! probe. The GStreamer runtime (gstreamer-rs elements, the
//! `nvmmalloc` allocator for feeding NVENC the very same dmabuf,
//! TensorRT via a thin C++ shim) lands with device bring-up in M1;
//! stage factories return `Unsupported` until then, so nothing
//! silently pretends to run.
//!
//! Reference material (read, not copied): the owner's
//! `orin-nx-camera-viber` (frame path, media-engine probe, launch
//! strings) and `gst-nvmm-cpp` (LGPL NVMM GStreamer elements;
//! validated on JetPack 5.1/6 — JetPack 7 validation is ours to do).

#![forbid(unsafe_code)]

#[cfg(feature = "gst")]
pub mod gst_source;
pub mod launch;
pub mod probe;

pub use probe::{choose_encoder, probe_media_engines, EncoderChoice, EncoderKind, MediaEngines};

use edge_core::{
    Codec, Encoder, EncoderConfig, Error, FrameMemory, FrameSource, InferenceEngine, ModelConfig,
    Overlay, PixelFormat, Platform, PlatformCaps, PreConfig, Preprocessor, Result, SourceConfig,
};
use std::path::Path;

/// The Jetson platform. Construct with the sysfs root to probe
/// (`/` on device; also how tests inject a fake device tree).
pub struct JetsonPlatform {
    engines: MediaEngines,
}

impl JetsonPlatform {
    pub fn new(sysfs_root: &Path) -> Self {
        Self {
            engines: probe_media_engines(sysfs_root),
        }
    }

    pub fn with_engines(engines: MediaEngines) -> Self {
        Self { engines }
    }

    pub fn engines(&self) -> MediaEngines {
        self.engines
    }
}

const M1_PENDING: &str = "platform-jetson device runtime lands in M1 \
    (GStreamer pipelines on the Orin; see launch.rs for the pinned strings)";

impl Platform for JetsonPlatform {
    fn name(&self) -> &'static str {
        "jetson"
    }

    fn caps(&self) -> PlatformCaps {
        PlatformCaps {
            frame_memory: FrameMemory::Nvmm,
            formats: vec![PixelFormat::Nv12],
            // IMX296 native is 1440x1080; the ISP/encoder ceiling
            // is higher, and 4K is the honest pipeline cap.
            max_width: 3840,
            max_height: 2160,
            codecs: if self.engines.nvenc {
                vec![Codec::H264, Codec::H265]
            } else {
                Vec::new()
            },
            has_hw_encoder: self.engines.nvenc,
            // launch::encode_launch has the VIC + x264/x265 strings;
            // the runtime lands with the M1 encoder stage.
            cpu_encode: true,
            accelerators: vec!["gpu".into()],
        }
    }

    fn source(&self, cfg: &SourceConfig) -> Result<Box<dyn FrameSource>> {
        #[cfg(feature = "gst")]
        {
            let launch = match cfg.kind.as_str() {
                "argus" => launch::capture_launch(cfg),
                "file" => {
                    let path = cfg
                        .path
                        .as_deref()
                        .ok_or_else(|| Error::invalid("source.path", "required for file source"))?;
                    launch::file_capture_launch(cfg, &path.to_string_lossy())
                }
                other => {
                    return Err(Error::Unsupported(format!(
                        "jetson source kind `{other}` (use \"argus\" or \"file\")"
                    )))
                }
            };
            Ok(Box::new(gst_source::GstLaunchSource::new(
                &launch,
                cfg.clone(),
                FrameMemory::Nvmm,
            )?))
        }
        #[cfg(not(feature = "gst"))]
        {
            let _ = cfg;
            Err(Error::Unsupported(
                "platform-jetson built without the `gst` feature".into(),
            ))
        }
    }

    fn preprocessor(&self, _cfg: &PreConfig) -> Result<Box<dyn Preprocessor>> {
        Err(Error::Unsupported(M1_PENDING.into()))
    }

    fn engine(&self, _cfg: &ModelConfig) -> Result<Box<dyn InferenceEngine>> {
        Err(Error::Unsupported(M1_PENDING.into()))
    }

    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        Err(Error::Unsupported(M1_PENDING.into()))
    }

    fn encoder(&self, _cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>> {
        // Honest even before the runtime exists: the option must
        // match caps, and caps come from the probe.
        if self.engines.nvenc {
            Err(Error::Unsupported(M1_PENDING.into()))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_follow_the_probe() {
        let nx = JetsonPlatform::with_engines(MediaEngines {
            nvenc: true,
            nvjpg: true,
        });
        assert!(nx.caps().has_hw_encoder);
        assert_eq!(nx.caps().codecs.len(), 2);

        let nano = JetsonPlatform::with_engines(MediaEngines::default());
        assert!(!nano.caps().has_hw_encoder);
        assert!(nano.caps().codecs.is_empty());
        assert!(matches!(nano.encoder(&EncoderConfig::default()), Ok(None)));
    }

    #[test]
    fn stages_are_honestly_unsupported_until_m1() {
        let p = JetsonPlatform::with_engines(MediaEngines {
            nvenc: true,
            nvjpg: false,
        });
        let err = match p.encoder(&EncoderConfig::default()) {
            Ok(_) => panic!("runtime must not pretend to exist"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("M1"), "{err}");
    }
}
