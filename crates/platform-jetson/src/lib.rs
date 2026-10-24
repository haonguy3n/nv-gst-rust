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
pub mod gst_encoder;
#[cfg(feature = "gst")]
pub mod gst_source;
pub mod launch;
pub mod plan_cache;
pub mod probe;

pub use probe::{
    choose_encoder, dla_cores, probe_media_engines, probe_module_identity, EncoderChoice,
    EncoderKind, MediaEngines, ModuleIdentity,
};

#[cfg(feature = "gst")]
use edge_core::Layout;
use edge_core::{
    Codec, Encoder, EncoderConfig, EngineState, Error, FrameMemory, FrameSource, GpuStream,
    ModelConfig, Overlay, PixelFormat, Platform, PlatformCaps, PreConfig, Precision, Preprocessor,
    Result, SourceConfig,
};
use std::path::Path;

/// The Jetson platform. Construct with the sysfs root to probe
/// (`/` on device; also how tests inject a fake device tree).
pub struct JetsonPlatform {
    engines: MediaEngines,
    module: ModuleIdentity,
    root: std::path::PathBuf,
    /// Frame rate of the last `source()`; the encoder's appsrc caps need it.
    source_fps: std::sync::atomic::AtomicU32,
}

impl JetsonPlatform {
    pub fn new(sysfs_root: &Path) -> Self {
        Self {
            engines: probe_media_engines(sysfs_root),
            module: probe_module_identity(sysfs_root),
            root: sysfs_root.to_path_buf(),
            source_fps: std::sync::atomic::AtomicU32::new(30),
        }
    }

    pub fn with_engines(engines: MediaEngines) -> Self {
        Self {
            engines,
            module: ModuleIdentity::default(),
            root: std::path::PathBuf::from("/"),
            source_fps: std::sync::atomic::AtomicU32::new(30),
        }
    }

    pub fn engines(&self) -> MediaEngines {
        self.engines
    }

    pub fn module_identity(&self) -> &ModuleIdentity {
        &self.module
    }
}

const M1_PENDING: &str = "platform-jetson device runtime lands in M1 \
    (GStreamer pipelines on the Orin; see launch.rs for the pinned strings)";

/// Is the CPU encode *stage* part of this build? Element presence is
/// checked in `encoder()`, so a missing `x264enc` fails with the package
/// name instead of being hidden behind a generic "no encoder" error.
fn cpu_encode_available() -> bool {
    cfg!(feature = "gst")
}

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
            // INT8 is not advertised until a calibration cache is part
            // of the plan key and an accuracy gate exists; `--int8`
            // without calibration silently uses dummy scales.
            precisions: vec![Precision::Fp32, Precision::Fp16],
            devices: {
                let mut devices = vec!["gpu".into()];
                devices.extend(dla_cores(&self.root));
                devices
            },
            has_hw_encoder: self.engines.nvenc,
            // The CPU stage is compiled in; element presence is
            // checked in `encoder()`, so a missing x264 names its
            // package instead of being hidden by caps. A strict
            // profile still refuses it in the builder.
            cpu_encode: cpu_encode_available(),
        }
    }

    fn source(&self, cfg: &SourceConfig) -> Result<Box<dyn FrameSource>> {
        self.source_fps
            .store(cfg.fps.max(1), std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "gst")]
        {
            let launch = match cfg.kind.as_str() {
                "argus" => launch::capture_launch(cfg),
                "file" => {
                    if cfg.path.is_none() {
                        return Err(Error::invalid("source.path", "required for file source"));
                    }
                    launch::file_capture_launch(cfg)
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
                // nvarguscamerasrc NVMM surfaces are block-linear.
                Layout::BlockLinear,
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

    fn engine(&self, _cfg: &ModelConfig) -> Result<EngineState> {
        Err(Error::Unsupported(M1_PENDING.into()))
    }

    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        Err(Error::Unsupported(M1_PENDING.into()))
    }

    fn stream(&self, _camera_id: u8) -> Result<GpuStream> {
        // Until T4 lands the real CudaStream, Jetson stages do not
        // run at all (M1_PENDING), so a no-op stream is honest.
        Ok(GpuStream::noop())
    }

    fn encoder(
        &self,
        cfg: &EncoderConfig,
        width: u32,
        height: u32,
    ) -> Result<Option<Box<dyn Encoder>>> {
        if cfg.hardware {
            // NVENC needs the NVMM capture branch and the NVENC stage;
            // both are M1 device work. A Nano has no NVENC at all.
            return if self.engines.nvenc {
                Err(Error::Unsupported(M1_PENDING.into()))
            } else {
                Err(Error::Unsupported(
                    "encoder.hardware = true but this Jetson has no NVENC \
                     (Orin Nano?); use zero_copy = \"relaxed\" with \
                     encoder.hardware = false and H.264 for the CPU path"
                        .into(),
                ))
            };
        }
        // CPU path: the Nano's only video encoder. H.264 first:
        // software H.265 cannot hold 1440x1080@30 on the Nano.
        if cfg.codec != Codec::H264 {
            return Err(Error::Unsupported(
                "software H.265 is not supported (it cannot hold 1440x1080@30 \
                 on the Nano); use codec = \"H264\" with encoder.hardware = false"
                    .into(),
            ));
        }
        #[cfg(feature = "gst")]
        {
            if !gst_encoder::software_h264_available() {
                return Err(Error::Unsupported(
                    "CPU H.264 encoder unavailable: GStreamer element `x264enc` \
                     (or `h264parse`) is missing; install \
                     gstreamer1.0-plugins-ugly-x264"
                        .into(),
                ));
            }
            Ok(Some(Box::new(
                gst_encoder::GstEncoder::new(EncoderKind::Cpu, cfg, width, height)?
                    .with_fps(self.source_fps.load(std::sync::atomic::Ordering::Relaxed)),
            )))
        }
        #[cfg(not(feature = "gst"))]
        {
            let _ = (width, height);
            Err(Error::Unsupported(
                "CPU encoder needs the `gst` feature (platform-jetson/gst)".into(),
            ))
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
        assert!(!nx.caps().precisions.contains(&Precision::Int8));

        let nano = JetsonPlatform::with_engines(MediaEngines::default());
        assert!(!nano.caps().has_hw_encoder);
        assert!(nano.caps().codecs.is_empty());
        // `EncoderConfig::default()` asks for hardware; a Nano has none.
        let err = match nano.encoder(&EncoderConfig::default(), 1440, 1080) {
            Ok(_) => panic!("hardware=true on a Nano must fail"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("no NVENC"), "{err}");
    }

    #[test]
    fn stages_are_honestly_unsupported_until_m1() {
        let p = JetsonPlatform::with_engines(MediaEngines {
            nvenc: true,
            nvjpg: false,
        });
        let err = match p.encoder(&EncoderConfig::default(), 1440, 1080) {
            Ok(_) => panic!("runtime must not pretend to exist"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("M1"), "{err}");
    }

    #[cfg(feature = "gst")]
    #[test]
    fn nano_cpu_encoder_is_h264_only_and_matches_caps() {
        let nano = JetsonPlatform::with_engines(MediaEngines::default());
        assert!(
            nano.caps().cpu_encode,
            "caps must advertise the CPU stage compiled into this build"
        );
        let h264 = EncoderConfig {
            enabled: true,
            hardware: false,
            codec: Codec::H264,
            ..EncoderConfig::default()
        };
        if !gst_encoder::software_h264_available() {
            let err = match nano.encoder(&h264, 1440, 1080) {
                Ok(_) => panic!("missing x264enc must fail"),
                Err(e) => e.to_string(),
            };
            assert!(
                err.contains("gstreamer1.0-plugins-ugly-x264"),
                "missing element must name the package: {err}"
            );
            return;
        }
        assert!(nano
            .encoder(&h264, 1440, 1080)
            .expect("CPU H.264 encoder")
            .is_some());
        let h265 = EncoderConfig {
            codec: Codec::H265,
            ..h264
        };
        let err = match nano.encoder(&h265, 1440, 1080) {
            Ok(_) => panic!("software H.265 must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("H.265") && err.contains("H264"), "{err}");
    }

    /// Strict mode keeps refusing CPU encoding before the platform is
    /// reached; the error names the mode.
    #[test]
    fn strict_zero_copy_refuses_cpu_encode_before_the_platform() {
        let err = edge_core::Config::from_toml_str(
            r#"
platform = "jetson"
zero_copy = "strict"
[source]
kind = "argus"
width = 1440
height = 1080
fps = 30
[model]
input_width = 416
input_height = 416
[encoder]
enabled = true
hardware = false
[stream]
metadata_only = false
"#,
        )
        .expect_err("strict + CPU encoder must be rejected");
        assert!(err.to_string().contains("strict"), "{err}");
    }
}
