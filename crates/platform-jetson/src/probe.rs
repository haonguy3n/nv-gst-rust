//! Which fixed-function media engines does this Jetson have?
//!
//! Every Orin has the VIC; not every Orin has NVENC (Orin Nano: none)
//! or NVJPG. Presence is a question, never an assumption: probe
//! sysfs for a platform device whose name contains `.nvenc` /
//! `.nvjpg` with a driver bound. A missing directory (a dev PC) is
//! simply "no engine".

use edge_core::{Error, Result, ZeroCopyMode};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MediaEngines {
    /// H.264/H.265 hardware encode (Orin NX: yes, Orin Nano: no).
    pub nvenc: bool,
    /// Hardware JPEG encode for snapshots.
    pub nvjpg: bool,
}

/// Pure core of the probe: `entries` is `(device_name, driver_bound)`
/// pairs. Separated so tests never touch the real `/sys`.
pub fn engines_from_entries(entries: &[(String, bool)]) -> MediaEngines {
    let has = |kind: &str| {
        entries
            .iter()
            .any(|(name, bound)| *bound && name.contains(kind))
    };
    MediaEngines {
        nvenc: has(".nvenc"),
        nvjpg: has(".nvjpg"),
    }
}

/// Probe `<root>/sys/bus/platform/devices` (`root` is `""` on device).
pub fn probe_media_engines(root: &Path) -> MediaEngines {
    let devices = root.join("sys/bus/platform/devices");
    let mut entries = Vec::new();
    if let Ok(read) = std::fs::read_dir(&devices) {
        for entry in read.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let bound = entry.path().join("driver").exists();
            entries.push((name, bound));
        }
    }
    engines_from_entries(&entries)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderChoice {
    /// Hardware when present; under `zero_copy = "strict"` a missing
    /// NVENC is an error, not a silent CPU fallback.
    Auto,
    Hardware,
    Software,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderKind {
    Nvenc,
    /// VIC copy out of NVMM into system-memory I420, then x264/x265.
    /// Measured in the reference product at 30 fps for roughly one
    /// Orin NX core; still one CPU copy of every pixel, so it is an
    /// explicit opt-in here, never the strict-mode path.
    Cpu,
}

/// Decide how video gets encoded on this machine.
pub fn choose_encoder(
    choice: EncoderChoice,
    present: MediaEngines,
    zero_copy: ZeroCopyMode,
) -> Result<EncoderKind> {
    match choice {
        EncoderChoice::Hardware => {
            if present.nvenc {
                Ok(EncoderKind::Nvenc)
            } else {
                Err(Error::Unsupported(
                    "encoder = hardware but this Jetson has no NVENC \
                     (Orin Nano?); use an Orin NX for video or stream metadata only"
                        .into(),
                ))
            }
        }
        EncoderChoice::Software => Ok(EncoderKind::Cpu),
        EncoderChoice::Auto => {
            if present.nvenc {
                Ok(EncoderKind::Nvenc)
            } else if zero_copy == ZeroCopyMode::Strict {
                Err(Error::Unsupported(
                    "no NVENC on this Jetson and zero_copy = \"strict\" refuses a CPU \
                     encoder; stream metadata only, or set zero_copy = \"relaxed\" \
                     to opt in to the measured CPU-encode path"
                        .into(),
                ))
            } else {
                Ok(EncoderKind::Cpu)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(pairs: &[(&str, bool)]) -> Vec<(String, bool)> {
        pairs.iter().map(|(n, b)| (n.to_string(), *b)).collect()
    }

    #[test]
    fn probe_finds_bound_engines_by_name() {
        let e = engines_from_entries(&entries(&[
            ("154c0000.nvenc", true),
            ("15500000.nvjpg", true),
            ("13e00000.host1x", true),
        ]));
        assert!(e.nvenc && e.nvjpg);
    }

    #[test]
    fn unbound_or_absent_device_is_no_engine() {
        let e = engines_from_entries(&entries(&[("154c0000.nvenc", false)]));
        assert!(!e.nvenc && !e.nvjpg);
        assert_eq!(engines_from_entries(&[]), MediaEngines::default());
    }

    #[test]
    fn probe_reads_a_sysfs_tree() {
        let root = std::env::temp_dir().join(format!("edge-probe-test-{}", std::process::id()));
        let devices = root.join("sys/bus/platform/devices");
        for (name, bound) in [("154c0000.nvenc", true), ("15500000.nvjpg", false)] {
            let dir = devices.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            if bound {
                std::fs::create_dir_all(dir.join("driver")).unwrap();
            }
        }
        let e = probe_media_engines(&root);
        assert!(e.nvenc, "bound .nvenc must be found");
        assert!(!e.nvjpg, "unbound .nvjpg must not count");
        let _ = std::fs::remove_dir_all(&root);
        // A dev PC without the directory probes as "no engines".
        assert_eq!(
            probe_media_engines(std::path::Path::new("/nonexistent")),
            MediaEngines::default()
        );
    }

    #[test]
    fn strict_auto_without_nvenc_is_an_error_not_a_fallback() {
        let none = MediaEngines::default();
        let err = choose_encoder(EncoderChoice::Auto, none, ZeroCopyMode::Strict)
            .unwrap_err()
            .to_string();
        assert!(err.contains("strict"), "{err}");
        assert_eq!(
            choose_encoder(EncoderChoice::Auto, none, ZeroCopyMode::Relaxed).unwrap(),
            EncoderKind::Cpu,
            "relaxed mode is the explicit CPU-encode opt-in"
        );
    }

    #[test]
    fn hardware_choice_requires_the_engine() {
        let none = MediaEngines::default();
        assert!(choose_encoder(EncoderChoice::Hardware, none, ZeroCopyMode::Relaxed).is_err());
        let nx = MediaEngines {
            nvenc: true,
            nvjpg: true,
        };
        assert_eq!(
            choose_encoder(EncoderChoice::Hardware, nx, ZeroCopyMode::Strict).unwrap(),
            EncoderKind::Nvenc
        );
        assert_eq!(
            choose_encoder(EncoderChoice::Software, nx, ZeroCopyMode::Strict).unwrap(),
            EncoderKind::Cpu,
            "software forces the Nano path even on an NX (for testing it)"
        );
    }
}
