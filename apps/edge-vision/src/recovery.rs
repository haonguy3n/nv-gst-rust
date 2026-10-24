//! Once-per-boot Argus recovery policy.
//!
//! A bare EOS from an Argus camera is not a clean end: the sensor or
//! `nvargus-daemon` may be gone, and the Argus client does not recover
//! in-process. The app restarts the daemon once, exits 75, and lets a
//! systemd unit with `RestartForceExitStatus=75` start a fresh process.
//! A second loss in the same boot is only logged, or a reboot loop
//! follows.

use std::path::PathBuf;
use std::process::Command;

const DEFAULT_MARKER: &str = "/run/edge-vision/argus-recovered";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// First loss this boot: restart the daemon and exit 75.
    RestartDaemonAndExit75,
    /// Already recovered this boot: log and fail; do not loop.
    LogAndFail,
}

pub struct CaptureRecoveryPolicy {
    marker: PathBuf,
}

impl CaptureRecoveryPolicy {
    pub fn new(marker: impl Into<PathBuf>) -> Self {
        Self {
            marker: marker.into(),
        }
    }

    pub fn from_env() -> Self {
        let marker = std::env::var_os("EDGE_VISION_RECOVERY_MARKER")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MARKER));
        Self::new(marker)
    }

    /// Mark the recovery and return what the app should do. A marker
    /// that cannot be written is treated as "already recovered" so a
    /// read-only /run cannot cause a restart loop.
    pub fn on_capture_loss(&self) -> RecoveryAction {
        if self.marker.exists() {
            return RecoveryAction::LogAndFail;
        }
        if let Some(parent) = self.marker.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return RecoveryAction::LogAndFail;
            }
        }
        if std::fs::write(&self.marker, b"argus recovered once this boot\n").is_err() {
            return RecoveryAction::LogAndFail;
        }
        RecoveryAction::RestartDaemonAndExit75
    }
}

/// Best-effort `systemctl --no-block restart nvargus-daemon`.
pub fn request_daemon_restart() -> std::io::Result<()> {
    let status = Command::new("systemctl")
        .args(["--no-block", "restart", "nvargus-daemon"])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "systemctl restart nvargus-daemon exited {status}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_marker(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("edge-argus-recovery-{name}-{}", std::process::id()))
    }

    #[test]
    fn recovers_once_per_boot() {
        let marker = temp_marker("once");
        let _ = std::fs::remove_file(&marker);
        let policy = CaptureRecoveryPolicy::new(&marker);
        assert_eq!(
            policy.on_capture_loss(),
            RecoveryAction::RestartDaemonAndExit75
        );
        assert!(marker.exists());
        assert_eq!(policy.on_capture_loss(), RecoveryAction::LogAndFail);
        let _ = std::fs::remove_file(&marker);
    }

    #[test]
    fn unwritable_marker_does_not_loop() {
        // A path under a regular file cannot be created.
        let dir = temp_marker("unwritable");
        let _ = std::fs::remove_file(&dir);
        std::fs::write(&dir, b"file").unwrap();
        let policy = CaptureRecoveryPolicy::new(dir.join("marker"));
        assert_eq!(policy.on_capture_loss(), RecoveryAction::LogAndFail);
        let _ = std::fs::remove_file(&dir);
    }
}
