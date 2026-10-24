//! Once-per-boot Argus recovery policy.
//!
//! A bare EOS from an Argus camera is not a clean end: the sensor or
//! `nvargus-daemon` may be gone, and the Argus client does not recover
//! in-process. The app restarts the daemon once, exits 75, and lets a
//! systemd unit with `RestartForceExitStatus=75` start a fresh process.
//! A second loss in the same boot is only logged, or a reboot loop
//! follows.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

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

const DEFAULT_CONTROL_SOCKET: &str = "/run/edge-vision-control.sock";

/// Ask for the nvargus-daemon restart. Under the product unit (non-root,
/// no polkit on the image) the root helper behind
/// `edge-vision-control.socket` does it; without that socket (a dev box,
/// the root dev unit) the app runs `systemctl` itself.
pub fn request_daemon_restart() -> std::io::Result<()> {
    let socket = std::env::var_os("EDGE_VISION_CONTROL_SOCKET")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONTROL_SOCKET));
    if socket.exists() {
        return restart_via_helper(&socket);
    }
    restart_via_systemctl()
}

/// One request per connection; the helper answers `ok` or why not.
fn restart_via_helper(socket: &Path) -> std::io::Result<()> {
    let mut stream = UnixStream::connect(socket)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.write_all(b"restart-capture\n")?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply)?;
    if reply.lines().last() == Some("ok") {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "control helper refused the restart: {}",
            reply.trim()
        )))
    }
}

fn restart_via_systemctl() -> std::io::Result<()> {
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

    fn serve_once(name: &str, reply: &'static str) -> (PathBuf, std::thread::JoinHandle<String>) {
        let sock = temp_marker(name);
        let _ = std::fs::remove_file(&sock);
        let listener = std::os::unix::net::UnixListener::bind(&sock).unwrap();
        let handle = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut line = String::new();
            std::io::BufRead::read_line(&mut std::io::BufReader::new(&conn), &mut line).unwrap();
            conn.write_all(reply.as_bytes()).unwrap();
            line
        });
        (sock, handle)
    }

    #[test]
    fn helper_socket_gets_the_restart_request() {
        let (sock, handle) = serve_once("helper-ok", "ok\n");
        restart_via_helper(&sock).expect("ok reply");
        assert_eq!(handle.join().unwrap(), "restart-capture\n");
        let _ = std::fs::remove_file(&sock);
    }

    #[test]
    fn helper_refusal_is_an_error() {
        let (sock, handle) = serve_once("helper-no", "unknown request\n");
        let err = restart_via_helper(&sock).expect_err("refused");
        assert!(err.to_string().contains("unknown request"), "{err}");
        handle.join().unwrap();
        let _ = std::fs::remove_file(&sock);
    }
}
