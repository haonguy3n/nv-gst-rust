//! Diagnostic log lines with their source file and line.
//!
//! Role: Logger (backend of the `log` facade). Under systemd
//! (`JOURNAL_STREAM` set) each line starts with a syslog priority
//! (`<3>`..`<7>`) that the journal parses and no timestamp; on a
//! terminal it starts with the seconds since start. One `write` per
//! line, so lines from different threads do not interleave.

use log::{Level, LevelFilter, Log, Metadata, Record};
use std::io::Write;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    Journal,
    Terminal,
}

struct Logger {
    style: Style,
    start: Instant,
}

fn syslog_priority(level: Level) -> u8 {
    match level {
        Level::Error => 3,
        Level::Warn => 4,
        Level::Info => 6,
        Level::Debug | Level::Trace => 7,
    }
}

/// `[<prio> | <elapsed>] LEVEL [thread] file:line message`, no newline.
fn format_line(
    style: Style,
    elapsed: Duration,
    record: &Record<'_>,
    thread: Option<&str>,
) -> String {
    let prefix = match style {
        Style::Journal => format!("<{}>", syslog_priority(record.level())),
        Style::Terminal => format!("{:8.3}s ", elapsed.as_secs_f64()),
    };
    let thread = thread.map_or_else(String::new, |t| format!("[{t}] "));
    format!(
        "{prefix}{:<5} {thread}{}:{} {}",
        record.level(),
        record.file().unwrap_or("?"),
        record.line().unwrap_or(0),
        record.args()
    )
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let current = std::thread::current();
        let mut line = format_line(self.style, self.start.elapsed(), record, current.name());
        line.push('\n');
        let _ = std::io::stderr().lock().write_all(line.as_bytes());
    }

    fn flush(&self) {}
}

pub fn parse_level(text: &str) -> Option<LevelFilter> {
    text.parse().ok()
}

/// Installs the logger once; later calls change nothing.
pub fn init(level: LevelFilter) {
    let style = if std::env::var_os("JOURNAL_STREAM").is_some() {
        Style::Journal
    } else {
        Style::Terminal
    };
    let logger = Box::new(Logger {
        style,
        start: Instant::now(),
    });
    if log::set_logger(Box::leak(logger)).is_ok() {
        log::set_max_level(level);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(style: Style, level: Level, thread: Option<&str>) -> String {
        let args = format_args!("viewer {} connected", "1.2.3.4:5");
        let record = Record::builder()
            .args(args)
            .level(level)
            .file(Some("crates/sinks/src/web_ui.rs"))
            .line(Some(245))
            .build();
        format_line(style, Duration::from_millis(12_345), &record, thread)
    }

    #[test]
    fn journal_line_has_priority_and_no_time() {
        assert_eq!(
            line(Style::Journal, Level::Warn, Some("encode-cam0")),
            "<4>WARN  [encode-cam0] crates/sinks/src/web_ui.rs:245 viewer 1.2.3.4:5 connected"
        );
        assert!(line(Style::Journal, Level::Error, None).starts_with("<3>ERROR "));
        assert!(line(Style::Journal, Level::Info, None).starts_with("<6>INFO "));
        assert!(line(Style::Journal, Level::Debug, None).starts_with("<7>DEBUG"));
        assert!(line(Style::Journal, Level::Trace, None).starts_with("<7>TRACE"));
    }

    #[test]
    fn terminal_line_has_elapsed_time() {
        assert_eq!(
            line(Style::Terminal, Level::Info, None),
            "  12.345s INFO  crates/sinks/src/web_ui.rs:245 viewer 1.2.3.4:5 connected"
        );
    }

    #[test]
    fn level_names_parse() {
        assert_eq!(parse_level("debug"), Some(LevelFilter::Debug));
        assert_eq!(parse_level("warn"), Some(LevelFilter::Warn));
        assert_eq!(parse_level("loud"), None);
    }
}
