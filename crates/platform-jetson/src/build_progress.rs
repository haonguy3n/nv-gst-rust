//! Turns TensorRT's build progress events into a few log lines.
//! Role: formatter with a throttle; the clock is a parameter so the PC
//! tests it. A build runs for minutes and TensorRT reports thousands of
//! steps and hundreds of nested phases per second. Top-level phases (no
//! parent) log their start and end at info, and at most one step line per
//! `STEP_LOG_EVERY` is kept overall. Nested phases are debug lines.

use std::collections::HashMap;
use std::time::Duration;
use trt_sys::ProgressEvent;

/// Minimum time between step lines (all phases together).
pub const STEP_LOG_EVERY: Duration = Duration::from_secs(10);

/// One line and the level to log it at.
#[derive(Debug, PartialEq, Eq)]
pub struct ProgressLine {
    pub text: String,
    pub level: log::Level,
}

#[derive(Debug, Default)]
pub struct BuildProgress {
    phases: HashMap<String, PhaseState>,
    last_step_log: Option<Duration>,
}

#[derive(Debug)]
struct PhaseState {
    started: Duration,
    steps: i32,
    top_level: bool,
}

impl BuildProgress {
    /// The line to log for @p event at @p now (time since the build began),
    /// if any.
    pub fn line(&mut self, event: ProgressEvent<'_>, now: Duration) -> Option<ProgressLine> {
        match event {
            ProgressEvent::PhaseStart {
                phase,
                parent,
                steps,
            } => {
                let top_level = parent.is_none();
                if top_level {
                    self.last_step_log = Some(now);
                }
                self.phases.insert(
                    phase.to_string(),
                    PhaseState {
                        started: now,
                        steps,
                        top_level,
                    },
                );
                let of = parent.map(|p| format!(" (in {p})")).unwrap_or_default();
                Some(Self::make(
                    top_level,
                    format!("tensorrt build: {phase}{of} started, {steps} steps"),
                ))
            }
            ProgressEvent::Step { phase, step } => {
                let state = self.phases.get(phase)?;
                if !state.top_level
                    || self
                        .last_step_log
                        .is_some_and(|t| now.saturating_sub(t) < STEP_LOG_EVERY)
                {
                    return None;
                }
                let total = state.steps.max(1);
                self.last_step_log = Some(now);
                let pct = (i64::from(step.saturating_add(1)) * 100 / i64::from(total)).min(100);
                Some(Self::make(
                    true,
                    format!(
                        "tensorrt build: {phase} {}/{total} ({pct}%), {:.0} s into the build",
                        step.saturating_add(1),
                        now.as_secs_f64()
                    ),
                ))
            }
            ProgressEvent::PhaseFinish { phase } => {
                let state = self.phases.remove(phase)?;
                Some(Self::make(
                    state.top_level,
                    format!(
                        "tensorrt build: {phase} done in {:.1} s",
                        now.saturating_sub(state.started).as_secs_f64()
                    ),
                ))
            }
        }
    }

    fn make(top_level: bool, text: String) -> ProgressLine {
        let level = if top_level {
            log::Level::Info
        } else {
            log::Level::Debug
        };
        ProgressLine { text, level }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    #[test]
    fn every_phase_start_and_finish_is_logged() {
        let mut p = BuildProgress::default();
        let start = p
            .line(
                ProgressEvent::PhaseStart {
                    phase: "Building",
                    parent: Some("Engine"),
                    steps: 40,
                },
                secs(0),
            )
            .expect("start line");
        let start = start.text;
        assert!(start.contains("Building") && start.contains("in Engine") && start.contains("40"));
        let done = p
            .line(ProgressEvent::PhaseFinish { phase: "Building" }, secs(95))
            .expect("finish line");
        assert!(done.text.contains("95.0 s"), "{done:?}");
        assert_eq!(done.level, log::Level::Debug);
        assert!(p
            .line(ProgressEvent::PhaseFinish { phase: "Building" }, secs(96))
            .is_none());
    }

    #[test]
    fn step_lines_are_throttled_per_phase() {
        let mut p = BuildProgress::default();
        p.line(
            ProgressEvent::PhaseStart {
                phase: "A",
                parent: None,
                steps: 100,
            },
            secs(0),
        );
        let step = |p: &mut BuildProgress, n, t| {
            p.line(
                ProgressEvent::Step {
                    phase: "A",
                    step: n,
                },
                secs(t),
            )
        };
        assert!(step(&mut p, 0, 1).is_none());
        assert!(step(&mut p, 9, 9).is_none());
        let line = step(&mut p, 49, 12).expect("after the throttle").text;
        assert!(line.contains("50/100") && line.contains("50%"), "{line}");
        assert!(step(&mut p, 50, 13).is_none());
        // a step of an unknown phase is ignored, not a panic
        assert!(p
            .line(
                ProgressEvent::Step {
                    phase: "B",
                    step: 0
                },
                secs(99)
            )
            .is_none());
    }

    #[test]
    fn nested_phases_are_debug_and_steps_share_one_throttle() {
        let mut p = BuildProgress::default();
        let start = |p: &mut BuildProgress, phase, parent, t| {
            p.line(
                ProgressEvent::PhaseStart {
                    phase,
                    parent,
                    steps: 4,
                },
                secs(t),
            )
        };
        let top = start(&mut p, "Top", None, 0).expect("top start");
        assert_eq!(top.level, log::Level::Info);
        for t in 1..200 {
            let nested = start(&mut p, "Tactic", Some("Top"), t).expect("nested start");
            assert_eq!(nested.level, log::Level::Debug);
            let step = ProgressEvent::Step {
                phase: "Tactic",
                step: 0,
            };
            assert!(p.line(step, secs(t)).is_none(), "nested steps stay silent");
            let done = p
                .line(ProgressEvent::PhaseFinish { phase: "Tactic" }, secs(t))
                .expect("nested done");
            assert_eq!(done.level, log::Level::Debug);
        }
        let mut info_steps = 0;
        for t in 0..200 {
            let step = ProgressEvent::Step {
                phase: "Top",
                step: 1,
            };
            if p.line(step, secs(t)).is_some() {
                info_steps += 1;
            }
        }
        assert_eq!(info_steps, 19, "one top-level step line per 10 s");
    }
}
