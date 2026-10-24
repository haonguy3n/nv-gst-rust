//! A named background thread that produces one value, polled without
//! blocking. Role: the engine-warm mechanism of D6 ("video never waits
//! for the plan"), generic so the PC tests it without TensorRT.
//!
//! Also holds the process-wide build lock: two cameras warming at once
//! must not both build the same plan.

use edge_core::{Error, Result};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};

static BUILD_LOCK: Mutex<()> = Mutex::new(());

/// Held while a plan is loaded or built. A poisoned lock is still a lock:
/// the panic that poisoned it was already reported by its job.
pub fn build_lock() -> MutexGuard<'static, ()> {
    BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The thread name of the plan warm-up.
pub const THREAD_NAME: &str = "engine-warm";

enum Delivery {
    Waiting,
    Taken,
}

pub struct BackgroundJob<T> {
    rx: Receiver<Result<T>>,
    cancel: Arc<AtomicBool>,
    state: Delivery,
}

impl<T: Send + 'static> BackgroundJob<T> {
    /// Starts @p work on a thread called @p name. The closure gets the
    /// cancel flag; it is set when the job is dropped, and a long build
    /// should stop when it sees it. A panic in @p work is an `Err`.
    pub fn spawn<F>(name: &str, work: F) -> Result<Self>
    where
        F: FnOnce(Arc<AtomicBool>) -> Result<T> + Send + 'static,
    {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancel);
        let thread_name = name.to_string();
        std::thread::Builder::new()
            .name(name.to_string())
            .spawn(move || {
                let outcome = catch_unwind(AssertUnwindSafe(|| work(flag))).unwrap_or_else(|_| {
                    Err(Error::Platform(format!("thread `{thread_name}` panicked")))
                });
                // The receiver is gone when the job was dropped: nothing to tell.
                let _ = tx.send(outcome);
            })
            .map_err(|e| Error::Platform(format!("cannot start thread `{name}`: {e}")))?;
        Ok(Self {
            rx,
            cancel,
            state: Delivery::Waiting,
        })
    }

    /// `Ok(None)` while the work runs, `Ok(Some(v))` once, `Err` when it
    /// failed (or when the value was taken already).
    pub fn poll(&mut self) -> Result<Option<T>> {
        if matches!(self.state, Delivery::Taken) {
            return Err(Error::Platform(
                "the warm-up result was taken already".into(),
            ));
        }
        match self.rx.try_recv() {
            Ok(outcome) => {
                self.state = Delivery::Taken;
                outcome.map(Some)
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => {
                self.state = Delivery::Taken;
                Err(Error::Platform(
                    "the warm-up thread ended without a result".into(),
                ))
            }
        }
    }
}

impl<T> Drop for BackgroundJob<T> {
    fn drop(&mut self) {
        // The thread is detached (a build cannot be joined for minutes); it
        // sees the flag at its next progress step and stops.
        self.cancel.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait<T: Send + 'static>(job: &mut BackgroundJob<T>) -> Result<T> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(v) = job.poll()? {
                return Ok(v);
            }
            assert!(Instant::now() < deadline, "job did not finish");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn delivers_once_and_runs_on_the_named_thread() {
        let mut job = BackgroundJob::spawn(THREAD_NAME, |_| {
            Ok(std::thread::current().name().map(str::to_string))
        })
        .expect("spawn");
        assert_eq!(
            wait(&mut job).expect("value").as_deref(),
            Some("engine-warm")
        );
        assert!(
            job.poll().is_err(),
            "second poll after delivery is an error"
        );
    }

    #[test]
    fn polling_does_not_block_while_the_work_runs() {
        let (release, gate) = channel::<()>();
        let mut job = BackgroundJob::spawn("test-warm", move |_| {
            let _ = gate.recv();
            Ok(7_u32)
        })
        .expect("spawn");
        assert!(matches!(job.poll(), Ok(None)));
        release.send(()).expect("release");
        assert_eq!(wait(&mut job).expect("value"), 7);
    }

    #[test]
    fn failure_and_panic_surface_from_the_poll() {
        let mut failing: BackgroundJob<u32> =
            BackgroundJob::spawn("test-warm", |_| Err(Error::Platform("build failed".into())))
                .expect("spawn");
        let err = wait(&mut failing).expect_err("failure");
        assert!(err.to_string().contains("build failed"), "{err}");

        let mut panicking: BackgroundJob<u32> =
            BackgroundJob::spawn("test-warm", |_| panic!("boom")).expect("spawn");
        let err = wait(&mut panicking).expect_err("panic");
        assert!(err.to_string().contains("panicked"), "{err}");
    }

    #[test]
    fn dropping_the_job_sets_the_cancel_flag() {
        let (tx, rx) = channel::<Arc<AtomicBool>>();
        let (release, gate) = channel::<()>();
        let job = BackgroundJob::spawn("test-warm", move |cancel| {
            tx.send(cancel).expect("send flag");
            let _ = gate.recv();
            Ok(())
        })
        .expect("spawn");
        let flag = rx.recv_timeout(Duration::from_secs(5)).expect("flag");
        assert!(!flag.load(Ordering::Acquire));
        drop(job);
        assert!(flag.load(Ordering::Acquire));
        let _ = release.send(());
    }

    #[test]
    fn build_lock_survives_a_poisoning_panic() {
        let _ = std::thread::spawn(|| {
            let _g = build_lock();
            panic!("poison");
        })
        .join();
        drop(build_lock());
    }
}
