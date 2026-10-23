//! Bounded channel with drop-oldest backpressure.
//!
//! When the sink is slow the oldest queued frame is dropped and
//! counted; the capture thread never blocks and the sensor is never
//! stalled. Only handles to hardware buffers move through here.
//!
//! Hand-rolled instead of `crossbeam_channel` because drop-oldest
//! here must *return* the evicted frame to the sender (so its pool
//! buffer is released on the capture thread) and expose the drop
//! count without a receiver; no off-the-shelf bounded channel
//! offers either.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

struct State<T> {
    queue: VecDeque<T>,
    closed: bool,
}

struct Shared<T> {
    state: Mutex<State<T>>,
    available: Condvar,
    capacity: usize,
    dropped: Arc<AtomicU64>,
}

pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>, Arc<AtomicU64>) {
    let dropped = Arc::new(AtomicU64::new(0));
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            queue: VecDeque::with_capacity(capacity),
            closed: false,
        }),
        available: Condvar::new(),
        capacity: capacity.max(1),
        dropped: Arc::clone(&dropped),
    });
    (
        Sender {
            shared: Arc::clone(&shared),
        },
        Receiver { shared },
        dropped,
    )
}

pub struct Sender<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Sender<T> {
    /// Never blocks. Drops (and returns) the oldest queued item when full.
    pub fn send(&self, item: T) -> Option<T> {
        let mut state = self.shared.state.lock().expect("channel mutex poisoned");
        let evicted = if state.queue.len() >= self.shared.capacity {
            self.shared.dropped.fetch_add(1, Ordering::Relaxed);
            state.queue.pop_front()
        } else {
            None
        };
        state.queue.push_back(item);
        drop(state);
        self.shared.available.notify_one();
        evicted
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.shared.state.lock() {
            state.closed = true;
            drop(state);
            self.shared.available.notify_all();
        }
    }
}

pub struct Receiver<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Receiver<T> {
    /// Blocks up to `timeout`. `None` on timeout, or when the sender
    /// is gone and the queue is drained.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
        // One deadline for the whole wait: a spurious wakeup waits
        // only for the time remaining, never a fresh full timeout.
        let deadline = Instant::now() + timeout;
        let mut state = self.shared.state.lock().expect("channel mutex poisoned");
        loop {
            if let Some(item) = state.queue.pop_front() {
                return Some(item);
            }
            if state.closed {
                return None;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let (guard, _wait) = self
                .shared
                .available
                .wait_timeout(state, remaining)
                .expect("channel mutex poisoned");
            state = guard;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.shared
            .state
            .lock()
            .expect("channel mutex poisoned")
            .queue
            .is_empty()
    }

    pub fn is_closed_and_empty(&self) -> bool {
        let state = self.shared.state.lock().expect("channel mutex poisoned");
        state.closed && state.queue.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recv_timeout_respects_its_deadline() {
        let (_tx, rx, _dropped) = channel::<u32>(2);
        let started = Instant::now();
        assert_eq!(rx.recv_timeout(Duration::from_millis(50)), None);
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(40),
            "returned early: {elapsed:?}"
        );
        assert!(elapsed < Duration::from_millis(500), "overran: {elapsed:?}");
    }

    #[test]
    fn drop_oldest_returns_evicted_and_counts() {
        let (tx, rx, dropped) = channel::<u32>(2);
        assert_eq!(tx.send(1), None);
        assert_eq!(tx.send(2), None);
        assert_eq!(tx.send(3), Some(1));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(rx.recv_timeout(Duration::ZERO), Some(2));
        assert_eq!(rx.recv_timeout(Duration::ZERO), Some(3));
    }
}
