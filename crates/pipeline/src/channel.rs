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
    /// Current queue depth, updated under the lock and readable
    /// from the metrics thread at any time.
    depth: Arc<AtomicU64>,
}

pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>, Arc<AtomicU64>) {
    let dropped = Arc::new(AtomicU64::new(0));
    let depth = Arc::new(AtomicU64::new(0));
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            queue: VecDeque::with_capacity(capacity),
            closed: false,
        }),
        available: Condvar::new(),
        capacity: capacity.max(1),
        dropped: Arc::clone(&dropped),
        depth: Arc::clone(&depth),
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
    /// Never blocks. Drops (and returns) the oldest queued item when
    /// full. `Err(item)` means the receiver is gone: a consumer has
    /// died, and the caller must stop instead of dropping the frame
    /// silently.
    pub fn send(&self, item: T) -> std::result::Result<Option<T>, T> {
        let mut state = self.shared.state.lock().expect("channel mutex poisoned");
        if state.closed {
            return Err(item);
        }
        let evicted = if state.queue.len() >= self.shared.capacity {
            self.shared.dropped.fetch_add(1, Ordering::Relaxed);
            state.queue.pop_front()
        } else {
            None
        };
        state.queue.push_back(item);
        // Published under the lock: the metrics reader can only
        // ever observe a depth that existed.
        self.shared
            .depth
            .store(state.queue.len() as u64, Ordering::Relaxed);
        drop(state);
        self.shared.available.notify_one();
        Ok(evicted)
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

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.shared.state.lock() {
            state.closed = true;
            drop(state);
            self.shared.available.notify_all();
        }
    }
}

impl<T> Receiver<T> {
    /// Live queue depth, for the metrics exporter.
    pub fn depth_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.shared.depth)
    }

    /// Blocks up to `timeout`. `None` on timeout, or when the sender
    /// is gone and the queue is drained.
    pub fn recv_timeout(&self, timeout: Duration) -> Option<T> {
        // One deadline for the whole wait: a spurious wakeup waits
        // only for the time remaining, never a fresh full timeout.
        let deadline = Instant::now() + timeout;
        let mut state = self.shared.state.lock().expect("channel mutex poisoned");
        loop {
            if let Some(item) = state.queue.pop_front() {
                self.shared
                    .depth
                    .store(state.queue.len() as u64, Ordering::Relaxed);
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
        assert!(tx.send(1).unwrap().is_none());
        assert!(tx.send(2).unwrap().is_none());
        assert_eq!(tx.send(3).unwrap(), Some(1));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(rx.recv_timeout(Duration::ZERO), Some(2));
        assert_eq!(rx.recv_timeout(Duration::ZERO), Some(3));
    }

    #[test]
    fn send_fails_when_the_receiver_is_gone() {
        let (tx, rx, _dropped) = channel::<u32>(2);
        drop(rx);
        assert_eq!(tx.send(7).unwrap_err(), 7);
    }

    #[test]
    fn depth_handle_tracks_the_queue() {
        let (tx, rx, _dropped) = channel::<u32>(3);
        let depth = rx.depth_handle();
        assert_eq!(depth.load(Ordering::Relaxed), 0);
        assert!(tx.send(1).unwrap().is_none());
        assert!(tx.send(2).unwrap().is_none());
        assert_eq!(depth.load(Ordering::Relaxed), 2);
        assert_eq!(rx.recv_timeout(Duration::ZERO), Some(1));
        assert_eq!(depth.load(Ordering::Relaxed), 1);
        // Drop-oldest keeps depth at capacity, not capacity + 1.
        assert!(tx.send(3).unwrap().is_none());
        assert!(tx.send(4).unwrap().is_none());
        assert_eq!(depth.load(Ordering::Relaxed), 3);
        assert_eq!(tx.send(5).unwrap(), Some(2));
        assert_eq!(depth.load(Ordering::Relaxed), 3);
    }
}
