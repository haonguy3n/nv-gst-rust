//! Bounded channel with drop-oldest backpressure.
//!
//! When the sink is slow the oldest queued frame is dropped and
//! counted; the capture thread never blocks and the sensor is never
//! stalled. Only handles to hardware buffers move through here.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

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
        let mut state = self.shared.state.lock().expect("channel mutex poisoned");
        loop {
            if let Some(item) = state.queue.pop_front() {
                return Some(item);
            }
            if state.closed {
                return None;
            }
            let (guard, wait) = self
                .shared
                .available
                .wait_timeout(state, timeout)
                .expect("channel mutex poisoned");
            state = guard;
            if wait.timed_out() && state.queue.is_empty() {
                return None;
            }
        }
    }

    pub fn is_closed_and_empty(&self) -> bool {
        let state = self.shared.state.lock().expect("channel mutex poisoned");
        state.closed && state.queue.is_empty()
    }
}
