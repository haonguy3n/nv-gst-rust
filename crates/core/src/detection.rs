use crate::frame::FrameMeta;
use serde::{Serialize, Serializer};
use std::time::Duration;

/// Serialise a `Duration` as one nanosecond count: metadata
/// consumers join on a number, not serde's `{secs, nanos}` object.
fn serialize_pts_ns<S: Serializer>(pts: &Duration, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u64(pts.as_nanos() as u64)
}

/// One decoded detection, coordinates normalised to `0.0..=1.0`
/// of the frame width/height.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
    pub class_id: u32,
}

/// Reused across frames: `PostProcessor::decode` clears and refills it
/// so the hot loop does not allocate.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Detections {
    pub frame_seq: u64,
    /// Which camera produced this frame (multi-camera metadata
    /// must be attributable without out-of-band knowledge).
    pub camera_id: u8,
    /// Sensor timestamp of the frame, for joining detections to
    /// encoded packets (which carry the same pts). Serialised as
    /// `pts_ns`, a single nanosecond count.
    #[serde(rename = "pts_ns", serialize_with = "serialize_pts_ns")]
    pub pts: Duration,
    pub items: Vec<Detection>,
}

impl Detections {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frame_seq: 0,
            camera_id: 0,
            pts: Duration::ZERO,
            items: Vec::with_capacity(capacity),
        }
    }

    /// Reset for a new frame, taking attribution from its meta so
    /// a decoder cannot forget it.
    pub fn clear(&mut self, meta: &FrameMeta) {
        self.frame_seq = meta.seq;
        self.camera_id = meta.camera_id;
        self.pts = meta.pts;
        self.items.clear();
    }

    pub fn push(&mut self, detection: Detection) {
        self.items.push(detection);
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
