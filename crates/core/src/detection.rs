use serde::Serialize;

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
#[derive(Debug, Default, Serialize)]
pub struct Detections {
    pub frame_seq: u64,
    pub items: Vec<Detection>,
}

impl Detections {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frame_seq: 0,
            items: Vec::with_capacity(capacity),
        }
    }

    pub fn clear(&mut self, frame_seq: u64) {
        self.frame_seq = frame_seq;
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
