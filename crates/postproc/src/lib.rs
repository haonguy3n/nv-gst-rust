//! Platform-neutral post-processing: decoders read only the small
//! output tensors, in place, and fill a reused `Detections` buffer.

#![forbid(unsafe_code)]

mod nms;
mod yolo;
mod yolox;

pub use nms::{iou, nms, nms_agnostic};
pub use yolo::YoloPostProcessor;
pub use yolox::YoloxPostProcessor;
