//! Steady-state allocation gate for the decoders (design:
//! "a counting global allocator in tests fails the build if
//! steady-state allocations are above 0"). Scoped to decode():
//! the decoders are the per-frame CPU work under our control on
//! every platform; capture/engine allocation behaviour is a
//! per-backend property checked on device.
//!
//! Counting is thread-local and opt-in per test so the harness's
//! own allocations never pollute the measurement.

use edge_core::{
    DType, Detections, FrameMeta, Layout, PixelFormat, PostProcessor, Tensor, TensorBuffer,
    TensorSpec,
};
use edge_postproc::{YoloPostProcessor, YoloxPostProcessor};
use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::any::Any;
use std::cell::Cell;
use std::time::Duration;

thread_local! {
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        let _ = ENABLED.try_with(|e| {
            if e.get() {
                COUNT.with(|c| c.set(c.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn measured<R>(f: impl FnOnce() -> R) -> (R, usize) {
    COUNT.with(|c| c.set(0));
    ENABLED.with(|e| e.set(true));
    let r = f();
    ENABLED.with(|e| e.set(false));
    (r, COUNT.with(|c| c.get()))
}

struct TestBuffer(Vec<f32>);
impl TensorBuffer for TestBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_f32_slice(&self) -> Option<&[f32]> {
        Some(&self.0)
    }
}

fn tensor(name: &str, shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::new(
        TensorSpec::new(name, shape, DType::F32),
        Box::new(TestBuffer(data)),
    )
}

fn meta() -> FrameMeta {
    FrameMeta {
        seq: 0,
        pts: Duration::ZERO,
        width: 640,
        height: 640,
        format: PixelFormat::Nv12,
        layout: Layout::PitchLinear,
        camera_id: 0,
        dmabuf_fd: -1,
    }
}

/// 200 candidates, half above threshold with overlapping boxes so
/// NMS does real work.
fn yolo_output(candidates: usize, classes: usize) -> Tensor {
    let channels = 4 + classes;
    let mut data = vec![0.0f32; channels * candidates];
    for i in 0..candidates {
        let at = |ch: usize| ch * candidates + i;
        data[at(0)] = 100.0 + (i % 50) as f32;
        data[at(1)] = 100.0 + (i % 40) as f32;
        data[at(2)] = 60.0;
        data[at(3)] = 60.0;
        data[at(4 + (i % classes))] = if i % 2 == 0 { 0.9 } else { 0.1 };
    }
    tensor("output0", vec![1, channels, candidates], data)
}

#[test]
fn yolo_decode_allocates_nothing_in_steady_state() {
    let outputs = [yolo_output(200, 3)];
    let mut pp = YoloPostProcessor::new(640, 640, 3, 0.25);
    let mut out = Detections::with_capacity(64);
    // Warm up: first calls grow the scratch and output buffers.
    for _ in 0..3 {
        pp.decode(&outputs, &meta(), &mut out).unwrap();
    }
    assert!(!out.is_empty());
    let (_, allocs) = measured(|| {
        for _ in 0..100 {
            pp.decode(&outputs, &meta(), &mut out).unwrap();
        }
    });
    assert_eq!(allocs, 0, "steady-state decode must not allocate");
}

#[test]
fn yolox_decode_allocates_nothing_in_steady_state() {
    let anchors = 200;
    let mut boxes = Vec::with_capacity(anchors * 4);
    let mut objectness = Vec::with_capacity(anchors);
    let mut scores = Vec::with_capacity(anchors * 3);
    for i in 0..anchors {
        boxes.extend_from_slice(&[100.0 + (i % 50) as f32, 100.0, 160.0, 160.0]);
        objectness.push(0.9);
        scores.extend_from_slice(&[0.9, 0.1, 0.1]);
    }
    let outputs = [
        tensor("boxes", vec![1, anchors, 4], boxes),
        tensor("objectness", vec![1, anchors, 1], objectness),
        tensor("scores", vec![1, anchors, 3], scores),
    ];
    let mut pp = YoloxPostProcessor::new(640, 640, 3, 0.3);
    let mut out = Detections::with_capacity(64);
    for _ in 0..3 {
        pp.decode(&outputs, &meta(), &mut out).unwrap();
    }
    assert!(!out.is_empty());
    let (_, allocs) = measured(|| {
        for _ in 0..100 {
            pp.decode(&outputs, &meta(), &mut out).unwrap();
        }
    });
    assert_eq!(allocs, 0, "steady-state decode must not allocate");
}
