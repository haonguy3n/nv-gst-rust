//! `MappedTensor`. Role: Adapter from "host-visible `f32` memory" to the
//! core's `TensorBuffer`. Pattern: the memory is a small trait (`F32Memory`)
//! so the CUDA mapped buffer plugs in on the device and a plain `Vec` in the
//! PC test.

use edge_core::TensorBuffer;
use std::any::Any;

/// Zeroed `f32` memory the CPU can read and write (D3: mapped host memory
/// on the device). Dropping it frees the allocation.
pub trait F32Memory: Send + Sync {
    fn as_slice(&self) -> &[f32];
    fn as_mut_slice(&mut self) -> &mut [f32];
}

/// Mapped host memory as a tensor's backing store.
pub struct MappedTensor {
    memory: Box<dyn F32Memory>,
}

impl MappedTensor {
    pub fn new(memory: impl F32Memory + 'static) -> Self {
        Self {
            memory: Box::new(memory),
        }
    }
}

impl TensorBuffer for MappedTensor {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_f32_slice(&self) -> Option<&[f32]> {
        Some(self.memory.as_slice())
    }

    fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        Some(self.memory.as_mut_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trt_logic::tensor_spec;
    use edge_core::Tensor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Stands in for `cudaHostAlloc`: a zeroed `Vec` that counts its frees.
    struct FakeAlloc {
        data: Vec<f32>,
        live: Arc<AtomicUsize>,
    }

    impl FakeAlloc {
        fn alloc(len: usize, live: &Arc<AtomicUsize>) -> Self {
            live.fetch_add(1, Ordering::SeqCst);
            Self {
                data: vec![0.0; len],
                live: Arc::clone(live),
            }
        }
    }

    impl F32Memory for FakeAlloc {
        fn as_slice(&self) -> &[f32] {
            &self.data
        }

        fn as_mut_slice(&mut self) -> &mut [f32] {
            &mut self.data
        }
    }

    impl Drop for FakeAlloc {
        fn drop(&mut self) {
            self.live.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn tensor(shape: &[i64], live: &Arc<AtomicUsize>) -> Tensor {
        let spec = tensor_spec("t", trt_sys::DType::F32, shape).expect("spec");
        let len = spec.num_elements();
        Tensor::new(
            spec,
            Box::new(MappedTensor::new(FakeAlloc::alloc(len, live))),
        )
    }

    #[test]
    fn length_follows_the_shape_and_starts_zeroed() {
        let live = Arc::new(AtomicUsize::new(0));
        let t = tensor(&[1, 3, 4, 5], &live);
        let view = t.as_f32_slice().expect("host view");
        assert_eq!(view.len(), 60);
        assert!(view.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn writes_are_seen_through_the_same_memory() {
        let live = Arc::new(AtomicUsize::new(0));
        let mut t = tensor(&[2, 2], &live);
        t.as_f32_slice_mut().expect("mut view")[3] = 7.5;
        assert_eq!(t.as_f32_slice().expect("view"), &[0.0, 0.0, 0.0, 7.5]);
        let any = t.buffer_mut().as_any_mut();
        assert!(any.downcast_mut::<MappedTensor>().is_some());
    }

    #[test]
    fn dropping_the_tensor_frees_the_allocation_once() {
        let live = Arc::new(AtomicUsize::new(0));
        let a = tensor(&[1, 8], &live);
        let b = tensor(&[1, 8], &live);
        assert_eq!(live.load(Ordering::SeqCst), 2);
        drop(a);
        assert_eq!(live.load(Ordering::SeqCst), 1);
        drop(b);
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }
}
