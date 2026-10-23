use std::any::Any;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I8,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
        }
    }
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn byte_len(&self) -> usize {
        self.num_elements() * self.dtype.size_bytes()
    }
}

/// Backing store for a tensor: a CUDA device allocation on Jetson, an
/// NPU tensor on NXP, host memory in the simulator. Input tensors are
/// written by the preprocessor straight into the engine's binding;
/// output tensors may be host-visible and are read in place.
pub trait TensorBuffer: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Host `f32` view, when this buffer is host-visible F32 memory.
    fn as_f32_slice(&self) -> Option<&[f32]> {
        None
    }
    fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        None
    }
}

/// Model input/output only (small). Same shape as `HwFrame`: a spec
/// plus an opaque buffer handle.
pub struct Tensor {
    pub spec: TensorSpec,
    pub buf: Arc<dyn TensorBuffer>,
}

impl Tensor {
    pub fn new(spec: TensorSpec, buf: Arc<dyn TensorBuffer>) -> Self {
        Self { spec, buf }
    }

    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        self.buf.as_f32_slice()
    }

    /// Mutable host view. Returns `None` when the buffer is shared or
    /// not host-visible; in-place GPU writes go through `as_any_mut`
    /// inside the owning backend instead.
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        Arc::get_mut(&mut self.buf)?.as_f32_slice_mut()
    }

    pub fn buffer_mut(&mut self) -> Option<&mut dyn TensorBuffer> {
        Arc::get_mut(&mut self.buf)
    }
}
