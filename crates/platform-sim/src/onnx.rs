//! ONNX Runtime engine for the simulator backend (`--features onnx`).
//!
//! The preprocessor writes into this engine's host F32 input binding
//! exactly as it does for the deterministic engine; `infer` hands
//! that buffer to ONNX Runtime and publishes the model's outputs as
//! host-visible tensors for post-processing. The copies at this
//! boundary are a simulator-only concession — on Jetson the same
//! trait is backed by TensorRT bindings in device memory.

use crate::HostTensor;
use edge_core::{
    DType, Error, GpuStream, InferenceEngine, ModelConfig, Result, Tensor, TensorSpec,
};
use ort::session::Session;
use std::path::Path;

pub struct OnnxEngine {
    session: Session,
    // Declared after `session` so the session drops first.
    _env: ort::environment::Environment,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    output_names: Vec<String>,
    input_shape: [usize; 4],
}

impl OnnxEngine {
    pub fn load(path: &Path, cfg: &ModelConfig) -> Result<Self> {
        let env = ort::init()
            .with_name("edge-vision-sim")
            .build()
            .map_err(|e| Error::Platform(format!("onnx runtime init: {e}")))?;
        let session = Session::builder(&env)
            .and_then(|mut b| b.commit_from_file(path))
            .map_err(|e| Error::Platform(format!("loading ONNX model {}: {e}", path.display())))?;

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        if output_names.is_empty() {
            return Err(Error::Platform(format!(
                "ONNX model {} declares no outputs",
                path.display()
            )));
        }

        let (w, h) = (cfg.input_width as usize, cfg.input_height as usize);
        // Check the model's declared input against the config now,
        // not at the first infer: a static dim that disagrees (or
        // a non-F32 input the preprocessor cannot write) is a
        // load-time error. Dynamic dims (-1) are accepted.
        let input_name = match session.inputs().first() {
            Some(declared) => {
                let dtype = declared.dtype();
                if let Some(elem) = dtype.tensor_type() {
                    if elem != ort::value::TensorElementType::Float32 {
                        return Err(Error::Platform(format!(
                            "ONNX model {} input `{}` is {elem:?}, expected Float32",
                            path.display(),
                            declared.name()
                        )));
                    }
                }
                if let Some(shape) = dtype.tensor_shape() {
                    let expected = [1i64, 3, h as i64, w as i64];
                    let mismatch = shape.len() != expected.len()
                        || shape
                            .iter()
                            .zip(expected.iter())
                            .any(|(&got, &want)| got > 0 && got != want);
                    if mismatch {
                        return Err(Error::Platform(format!(
                            "ONNX model {} input `{}` shape {shape} does not match \
                             config [1, 3, {h}, {w}]",
                            path.display(),
                            declared.name()
                        )));
                    }
                }
                declared.name().to_string()
            }
            None => {
                return Err(Error::Platform(format!(
                    "ONNX model {} declares no inputs",
                    path.display()
                )));
            }
        };
        let inputs = vec![Tensor::new(
            TensorSpec::new(input_name, vec![1, 3, h, w], DType::F32),
            HostTensor::zeros(3 * h * w),
        )];
        // Preallocate outputs from their declared shapes when
        // static; dynamic outputs grow into place on first infer
        // and are reused from then on.
        let outputs = session
            .outputs()
            .iter()
            .map(|o| {
                let (shape, elems) = match o.dtype().tensor_shape() {
                    Some(shape) if shape.iter().all(|&d| d > 0) => (
                        shape.iter().map(|&d| d as usize).collect::<Vec<usize>>(),
                        shape.num_elements(),
                    ),
                    _ => (Vec::new(), 0),
                };
                Tensor::new(
                    TensorSpec::new(o.name().to_string(), shape, DType::F32),
                    HostTensor::zeros(elems),
                )
            })
            .collect();

        Ok(Self {
            session,
            _env: env,
            inputs,
            outputs,
            output_names,
            input_shape: [1, 3, h, w],
        })
    }
}

impl InferenceEngine for OnnxEngine {
    fn input_bindings(&mut self) -> &mut [Tensor] {
        &mut self.inputs
    }

    fn output_bindings(&self) -> &[Tensor] {
        &self.outputs
    }

    fn infer(&mut self, _stream: &GpuStream) -> Result<()> {
        // Zero-copy view over the input binding: no per-frame
        // clone of the (multi-MB) input plane.
        let data = self.inputs[0]
            .as_f32_slice()
            .ok_or_else(|| Error::Platform("onnx input binding is not host F32".into()))?;
        let input = ort::value::TensorRef::from_array_view((self.input_shape, data))
            .map_err(|e| Error::Platform(format!("onnx input tensor: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![input])
            .map_err(|e| Error::Platform(format!("onnx inference: {e}")))?;

        // Copy into the preallocated output bindings in place;
        // only the first frame (or a shape change) can grow them.
        for (i, (_name, value)) in outputs.iter().enumerate() {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Platform(format!("onnx output {i}: {e}")))?;
            let shape: Vec<usize> = shape
                .iter()
                .map(|&d| {
                    usize::try_from(d)
                        .map_err(|_| Error::Platform(format!("onnx output {i}: negative dim {d}")))
                })
                .collect::<Result<_>>()?;
            if i >= self.outputs.len() {
                let name = self
                    .output_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("output{i}"));
                self.outputs.push(Tensor::new(
                    TensorSpec::new(name, Vec::new(), DType::F32),
                    HostTensor::zeros(0),
                ));
            }
            let tensor = &mut self.outputs[i];
            tensor.spec.shape = shape;
            match tensor.buf.as_any_mut().downcast_mut::<HostTensor>() {
                Some(host) => host.replace(data),
                None => tensor.buf = HostTensor::from_vec(data.to_vec()),
            }
        }
        self.outputs.truncate(outputs.len());
        Ok(())
    }
}
