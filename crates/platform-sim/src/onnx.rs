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
        let inputs = vec![Tensor::new(
            TensorSpec::new("images", vec![1, 3, h, w], DType::F32),
            HostTensor::zeros(3 * h * w),
        )];
        // Real shapes are published after the first inference.
        let outputs = output_names
            .iter()
            .map(|name| {
                Tensor::new(
                    TensorSpec::new(name.clone(), Vec::new(), DType::F32),
                    HostTensor::zeros(0),
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
        let data: Vec<f32> = self.inputs[0]
            .as_f32_slice()
            .ok_or_else(|| Error::Platform("onnx input binding is not host F32".into()))?
            .to_vec();
        let input = ort::value::Tensor::from_array((self.input_shape, data))
            .map_err(|e| Error::Platform(format!("onnx input tensor: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![input])
            .map_err(|e| Error::Platform(format!("onnx inference: {e}")))?;

        let mut published = Vec::with_capacity(outputs.len());
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
            let name = self
                .output_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("output{i}"));
            published.push(Tensor::new(
                TensorSpec::new(name, shape, DType::F32),
                HostTensor::from_vec(data.to_vec()),
            ));
        }
        self.outputs = published;
        Ok(())
    }
}
