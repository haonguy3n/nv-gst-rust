//! `TrtEngine`: one TensorRT execution context on one camera's stream.
//! Role: Adapter from `trt_sys` to `InferenceEngine`.
//!
//! Every I/O tensor lives in mapped host memory (D3) bound once at load:
//! batch 1, F32 only, static shapes only. `infer` enqueues and
//! synchronizes; after the first good inference the enqueue is captured
//! into a CUDA graph and replayed (T10), with one fallback to plain
//! `enqueueV3` if the capture fails. The decisions are in `trt_logic`.
//!
//! A failed enqueue or synchronize marks the engine broken (CUDA errors
//! are sticky): every later `infer` returns `Err`, so the runner stops
//! instead of reading stale outputs.

use crate::cuda::{platform_err, CudaStream, GpuTimer, MappedTensor};
use crate::trt_logic::{check_io_set, input_shape_fix, tensor_spec, GraphPolicy, InferAction};
use edge_core::{Error, GpuStream, InferenceEngine, Result, Tensor};
use std::sync::Arc;
use std::time::Duration;
use trt_sys::{Context, Graph, IoDesc, IoMode, MappedBuffer, Runtime, Stream};

pub struct TrtEngine {
    // Field order is drop order: the graph before the context it came
    // from, the context before the buffers it reads and writes (and, by
    // `Arc`, before its engine and runtime).
    graph: Option<Graph>,
    context: Context,
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    policy: GraphPolicy,
    /// Used when the caller passes a stream that is not ours (tests).
    fallback: Option<Stream>,
    /// Events around the enqueue; `None` when they could not be created.
    timer: Option<GpuTimer>,
    gpu_time: Option<Duration>,
    broken: Option<String>,
}

impl TrtEngine {
    /// Loads a serialized plan and binds its tensors.
    pub fn from_plan(runtime: &Arc<Runtime>, plan: &[u8], use_graph: bool) -> Result<Self> {
        let engine = trt_sys::Engine::deserialize(runtime, plan).map_err(platform_err)?;
        let ios = engine.io_descs().map_err(platform_err)?;
        check_io_set(&ios)?;
        let mut context = Context::new(&engine).map_err(platform_err)?;
        for io in ios.iter().filter(|io| io.mode == IoMode::Input) {
            if let Some(shape) = input_shape_fix(io)? {
                context
                    .set_input_shape(&io.name, &shape)
                    .map_err(platform_err)?;
            }
        }
        let (mut inputs, mut outputs) = (Vec::new(), Vec::new());
        for io in &ios {
            let tensor = bind_tensor(&mut context, io)?;
            match io.mode {
                IoMode::Input => inputs.push(tensor),
                IoMode::Output => outputs.push(tensor),
            }
        }
        Ok(Self {
            graph: None,
            context,
            inputs,
            outputs,
            policy: GraphPolicy::new(use_graph),
            fallback: None,
            timer: create_timer(),
            gpu_time: None,
            broken: None,
        })
    }

    fn run(&mut self, stream: &GpuStream) -> Result<()> {
        trt_sys::set_device_for_thread().map_err(platform_err)?;
        let stream = pick_stream(stream, &mut self.fallback)?;
        let action = self.policy.action();
        let timer = self.timer.as_ref();
        self.gpu_time = None;
        // A failed event record only loses this frame's GPU time.
        let timed = timer.is_some_and(|t| t.begin(stream).is_ok());
        match (action, &self.graph) {
            (InferAction::Replay, Some(graph)) => graph.launch(stream),
            (InferAction::Replay, None) => {
                return Err(Error::Platform("CUDA graph state lost".into()))
            }
            _ => self.context.enqueue(stream),
        }
        .map_err(platform_err)?;
        let timed = timed && timer.is_some_and(|t| t.finish(stream).is_ok());
        stream.synchronize().map_err(platform_err)?;
        if let (true, Some(timer)) = (timed, timer) {
            self.gpu_time = timer.elapsed().ok();
        }
        if action == InferAction::EnqueueThenCapture {
            // The warm-up pass has finished: its outputs stay in the buffers.
            self.graph = capture_graph(&mut self.context, stream);
            self.policy = self.policy.capture_done(self.graph.is_some());
        }
        Ok(())
    }
}

/// `None` (GPU time off, with a warning) when the events cannot be created.
fn create_timer() -> Option<GpuTimer> {
    match GpuTimer::new() {
        Ok(timer) => Some(timer),
        Err(e) => {
            log::warn!("tensorrt: GPU timing off, cannot create CUDA events: {e}");
            None
        }
    }
}

/// `None` when the capture fails; the plain enqueue is then used for good.
fn capture_graph(context: &mut Context, stream: &Stream) -> Option<Graph> {
    match context.capture_graph(stream) {
        Ok(graph) => {
            log::info!("tensorrt: CUDA graph captured: each inference launches one graph");
            Some(graph)
        }
        Err(e) => {
            log::warn!("tensorrt: CUDA graph capture failed ({e}): running without it");
            None
        }
    }
}

/// The caller's `CudaStream`, else a private one (created once).
fn pick_stream<'a>(stream: &'a GpuStream, fallback: &'a mut Option<Stream>) -> Result<&'a Stream> {
    if let Some(cuda) = stream.backend::<CudaStream>() {
        return Ok(cuda.inner());
    }
    if fallback.is_none() {
        *fallback = Some(Stream::new().map_err(platform_err)?);
    }
    fallback
        .as_ref()
        .ok_or_else(|| Error::Platform("no CUDA stream".into()))
}

/// Allocates the tensor's mapped buffer from the context's shape and binds it.
fn bind_tensor(context: &mut Context, io: &IoDesc) -> Result<Tensor> {
    let shape = context.tensor_shape(&io.name).map_err(platform_err)?;
    let spec = tensor_spec(&io.name, io.dtype, &shape)?;
    let buffer = MappedBuffer::alloc(spec.num_elements()).map_err(platform_err)?;
    context.bind(&io.name, &buffer).map_err(platform_err)?;
    Ok(Tensor::new(spec, Box::new(MappedTensor::new(buffer))))
}

impl InferenceEngine for TrtEngine {
    fn input_bindings(&mut self) -> &mut [Tensor] {
        &mut self.inputs
    }

    fn output_bindings(&self) -> &[Tensor] {
        &self.outputs
    }

    fn infer(&mut self, stream: &GpuStream) -> Result<()> {
        if let Some(why) = &self.broken {
            return Err(Error::Platform(format!(
                "the TensorRT engine failed earlier and is stopped: {why}"
            )));
        }
        let result = self.run(stream);
        if let Err(e) = &result {
            self.broken = Some(e.to_string());
        }
        result
    }

    fn last_gpu_time(&self) -> Option<Duration> {
        self.gpu_time
    }
}
