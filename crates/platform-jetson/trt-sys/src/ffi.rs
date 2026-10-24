//! Hand-written declarations of `shim/trt_shim.h`. Keep the two in step.

use std::os::raw::{c_char, c_void};

macro_rules! opaque {
    ($($name:ident),*) => {$(
        #[repr(C)]
        pub struct $name {
            _private: [u8; 0],
        }
    )*};
}
opaque!(TrtRuntime, TrtEngine, TrtContext, TrtGraph, TrtPlan);

pub type TrtLogFn = extern "C" fn(severity: i32, message: *const c_char);
pub type TrtProgressFn = extern "C" fn(
    user: *mut c_void,
    event: i32,
    phase: *const c_char,
    parent: *const c_char,
    step: i32,
    steps: i32,
) -> i32;

extern "C" {
    pub fn trt_set_log_callback(fun: Option<TrtLogFn>);
    pub fn trt_last_error() -> *const c_char;
    pub fn trt_version() -> i32;
    pub fn trt_device_info(
        sm_major: *mut i32,
        sm_minor: *mut i32,
        cuda_runtime: *mut i32,
        name: *mut c_char,
        name_cap: usize,
    ) -> i32;
    pub fn trt_cuda_set_device(device: i32) -> i32;

    pub fn trt_runtime_create() -> *mut TrtRuntime;
    pub fn trt_runtime_destroy(runtime: *mut TrtRuntime);

    pub fn trt_engine_deserialize(
        runtime: *mut TrtRuntime,
        blob: *const c_void,
        size: usize,
    ) -> *mut TrtEngine;
    pub fn trt_engine_destroy(engine: *mut TrtEngine);
    pub fn trt_engine_nb_io(engine: *const TrtEngine) -> i32;
    pub fn trt_engine_io_name(engine: *const TrtEngine, index: i32) -> *const c_char;
    pub fn trt_engine_io_mode(engine: *const TrtEngine, name: *const c_char) -> i32;
    pub fn trt_engine_io_dtype(engine: *const TrtEngine, name: *const c_char) -> i32;
    pub fn trt_engine_io_shape(
        engine: *const TrtEngine,
        name: *const c_char,
        dims: *mut i64,
        cap: i32,
    ) -> i32;

    pub fn trt_context_create(engine: *mut TrtEngine) -> *mut TrtContext;
    pub fn trt_context_destroy(context: *mut TrtContext);
    pub fn trt_context_set_input_shape(
        context: *mut TrtContext,
        name: *const c_char,
        dims: *const i64,
        rank: i32,
    ) -> i32;
    pub fn trt_context_tensor_shape(
        context: *const TrtContext,
        name: *const c_char,
        dims: *mut i64,
        cap: i32,
    ) -> i32;
    pub fn trt_context_set_tensor_address(
        context: *mut TrtContext,
        name: *const c_char,
        device_ptr: *mut c_void,
    ) -> i32;
    pub fn trt_context_enqueue(context: *mut TrtContext, stream: *mut c_void) -> i32;

    pub fn trt_graph_capture(context: *mut TrtContext, stream: *mut c_void) -> *mut TrtGraph;
    pub fn trt_graph_launch(graph: *mut TrtGraph, stream: *mut c_void) -> i32;
    pub fn trt_graph_destroy(graph: *mut TrtGraph);

    pub fn trt_stream_create() -> *mut c_void;
    pub fn trt_stream_destroy(stream: *mut c_void);
    pub fn trt_stream_synchronize(stream: *mut c_void) -> i32;

    pub fn trt_host_alloc_mapped(
        bytes: usize,
        host: *mut *mut c_void,
        device: *mut *mut c_void,
    ) -> i32;
    pub fn trt_host_free(host: *mut c_void);

    pub fn trt_build_plan(
        onnx_path: *const c_char,
        fp16: i32,
        workspace_bytes: u64,
        timing_cache_path: *const c_char,
        progress: Option<TrtProgressFn>,
        user: *mut c_void,
    ) -> *mut TrtPlan;
    pub fn trt_plan_data(plan: *const TrtPlan) -> *const c_void;
    pub fn trt_plan_size(plan: *const TrtPlan) -> usize;
    pub fn trt_plan_destroy(plan: *mut TrtPlan);
}
