/* C ABI over TensorRT and the CUDA runtime. Rust declares the same
 * functions by hand (src/ffi.rs); keep the two in step.
 *
 * Rules: every function catches every C++ exception; a failure returns
 * null / a negative status and leaves a message for trt_last_error() on the
 * calling thread. Handles are opaque. Destroy order, enforced by the Rust
 * wrappers: graph and context, then engine, then runtime. */
#ifndef TRT_SHIM_H
#define TRT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TrtRuntime TrtRuntime;
typedef struct TrtEngine TrtEngine;
typedef struct TrtContext TrtContext;
typedef struct TrtGraph TrtGraph;
typedef struct TrtPlan TrtPlan;

/* severity: nvinfer1::ILogger::Severity (0 internal error .. 4 verbose). */
typedef void (*TrtLogFn)(int32_t severity, const char* message);

/* event: 0 phase start (steps = total), 1 step complete (step, steps = 0),
 * 2 phase finish. parent is null for a top-level phase. Return 0 to cancel
 * the build (only honoured on event 1), nonzero to continue. */
typedef int32_t (*TrtProgressFn)(
    void* user, int32_t event, const char* phase, const char* parent, int32_t step, int32_t steps);

/* io mode codes: 0 input, 1 output. dtype codes: 0 f32, 1 f16, 2 i8, 3 i32,
 * 4 bool, 5 u8, 6 i64, 7 bf16, 99 other. */

void trt_set_log_callback(TrtLogFn fn);
const char* trt_last_error(void);
int32_t trt_version(void);

/* Device 0: compute capability, CUDA runtime version (e.g. 13020), name. */
int32_t trt_device_info(int32_t* sm_major, int32_t* sm_minor, int32_t* cuda_runtime, char* name, size_t name_cap);
/* cudaSetDevice(device) once per calling thread. */
int32_t trt_cuda_set_device(int32_t device);

TrtRuntime* trt_runtime_create(void);
void trt_runtime_destroy(TrtRuntime* runtime);

TrtEngine* trt_engine_deserialize(TrtRuntime* runtime, const void* blob, size_t size);
void trt_engine_destroy(TrtEngine* engine);
int32_t trt_engine_nb_io(const TrtEngine* engine);
const char* trt_engine_io_name(const TrtEngine* engine, int32_t index);
int32_t trt_engine_io_mode(const TrtEngine* engine, const char* name);
int32_t trt_engine_io_dtype(const TrtEngine* engine, const char* name);
/* The engine's shape (-1 where dynamic); returns the rank, or -1. */
int32_t trt_engine_io_shape(const TrtEngine* engine, const char* name, int64_t* dims, int32_t cap);

TrtContext* trt_context_create(TrtEngine* engine);
void trt_context_destroy(TrtContext* context);
int32_t trt_context_set_input_shape(TrtContext* context, const char* name, const int64_t* dims, int32_t rank);
/* The context's shape, with dynamic dimensions resolved; returns the rank, or -1. */
int32_t trt_context_tensor_shape(const TrtContext* context, const char* name, int64_t* dims, int32_t cap);
int32_t trt_context_set_tensor_address(TrtContext* context, const char* name, void* device_ptr);
int32_t trt_context_enqueue(TrtContext* context, void* stream);

/* Captures one enqueue into a CUDA graph. Null on failure; the stream is
 * then usable again. */
TrtGraph* trt_graph_capture(TrtContext* context, void* stream);
int32_t trt_graph_launch(TrtGraph* graph, void* stream);
void trt_graph_destroy(TrtGraph* graph);

void* trt_stream_create(void);
void trt_stream_destroy(void* stream);
int32_t trt_stream_synchronize(void* stream);

/* CUDA events with timing, for GPU time. record queues on the stream;
 * elapsed_ms is valid once the stream has synchronized past @p end. */
void* trt_event_create(void);
void trt_event_destroy(void* event);
int32_t trt_event_record(void* event, void* stream);
int32_t trt_event_elapsed_ms(void* start, void* end, float* ms);

/* cudaHostAlloc(mapped), zeroed; the device pointer is the GPU's view. */
int32_t trt_host_alloc_mapped(size_t bytes, void** host, void** device);
void trt_host_free(void* host);

/* Builds a plan from an ONNX file. Null on failure or cancel. */
TrtPlan* trt_build_plan(
    const char* onnx_path,
    int32_t fp16,
    uint64_t workspace_bytes,
    const char* timing_cache_path,
    TrtProgressFn progress,
    void* user);
const void* trt_plan_data(const TrtPlan* plan);
size_t trt_plan_size(const TrtPlan* plan);
void trt_plan_destroy(TrtPlan* plan);

#ifdef __cplusplus
}
#endif

#endif
