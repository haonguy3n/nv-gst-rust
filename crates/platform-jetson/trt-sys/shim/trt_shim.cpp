// C ABI over TensorRT 10 and the CUDA runtime (see trt_shim.h). Ported in
// behaviour from the owner's camera-viber TensorRtEngine / BuildProgressAdapter.
// No exception leaves this file: every entry point goes through Guarded().

#include "trt_shim.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

struct TrtRuntime
{
    std::unique_ptr<nvinfer1::IRuntime> p;
};
struct TrtEngine
{
    std::unique_ptr<nvinfer1::ICudaEngine> p;
};
struct TrtContext
{
    std::unique_ptr<nvinfer1::IExecutionContext> p;
};
struct TrtGraph
{
    cudaGraphExec_t exec;
};
struct TrtPlan
{
    std::unique_ptr<nvinfer1::IHostMemory> p;
};

namespace
{

thread_local std::string t_error;
std::atomic<TrtLogFn> g_logFn {nullptr};

void SetError(std::string const& message) noexcept
{
    try
    {
        t_error = message;
    }
    catch (...)
    {
    }
}

/// Runs @p f; an exception becomes the thread's error text and @p failure.
template <class R, class F>
R Guarded(R failure, F&& f) noexcept
{
    try
    {
        return f();
    }
    catch (std::exception const& e)
    {
        SetError(std::string("tensorrt shim: exception: ") + e.what());
    }
    catch (...)
    {
        SetError("tensorrt shim: unknown exception");
    }
    return failure;
}

/// Records a CUDA failure and clears the sticky last-error slot, so the
/// next unrelated CUDA call on this thread does not read it back.
int32_t CudaFail(char const* what, cudaError_t error)
{
    (void) cudaGetLastError();
    SetError(std::string("cuda: ") + what + ": " + cudaGetErrorString(error));
    return -1;
}

/// TensorRT's log into the Rust callback. Alive for the whole process, as
/// TensorRT requires of its logger.
class Logger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* message) noexcept override
    {
        if (TrtLogFn fn = g_logFn.load(std::memory_order_acquire))
        {
            fn(static_cast<int32_t>(severity), message);
        }
    }
};

Logger& TrtLogger()
{
    static Logger s_logger;
    return s_logger;
}

int32_t DTypeCode(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
        return 0;
    case nvinfer1::DataType::kHALF:
        return 1;
    case nvinfer1::DataType::kINT8:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 3;
    case nvinfer1::DataType::kBOOL:
        return 4;
    case nvinfer1::DataType::kUINT8:
        return 5;
    case nvinfer1::DataType::kINT64:
        return 6;
    case nvinfer1::DataType::kBF16:
        return 7;
    default:
        return 99;
    }
}

int32_t CopyDims(nvinfer1::Dims const& dims, int64_t* out, int32_t cap)
{
    if (dims.nbDims < 0 || dims.nbDims > cap || (dims.nbDims > 0 && out == nullptr))
    {
        SetError("tensorrt: tensor rank does not fit the caller's buffer");
        return -1;
    }
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        out[i] = dims.d[i];
    }
    return dims.nbDims;
}

/// Reads a whole file; false when it is missing or unreadable.
bool ReadFile(std::string const& path, std::vector<char>& bytes)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        return false;
    }
    std::streamoff const size = in.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) > (std::uint64_t {4} << 30U))
    {
        return false;
    }
    bytes.resize(static_cast<std::size_t>(size));
    in.seekg(0);
    return static_cast<bool>(in.read(bytes.data(), size));
}

/// Written under a temporary name and renamed: a crash never leaves a torn file.
bool WriteFileAtomic(std::string const& path, void const* data, std::size_t size)
{
    std::string const tmp = path + ".part";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out || !out.write(static_cast<char const*>(data), static_cast<std::streamsize>(size)))
        {
            return false;
        }
    }
    std::error_code error;
    std::filesystem::rename(tmp, path, error);
    if (error)
    {
        std::filesystem::remove(tmp, error);
        return false;
    }
    return true;
}

/// Forwards TensorRT's build progress to the Rust callback.
class ProgressForwarder final : public nvinfer1::IProgressMonitor
{
public:
    ProgressForwarder(TrtProgressFn fn, void* user)
        : m_fn(fn)
        , m_user(user)
    {
    }

    void phaseStart(char const* phase, char const* parent, int32_t steps) noexcept override
    {
        if (m_fn != nullptr)
        {
            m_fn(m_user, 0, phase, parent, 0, steps);
        }
    }

    bool stepComplete(char const* phase, int32_t step) noexcept override
    {
        if (m_fn == nullptr)
        {
            return true;
        }
        bool const bContinue = m_fn(m_user, 1, phase, nullptr, step, 0) != 0;
        m_bCancelled = m_bCancelled || !bContinue;
        return bContinue;
    }

    void phaseFinish(char const* phase) noexcept override
    {
        if (m_fn != nullptr)
        {
            m_fn(m_user, 2, phase, nullptr, 0, 0);
        }
    }

    bool Cancelled() const
    {
        return m_bCancelled;
    }

private:
    TrtProgressFn m_fn;
    void* m_user;
    bool m_bCancelled {false};
};

/// One timing cache attached to @p config from @p path: an empty one when the
/// file is missing or TensorRT refuses it. Null when TensorRT takes none.
std::unique_ptr<nvinfer1::ITimingCache> AttachTimingCache(nvinfer1::IBuilderConfig& config, std::string const& path)
{
    std::vector<char> bytes;
    bool const bHave = !path.empty() && ReadFile(path, bytes);
    std::unique_ptr<nvinfer1::ITimingCache> cache(
        config.createTimingCache(bHave ? bytes.data() : nullptr, bHave ? bytes.size() : 0));
    if (cache && config.setTimingCache(*cache, false))
    {
        return cache;
    }
    cache.reset(config.createTimingCache(nullptr, 0));
    if (cache && config.setTimingCache(*cache, false))
    {
        return cache;
    }
    return nullptr;
}

/// One optimization profile pinning a dynamic batch (dimension 0) of every
/// input to 1. Another dynamic dimension is an error: returns false.
bool AddBatchProfile(
    nvinfer1::IBuilder& builder, nvinfer1::INetworkDefinition const& network, nvinfer1::IBuilderConfig& config)
{
    nvinfer1::IOptimizationProfile* profile = nullptr; // owned by the builder
    for (int32_t n = 0; n < network.getNbInputs(); ++n)
    {
        nvinfer1::ITensor const& input = *network.getInput(n);
        nvinfer1::Dims dims = input.getDimensions();
        bool bDynamic = false;
        for (int32_t i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] < 0 && i != 0)
            {
                SetError(
                    std::string("tensorrt: input '") + input.getName() + "' has a dynamic dimension other than the batch");
                return false;
            }
            bDynamic = bDynamic || dims.d[i] < 0;
            dims.d[i] = dims.d[i] < 0 ? 1 : dims.d[i];
        }
        if (!bDynamic)
        {
            continue;
        }
        profile = profile != nullptr ? profile : builder.createOptimizationProfile();
        for (auto const selector : {nvinfer1::OptProfileSelector::kMIN,
                                    nvinfer1::OptProfileSelector::kOPT,
                                    nvinfer1::OptProfileSelector::kMAX})
        {
            profile->setDimensions(input.getName(), selector, dims);
        }
    }
    if (profile != nullptr)
    {
        config.addOptimizationProfile(profile);
    }
    return true;
}

TrtPlan* BuildPlan(
    char const* onnxPath, int32_t fp16, uint64_t workspace, char const* timingCachePath, TrtProgressFn fn, void* user)
{
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(TrtLogger()));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder ? builder->createNetworkV2(0U) : nullptr);
    if (!network)
    {
        SetError("tensorrt: cannot create a builder");
        return nullptr;
    }
    // Declared after the network: the parser must go first.
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, TrtLogger()));
    if (!parser || !parser->parseFromFile(onnxPath, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        char const* why = parser && parser->getNbErrors() > 0 ? parser->getError(0)->desc() : "unknown error";
        SetError(std::string("tensorrt: cannot parse ") + onnxPath + ": " + why);
        return nullptr;
    }
    // Declared before the config: the monitor and the cache outlive it.
    ProgressForwarder progress(fn, user);
    std::unique_ptr<nvinfer1::ITimingCache> timingCache;
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config)
    {
        SetError("tensorrt: cannot create a builder config");
        return nullptr;
    }
    config->setProgressMonitor(&progress);
    // The default workspace is all device memory, which on the Orin is all RAM.
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, static_cast<std::size_t>(workspace));
    if (fp16 != 0)
    {
        // kFP16 is deprecated from TensorRT 10.12 but still honoured in 10.16;
        // the replacement (strong typing) needs an FP16 model.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
    }
    std::string const cachePath = timingCachePath != nullptr ? timingCachePath : "";
    timingCache = AttachTimingCache(*config, cachePath);
    if (!AddBatchProfile(*builder, *network, *config))
    {
        return nullptr;
    }
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        SetError(progress.Cancelled() ? "tensorrt: the engine build was cancelled" : std::string("tensorrt: building an engine for ") + onnxPath + " failed");
        return nullptr;
    }
    if (timingCache && !cachePath.empty())
    {
        std::unique_ptr<nvinfer1::IHostMemory> const blob(timingCache->serialize());
        if (blob == nullptr || !WriteFileAtomic(cachePath, blob->data(), blob->size()))
        {
            // Best effort: only the next build's speed-up is lost.
            TrtLogger().log(nvinfer1::ILogger::Severity::kWARNING, "cannot save the timing cache");
        }
    }
    return new TrtPlan {std::move(plan)};
}

} // namespace

extern "C" {

void trt_set_log_callback(TrtLogFn fn)
{
    g_logFn.store(fn, std::memory_order_release);
}

char const* trt_last_error(void)
{
    return t_error.c_str();
}

int32_t trt_version(void)
{
    return getInferLibVersion();
}

int32_t trt_device_info(int32_t* smMajor, int32_t* smMinor, int32_t* cudaRuntime, char* name, size_t nameCap)
{
    return Guarded(-1, [&]() -> int32_t {
        int major = 0;
        int minor = 0;
        int runtime = 0;
        cudaDeviceProp props {};
        cudaError_t error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
        error = error == cudaSuccess ? cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) : error;
        error = error == cudaSuccess ? cudaRuntimeGetVersion(&runtime) : error;
        error = error == cudaSuccess ? cudaGetDeviceProperties(&props, 0) : error;
        if (error != cudaSuccess)
        {
            return CudaFail("device info", error);
        }
        *smMajor = major;
        *smMinor = minor;
        *cudaRuntime = runtime;
        if (nameCap > 0)
        {
            std::snprintf(name, nameCap, "%s", props.name);
        }
        return 0;
    });
}

int32_t trt_cuda_set_device(int32_t device)
{
    return Guarded(-1, [&]() -> int32_t {
        thread_local int32_t t_device = -1;
        if (t_device == device)
        {
            return 0;
        }
        if (cudaError_t const error = cudaSetDevice(device); error != cudaSuccess)
        {
            return CudaFail("cudaSetDevice", error);
        }
        t_device = device;
        return 0;
    });
}

TrtRuntime* trt_runtime_create(void)
{
    return Guarded<TrtRuntime*>(nullptr, [&]() -> TrtRuntime* {
        std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(TrtLogger()));
        if (!runtime)
        {
            SetError("tensorrt: cannot create a runtime");
            return nullptr;
        }
        return new TrtRuntime {std::move(runtime)};
    });
}

void trt_runtime_destroy(TrtRuntime* runtime)
{
    Guarded(0, [&] {
        delete runtime;
        return 0;
    });
}

TrtEngine* trt_engine_deserialize(TrtRuntime* runtime, void const* blob, size_t size)
{
    return Guarded<TrtEngine*>(nullptr, [&]() -> TrtEngine* {
        std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->p->deserializeCudaEngine(blob, size));
        if (!engine)
        {
            SetError("tensorrt: the plan does not load (wrong TensorRT version, device or a damaged file)");
            return nullptr;
        }
        return new TrtEngine {std::move(engine)};
    });
}

void trt_engine_destroy(TrtEngine* engine)
{
    Guarded(0, [&] {
        delete engine;
        return 0;
    });
}

int32_t trt_engine_nb_io(TrtEngine const* engine)
{
    return Guarded(-1, [&] { return engine->p->getNbIOTensors(); });
}

char const* trt_engine_io_name(TrtEngine const* engine, int32_t index)
{
    return Guarded<char const*>(nullptr, [&] { return engine->p->getIOTensorName(index); });
}

int32_t trt_engine_io_mode(TrtEngine const* engine, char const* name)
{
    return Guarded(-1, [&]() -> int32_t {
        switch (engine->p->getTensorIOMode(name))
        {
        case nvinfer1::TensorIOMode::kINPUT:
            return 0;
        case nvinfer1::TensorIOMode::kOUTPUT:
            return 1;
        default:
            return -1;
        }
    });
}

int32_t trt_engine_io_dtype(TrtEngine const* engine, char const* name)
{
    return Guarded(-1, [&] { return DTypeCode(engine->p->getTensorDataType(name)); });
}

int32_t trt_engine_io_shape(TrtEngine const* engine, char const* name, int64_t* dims, int32_t cap)
{
    return Guarded(-1, [&] { return CopyDims(engine->p->getTensorShape(name), dims, cap); });
}

TrtContext* trt_context_create(TrtEngine* engine)
{
    return Guarded<TrtContext*>(nullptr, [&]() -> TrtContext* {
        std::unique_ptr<nvinfer1::IExecutionContext> context(engine->p->createExecutionContext());
        if (!context)
        {
            SetError("tensorrt: cannot create an execution context");
            return nullptr;
        }
        return new TrtContext {std::move(context)};
    });
}

void trt_context_destroy(TrtContext* context)
{
    Guarded(0, [&] {
        delete context;
        return 0;
    });
}

int32_t trt_context_set_input_shape(TrtContext* context, char const* name, int64_t const* dims, int32_t rank)
{
    return Guarded(-1, [&]() -> int32_t {
        nvinfer1::Dims shape {};
        if (rank < 0 || rank > nvinfer1::Dims::MAX_DIMS)
        {
            SetError("tensorrt: input rank out of range");
            return -1;
        }
        shape.nbDims = rank;
        for (int32_t i = 0; i < rank; ++i)
        {
            shape.d[i] = dims[i];
        }
        if (!context->p->setInputShape(name, shape))
        {
            SetError(std::string("tensorrt: cannot set the shape of input '") + name + "'");
            return -1;
        }
        return 0;
    });
}

int32_t trt_context_tensor_shape(TrtContext const* context, char const* name, int64_t* dims, int32_t cap)
{
    return Guarded(-1, [&] { return CopyDims(context->p->getTensorShape(name), dims, cap); });
}

int32_t trt_context_set_tensor_address(TrtContext* context, char const* name, void* devicePtr)
{
    return Guarded(-1, [&]() -> int32_t {
        if (!context->p->setTensorAddress(name, devicePtr))
        {
            SetError(std::string("tensorrt: cannot bind tensor '") + name + "'");
            return -1;
        }
        return 0;
    });
}

int32_t trt_context_enqueue(TrtContext* context, void* stream)
{
    return Guarded(-1, [&]() -> int32_t {
        if (!context->p->enqueueV3(static_cast<cudaStream_t>(stream)))
        {
            SetError("tensorrt: enqueueV3 failed");
            return -1;
        }
        return 0;
    });
}

TrtGraph* trt_graph_capture(TrtContext* context, void* stream)
{
    return Guarded<TrtGraph*>(nullptr, [&]() -> TrtGraph* {
        auto const cudaStream = static_cast<cudaStream_t>(stream);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        // Thread-local: another camera's CUDA calls on its own thread must not
        // invalidate this capture.
        cudaError_t error = cudaStreamBeginCapture(cudaStream, cudaStreamCaptureModeThreadLocal);
        if (error != cudaSuccess)
        {
            CudaFail("cudaStreamBeginCapture", error);
            return nullptr;
        }
        bool const bEnqueued = context->p->enqueueV3(cudaStream);
        error = cudaStreamEndCapture(cudaStream, &graph); // a capture that began is always ended
        if (error == cudaSuccess)
        {
            error = bEnqueued && graph != nullptr ? cudaGraphInstantiate(&exec, graph, 0) : cudaErrorStreamCaptureInvalidated;
        }
        if (graph != nullptr)
        {
            cudaGraphDestroy(graph);
        }
        if (error != cudaSuccess)
        {
            CudaFail("graph capture", error);
            return nullptr;
        }
        return new TrtGraph {exec};
    });
}

int32_t trt_graph_launch(TrtGraph* graph, void* stream)
{
    return Guarded(-1, [&]() -> int32_t {
        cudaError_t const error = cudaGraphLaunch(graph->exec, static_cast<cudaStream_t>(stream));
        return error == cudaSuccess ? 0 : CudaFail("cudaGraphLaunch", error);
    });
}

void trt_graph_destroy(TrtGraph* graph)
{
    Guarded(0, [&] {
        if (graph != nullptr)
        {
            cudaGraphExecDestroy(graph->exec);
            delete graph;
        }
        return 0;
    });
}

void* trt_stream_create(void)
{
    return Guarded<void*>(nullptr, [&]() -> void* {
        cudaStream_t stream = nullptr;
        if (cudaError_t const error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); error != cudaSuccess)
        {
            CudaFail("cudaStreamCreateWithFlags", error);
            return nullptr;
        }
        return stream;
    });
}

void trt_stream_destroy(void* stream)
{
    Guarded(0, [&] {
        if (stream != nullptr)
        {
            cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        }
        return 0;
    });
}

int32_t trt_stream_synchronize(void* stream)
{
    return Guarded(-1, [&]() -> int32_t {
        cudaError_t const error = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        return error == cudaSuccess ? 0 : CudaFail("cudaStreamSynchronize", error);
    });
}

void* trt_event_create(void)
{
    return Guarded<void*>(nullptr, [&]() -> void* {
        cudaEvent_t event = nullptr;
        if (cudaError_t const error = cudaEventCreate(&event); error != cudaSuccess)
        {
            CudaFail("cudaEventCreate", error);
            return nullptr;
        }
        return event;
    });
}

void trt_event_destroy(void* event)
{
    Guarded(0, [&] {
        if (event != nullptr)
        {
            cudaEventDestroy(static_cast<cudaEvent_t>(event));
        }
        return 0;
    });
}

int32_t trt_event_record(void* event, void* stream)
{
    return Guarded(-1, [&]() -> int32_t {
        cudaError_t const error = cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream));
        return error == cudaSuccess ? 0 : CudaFail("cudaEventRecord", error);
    });
}

int32_t trt_event_elapsed_ms(void* start, void* end, float* ms)
{
    return Guarded(-1, [&]() -> int32_t {
        cudaError_t const error = cudaEventElapsedTime(ms, static_cast<cudaEvent_t>(start), static_cast<cudaEvent_t>(end));
        return error == cudaSuccess ? 0 : CudaFail("cudaEventElapsedTime", error);
    });
}

int32_t trt_host_alloc_mapped(size_t bytes, void** host, void** device)
{
    return Guarded(-1, [&]() -> int32_t {
        void* pHost = nullptr;
        void* pDevice = nullptr;
        if (cudaError_t const error = cudaHostAlloc(&pHost, bytes, cudaHostAllocMapped); error != cudaSuccess)
        {
            return CudaFail("cudaHostAlloc", error);
        }
        if (cudaError_t const error = cudaHostGetDevicePointer(&pDevice, pHost, 0); error != cudaSuccess)
        {
            cudaFreeHost(pHost);
            return CudaFail("cudaHostGetDevicePointer", error);
        }
        std::memset(pHost, 0, bytes);
        *host = pHost;
        *device = pDevice;
        return 0;
    });
}

void trt_host_free(void* host)
{
    Guarded(0, [&] {
        if (host != nullptr)
        {
            cudaFreeHost(host);
        }
        return 0;
    });
}

TrtPlan* trt_build_plan(
    char const* onnxPath,
    int32_t fp16,
    uint64_t workspaceBytes,
    char const* timingCachePath,
    TrtProgressFn progress,
    void* user)
{
    return Guarded<TrtPlan*>(nullptr, [&] { return BuildPlan(onnxPath, fp16, workspaceBytes, timingCachePath, progress, user); });
}

void const* trt_plan_data(TrtPlan const* plan)
{
    return plan->p->data();
}

size_t trt_plan_size(TrtPlan const* plan)
{
    return plan->p->size();
}

void trt_plan_destroy(TrtPlan* plan)
{
    Guarded(0, [&] {
        delete plan;
        return 0;
    });
}

} // extern "C"
