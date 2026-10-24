// The GPU's planar tensor writer: RGBA bytes of a registered EGL image to a planar float tensor, one thread per
// pixel. The arithmetic is the CPU writer's (nvbuf-sys planar.rs): byte (as f32) times scale, one multiply, so the
// two agree bit for bit. The VIC has already letterboxed and converted the frame; this kernel only rearranges.
// The idea follows camera-viber's NvmmGpuTensorWriter / CudaTensorKernels (same owner).
#include "preprocess_shim.h"

#include <cuda_egl_interop.h>
#include <cuda_runtime_api.h>

#include <exception>
#include <string>

namespace
{

thread_local std::string t_error;

int32_t Fail(char const* what, cudaError_t error)
{
    t_error = std::string(what) + ": " + cudaGetErrorString(error);
    return -1;
}

int32_t FailText(std::string text)
{
    t_error = std::move(text);
    return -1;
}

constexpr int k_blockSide = 16;

__global__ void WriteTensorKernel(
    uint8_t const* pixels, size_t pitch, int width, int height, bool swapRb, float scale, float* out)
{
    int const x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int const y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    uchar4 const px = reinterpret_cast<uchar4 const*>(pixels + (static_cast<size_t>(y) * pitch))[x];
    size_t const plane = static_cast<size_t>(width) * height;
    size_t const at = (static_cast<size_t>(y) * width) + x;
    out[at] = static_cast<float>(swapRb ? px.z : px.x) * scale;
    out[plane + at] = static_cast<float>(px.y) * scale;
    out[(2 * plane) + at] = static_cast<float>(swapRb ? px.x : px.z) * scale;
}

} // namespace

struct CpSource
{
    cudaGraphicsResource_t resource = nullptr;
    uint8_t const* pixels = nullptr; // pitch frame: the mapped surface; array frame: the staging buffer
    size_t pitch = 0;
    cudaArray_t array = nullptr; // array frame only
    uint32_t width = 0;
    uint32_t height = 0;
};

namespace
{

// Fills pixels/pitch (or array + a staging buffer) from the registered frame.
int32_t MapFrame(CpSource& s)
{
    cudaEglFrame frame {};
    if (cudaError_t const e = cudaGraphicsResourceGetMappedEglFrame(&frame, s.resource, 0, 0); e != cudaSuccess)
    {
        return Fail("cudaGraphicsResourceGetMappedEglFrame", e);
    }
    if (frame.planeCount < 1 || frame.planeDesc[0].numChannels != 4)
    {
        return FailText("the EGL frame is not a 4-channel RGBA plane");
    }
    if (frame.planeDesc[0].width < s.width || frame.planeDesc[0].height < s.height)
    {
        return FailText("the EGL frame is smaller than the surface");
    }
    size_t const rowBytes = static_cast<size_t>(s.width) * 4;
    if (frame.frameType == cudaEglFrameTypePitch)
    {
        s.pixels = static_cast<uint8_t const*>(frame.frame.pPitch[0].ptr);
        s.pitch = frame.frame.pPitch[0].pitch;
        if (s.pixels == nullptr || s.pitch < rowBytes || s.pitch % 4 != 0)
        {
            return FailText("the pitch frame has a null pointer or an unusable pitch");
        }
        return 0;
    }
    s.array = frame.frame.pArray[0];
    void* staging = nullptr;
    if (cudaError_t const e = cudaMallocPitch(&staging, &s.pitch, rowBytes, s.height); e != cudaSuccess)
    {
        return Fail("cudaMallocPitch", e);
    }
    s.pixels = static_cast<uint8_t const*>(staging);
    return 0;
}

} // namespace

extern "C"
{

char const* cp_last_error(void)
{
    return t_error.c_str();
}

CpSource* cp_source_register(void* eglImage, uint32_t width, uint32_t height)
{
    try
    {
        if (eglImage == nullptr || width == 0 || height == 0)
        {
            FailText("cp_source_register: null image or empty size");
            return nullptr;
        }
        auto* source = new CpSource;
        source->width = width;
        source->height = height;
        cudaError_t const e = cudaGraphicsEGLRegisterImage(
            &source->resource, static_cast<EGLImageKHR>(eglImage), cudaGraphicsRegisterFlagsReadOnly);
        if (e != cudaSuccess)
        {
            Fail("cudaGraphicsEGLRegisterImage", e);
            source->resource = nullptr;
            delete source;
            return nullptr;
        }
        if (MapFrame(*source) < 0)
        {
            cp_source_destroy(source);
            return nullptr;
        }
        return source;
    }
    catch (std::exception const& e)
    {
        t_error = e.what();
    }
    catch (...)
    {
        t_error = "unknown C++ exception";
    }
    return nullptr;
}

void cp_source_destroy(CpSource* source)
{
    if (source == nullptr)
    {
        return;
    }
    if (source->array != nullptr && source->pixels != nullptr)
    {
        cudaFree(const_cast<uint8_t*>(source->pixels));
    }
    if (source->resource != nullptr)
    {
        cudaGraphicsUnregisterResource(source->resource);
    }
    delete source;
}

int32_t cp_write_tensor(CpSource const* source, int32_t swapRb, float scale, void* tensorHost, void* stream)
{
    if (source == nullptr || tensorHost == nullptr)
    {
        return FailText("cp_write_tensor: null source or tensor");
    }
    auto const cudaStream = static_cast<cudaStream_t>(stream);
    if (source->array != nullptr)
    {
        // The array frame is copied into the staging buffer first (unverified on JetPack 7: the VIC surface is
        // expected to be a pitch frame).
        if (cudaError_t const e = cudaMemcpy2DFromArrayAsync(
                const_cast<uint8_t*>(source->pixels), source->pitch, source->array, 0, 0,
                static_cast<size_t>(source->width) * 4, source->height, cudaMemcpyDeviceToDevice, cudaStream);
            e != cudaSuccess)
        {
            return Fail("cudaMemcpy2DFromArrayAsync", e);
        }
    }
    void* device = nullptr;
    if (cudaError_t const e = cudaHostGetDevicePointer(&device, tensorHost, 0); e != cudaSuccess)
    {
        return Fail("cudaHostGetDevicePointer", e);
    }
    dim3 const block(k_blockSide, k_blockSide);
    dim3 const grid((source->width + k_blockSide - 1) / k_blockSide, (source->height + k_blockSide - 1) / k_blockSide);
    WriteTensorKernel<<<grid, block, 0, cudaStream>>>(
        source->pixels, source->pitch, static_cast<int>(source->width), static_cast<int>(source->height),
        swapRb != 0, scale, static_cast<float*>(device));
    if (cudaError_t const e = cudaGetLastError(); e != cudaSuccess)
    {
        return Fail("the kernel launch", e);
    }
    return 0;
}

} // extern "C"
