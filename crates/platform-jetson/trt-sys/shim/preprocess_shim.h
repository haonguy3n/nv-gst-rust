/* C ABI of the CUDA preprocessor (shim/preprocess_kernels.cu), built with
 * the `cuda-preprocess` feature. Rust declares the same functions by hand
 * (src/ffi_preprocess.rs); keep the two in step.
 *
 * Same rules as trt_shim.h: no exception escapes; a failure returns null or
 * a negative status and leaves a message for cp_last_error() on the calling
 * thread. */
#ifndef PREPROCESS_SHIM_H
#define PREPROCESS_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CpSource CpSource;

const char* cp_last_error(void);

/* Registers an EGLImageKHR (NvBufSurfaceMapEglImage of a pitch-linear RGBA
 * surface of width x height) with CUDA, read-only, and maps its frame. A
 * pitch frame is read in place; an array frame is copied to a staging buffer
 * each time. Null on failure. The image must outlive the source. */
CpSource* cp_source_register(void* egl_image, uint32_t width, uint32_t height);
/* Unregisters. Call before the EGL image is unmapped. */
void cp_source_destroy(CpSource* source);

/* Queues, on @p stream (a cudaStream_t), the planar write of the source's
 * pixels into @p tensor_host (a cudaHostAlloc(mapped) allocation of at least
 * 3 * width * height floats; its device alias is looked up here):
 * plane c = byte * scale, with plane 0 = B and plane 2 = R when swap_rb. */
int32_t cp_write_tensor(const CpSource* source, int32_t swap_rb, float scale, void* tensor_host, void* stream);

#ifdef __cplusplus
}
#endif

#endif
