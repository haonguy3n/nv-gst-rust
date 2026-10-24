//! Safe wrappers over the C shim. Raw pointers never leave this file.

use crate::ffi;
use crate::types::{
    severity_to_level, BuildOptions, DType, DeviceInfo, Error, IoDesc, IoMode, ProgressEvent,
    Result,
};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::os::unix::ffi::OsStrExt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Arc, Once};

/// Longest tensor rank the shim copies (TensorRT's `Dims::MAX_DIMS`).
const MAX_RANK: usize = 8;

/// The shim's message for the failure that just happened on this thread.
fn last_error() -> Error {
    // SAFETY: the shim always returns a valid NUL-terminated string that
    // stays valid until the next shim call on this thread.
    let text = unsafe { CStr::from_ptr(ffi::trt_last_error()) };
    Error::new(text.to_string_lossy().into_owned())
}

fn status(code: i32) -> Result<()> {
    if code < 0 {
        Err(last_error())
    } else {
        Ok(())
    }
}

fn cstring(text: &str) -> Result<CString> {
    CString::new(text).map_err(|_| Error::new(format!("`{text}` contains a NUL byte")))
}

/// Forwards TensorRT's log to the `log` crate. Must not unwind into C++.
extern "C" fn log_trampoline(severity: i32, message: *const c_char) {
    let _ = catch_unwind(|| {
        if message.is_null() {
            return;
        }
        // SAFETY: TensorRT passes a NUL-terminated message for this call.
        let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
        log::log!(target: "tensorrt", severity_to_level(severity), "{}", text.trim_end());
    });
}

fn init() {
    static INIT: Once = Once::new();
    // SAFETY: registers a plain function that never unwinds.
    INIT.call_once(|| unsafe { ffi::trt_set_log_callback(Some(log_trampoline)) });
}

pub fn version() -> i32 {
    // SAFETY: no arguments.
    unsafe { ffi::trt_version() }
}

/// Device 0 properties for the plan key.
pub fn device_info() -> Result<DeviceInfo> {
    let (mut major, mut minor, mut cuda) = (0i32, 0i32, 0i32);
    let mut name = [0 as c_char; 256];
    // SAFETY: every pointer is a live local; the shim writes at most
    // `name.len()` bytes into `name`.
    let code = unsafe {
        ffi::trt_device_info(
            &mut major,
            &mut minor,
            &mut cuda,
            name.as_mut_ptr(),
            name.len(),
        )
    };
    status(code)?;
    // SAFETY: the shim wrote a NUL-terminated string within the array.
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    Ok(DeviceInfo {
        sm: (major.max(0) as u32, minor.max(0) as u32),
        cuda_runtime: cuda.max(0) as u32,
        name,
    })
}

/// `cudaSetDevice(0)` once per calling thread; call it on every thread that
/// creates or enqueues GPU work.
pub fn set_device_for_thread() -> Result<()> {
    // SAFETY: no pointers.
    status(unsafe { ffi::trt_cuda_set_device(0) })
}

pub struct Runtime {
    raw: NonNull<ffi::TrtRuntime>,
}

// SAFETY: IRuntime may be used from several threads (TensorRT's rules).
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    pub fn new() -> Result<Arc<Self>> {
        init();
        set_device_for_thread()?;
        // SAFETY: no arguments; null is handled.
        let raw = NonNull::new(unsafe { ffi::trt_runtime_create() }).ok_or_else(last_error)?;
        Ok(Arc::new(Self { raw }))
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: `raw` is live and no `Engine` outlives us (they hold an `Arc`).
        unsafe { ffi::trt_runtime_destroy(self.raw.as_ptr()) }
    }
}

pub struct Engine {
    raw: NonNull<ffi::TrtEngine>,
    // Dropped after `raw` is destroyed in `drop`: the runtime goes last.
    _runtime: Arc<Runtime>,
}

// SAFETY: ICudaEngine is thread-safe for the const queries used here.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    pub fn deserialize(runtime: &Arc<Runtime>, plan: &[u8]) -> Result<Arc<Self>> {
        set_device_for_thread()?;
        // SAFETY: the pointer/length pair is a live slice; TensorRT copies it.
        let raw = NonNull::new(unsafe {
            ffi::trt_engine_deserialize(runtime.raw.as_ptr(), plan.as_ptr().cast(), plan.len())
        })
        .ok_or_else(last_error)?;
        Ok(Arc::new(Self {
            raw,
            _runtime: Arc::clone(runtime),
        }))
    }

    /// Every I/O tensor with the engine's own shape (`-1` = dynamic).
    pub fn io_descs(&self) -> Result<Vec<IoDesc>> {
        // SAFETY: `raw` is live.
        let count = unsafe { ffi::trt_engine_nb_io(self.raw.as_ptr()) };
        status(count)?;
        (0..count).map(|i| self.io_desc(i)).collect()
    }

    fn io_desc(&self, index: i32) -> Result<IoDesc> {
        // SAFETY: `raw` is live; the name is valid while the engine is.
        let name_ptr = unsafe { ffi::trt_engine_io_name(self.raw.as_ptr(), index) };
        if name_ptr.is_null() {
            return Err(last_error());
        }
        // SAFETY: non-null, NUL-terminated, owned by the engine.
        let cname = unsafe { CStr::from_ptr(name_ptr) };
        let mut dims = [0i64; MAX_RANK];
        // SAFETY: `cname` is live; `dims` holds `MAX_RANK` entries.
        let (mode, dtype, rank) = unsafe {
            (
                ffi::trt_engine_io_mode(self.raw.as_ptr(), cname.as_ptr()),
                ffi::trt_engine_io_dtype(self.raw.as_ptr(), cname.as_ptr()),
                ffi::trt_engine_io_shape(
                    self.raw.as_ptr(),
                    cname.as_ptr(),
                    dims.as_mut_ptr(),
                    MAX_RANK as i32,
                ),
            )
        };
        status(rank)?;
        let name = cname.to_string_lossy().into_owned();
        Ok(IoDesc {
            mode: IoMode::from_code(mode)
                .ok_or_else(|| Error::new(format!("tensor `{name}` has an unknown I/O mode")))?,
            dtype: DType::from_code(dtype),
            shape: dims[..rank as usize].to_vec(),
            name,
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: `raw` is live; no `Context` outlives us (they hold an `Arc`).
        unsafe { ffi::trt_engine_destroy(self.raw.as_ptr()) }
    }
}

pub struct Context {
    raw: NonNull<ffi::TrtContext>,
    // Released after the context is destroyed in `drop`.
    engine: Arc<Engine>,
}

// SAFETY: one thread uses an IExecutionContext at a time; `&mut self` on
// every mutating call, and no `Sync`.
unsafe impl Send for Context {}

impl Context {
    pub fn new(engine: &Arc<Engine>) -> Result<Self> {
        set_device_for_thread()?;
        // SAFETY: `engine.raw` is live.
        let raw = NonNull::new(unsafe { ffi::trt_context_create(engine.raw.as_ptr()) })
            .ok_or_else(last_error)?;
        Ok(Self {
            raw,
            engine: Arc::clone(engine),
        })
    }

    pub fn engine(&self) -> &Arc<Engine> {
        &self.engine
    }

    pub fn set_input_shape(&mut self, name: &str, dims: &[i64]) -> Result<()> {
        let cname = cstring(name)?;
        // SAFETY: live context; the slice is live for the call.
        status(unsafe {
            ffi::trt_context_set_input_shape(
                self.raw.as_ptr(),
                cname.as_ptr(),
                dims.as_ptr(),
                dims.len() as i32,
            )
        })
    }

    /// The context's shape of a tensor, dynamic dimensions resolved.
    pub fn tensor_shape(&self, name: &str) -> Result<Vec<i64>> {
        let cname = cstring(name)?;
        let mut dims = [0i64; MAX_RANK];
        // SAFETY: live context; `dims` holds `MAX_RANK` entries.
        let rank = unsafe {
            ffi::trt_context_tensor_shape(
                self.raw.as_ptr(),
                cname.as_ptr(),
                dims.as_mut_ptr(),
                MAX_RANK as i32,
            )
        };
        status(rank)?;
        Ok(dims[..rank as usize].to_vec())
    }

    /// Binds a tensor to a mapped buffer's device address. The buffer must
    /// stay alive and unmoved in memory (its allocation never moves) for as
    /// long as this context, and any graph captured from it, is used.
    pub fn bind(&mut self, name: &str, buffer: &MappedBuffer) -> Result<()> {
        let cname = cstring(name)?;
        // SAFETY: live context; the address stays valid per the contract above.
        status(unsafe {
            ffi::trt_context_set_tensor_address(
                self.raw.as_ptr(),
                cname.as_ptr(),
                buffer.device.as_ptr(),
            )
        })
    }

    pub fn enqueue(&mut self, stream: &Stream) -> Result<()> {
        // SAFETY: live context and stream.
        status(unsafe { ffi::trt_context_enqueue(self.raw.as_ptr(), stream.raw.as_ptr()) })
    }

    /// Captures one enqueue into a CUDA graph. The stream stays usable on
    /// failure.
    pub fn capture_graph(&mut self, stream: &Stream) -> Result<Graph> {
        // SAFETY: live context and stream.
        let raw =
            NonNull::new(unsafe { ffi::trt_graph_capture(self.raw.as_ptr(), stream.raw.as_ptr()) })
                .ok_or_else(last_error)?;
        Ok(Graph { raw })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: `raw` is live; the engine is released after, by field drop.
        unsafe { ffi::trt_context_destroy(self.raw.as_ptr()) }
    }
}

/// An instantiated CUDA graph of one enqueue. Drop it before the context it
/// was captured from.
pub struct Graph {
    raw: NonNull<ffi::TrtGraph>,
}

// SAFETY: a graph exec is launched from one thread at a time (`&self` is
// only used by the owner that holds it with the context).
unsafe impl Send for Graph {}

impl Graph {
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        // SAFETY: live graph and stream.
        status(unsafe { ffi::trt_graph_launch(self.raw.as_ptr(), stream.raw.as_ptr()) })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: `raw` is live.
        unsafe { ffi::trt_graph_destroy(self.raw.as_ptr()) }
    }
}

/// A non-blocking CUDA stream.
pub struct Stream {
    raw: NonNull<c_void>,
}

// SAFETY: a cudaStream_t may be used from any thread.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        set_device_for_thread()?;
        // SAFETY: no arguments; null is handled.
        let raw = NonNull::new(unsafe { ffi::trt_stream_create() }).ok_or_else(last_error)?;
        Ok(Self { raw })
    }

    /// For the other modules of this crate that queue work on the stream.
    #[cfg(feature = "cuda-preprocess")]
    pub(crate) fn as_raw(&self) -> *mut c_void {
        self.raw.as_ptr()
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: live stream.
        status(unsafe { ffi::trt_stream_synchronize(self.raw.as_ptr()) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: `raw` is live.
        unsafe { ffi::trt_stream_destroy(self.raw.as_ptr()) }
    }
}

/// A CUDA event with timing. Created once and recorded again and again, so
/// GPU timing allocates nothing per frame.
pub struct Event {
    raw: NonNull<c_void>,
}

// SAFETY: a cudaEvent_t may be used from any thread.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        set_device_for_thread()?;
        // SAFETY: no arguments; null is handled.
        let raw = NonNull::new(unsafe { ffi::trt_event_create() }).ok_or_else(last_error)?;
        Ok(Self { raw })
    }

    /// Queues the event on the stream: it completes after the work queued
    /// before it.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        // SAFETY: live event and stream.
        status(unsafe { ffi::trt_event_record(self.raw.as_ptr(), stream.raw.as_ptr()) })
    }

    /// Milliseconds from `start` to this event. Both must have been
    /// recorded, and the stream synchronized past this one.
    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32> {
        let mut ms = 0f32;
        // SAFETY: live events; `ms` is a live local.
        status(unsafe {
            ffi::trt_event_elapsed_ms(start.raw.as_ptr(), self.raw.as_ptr(), &mut ms)
        })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: `raw` is live.
        unsafe { ffi::trt_event_destroy(self.raw.as_ptr()) }
    }
}

/// Zeroed `f32` memory the CPU and the GPU both see (integrated memory:
/// `cudaHostAlloc(mapped)`). The allocation never moves, so binding its
/// device address once stays valid while the buffer lives.
///
/// The host view is only coherent between GPU work: write before the
/// enqueue, read after the stream synchronized.
pub struct MappedBuffer {
    host: NonNull<f32>,
    device: NonNull<c_void>,
    len: usize,
}

// SAFETY: plain memory; the owner synchronizes CPU and GPU access.
unsafe impl Send for MappedBuffer {}
unsafe impl Sync for MappedBuffer {}

impl MappedBuffer {
    pub fn alloc(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|b| *b > 0)
            .ok_or_else(|| Error::new(format!("cannot allocate {len} floats")))?;
        set_device_for_thread()?;
        let (mut host, mut device): (*mut c_void, *mut c_void) =
            (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: both out-pointers are live locals.
        status(unsafe { ffi::trt_host_alloc_mapped(bytes, &mut host, &mut device) })?;
        match (NonNull::new(host.cast::<f32>()), NonNull::new(device)) {
            (Some(host), Some(device)) => Ok(Self { host, device, len }),
            _ => Err(Error::new("cudaHostAlloc returned a null pointer")),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `len` floats were allocated and zeroed at `host`.
        unsafe { std::slice::from_raw_parts(self.host.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: as above, and `&mut self` is exclusive on the CPU side.
        unsafe { std::slice::from_raw_parts_mut(self.host.as_ptr(), self.len) }
    }
}

impl Drop for MappedBuffer {
    fn drop(&mut self) {
        // SAFETY: `host` came from `trt_host_alloc_mapped`.
        unsafe { ffi::trt_host_free(self.host.as_ptr().cast()) }
    }
}

/// A built (serialized) plan.
pub struct Plan {
    raw: NonNull<ffi::TrtPlan>,
}

// SAFETY: immutable bytes.
unsafe impl Send for Plan {}
unsafe impl Sync for Plan {}

impl Plan {
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the plan owns `size` bytes at `data` until it is dropped.
        unsafe {
            let data = ffi::trt_plan_data(self.raw.as_ptr());
            let size = ffi::trt_plan_size(self.raw.as_ptr());
            if data.is_null() || size == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(data.cast(), size)
            }
        }
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: `raw` is live.
        unsafe { ffi::trt_plan_destroy(self.raw.as_ptr()) }
    }
}

type ProgressFn<'a> = dyn FnMut(ProgressEvent<'_>) -> bool + 'a;

struct ProgressSlot<'a> {
    callback: &'a mut ProgressFn<'a>,
}

/// Must not unwind into C++; a panicking callback lets the build go on.
extern "C" fn progress_trampoline(
    user: *mut c_void,
    event: i32,
    phase: *const c_char,
    parent: *const c_char,
    step: i32,
    steps: i32,
) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `user` is the `ProgressSlot` that `build_plan` keeps alive
        // for the whole call, used from this one thread.
        let slot = unsafe { &mut *user.cast::<ProgressSlot<'_>>() };
        let text = |p: *const c_char| -> Option<String> {
            // SAFETY: TensorRT passes NUL-terminated names for this call.
            (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        };
        let phase = text(phase).unwrap_or_default();
        let parent = text(parent);
        let event = match event {
            0 => ProgressEvent::PhaseStart {
                phase: &phase,
                parent: parent.as_deref(),
                steps,
            },
            1 => ProgressEvent::Step {
                phase: &phase,
                step,
            },
            _ => ProgressEvent::PhaseFinish { phase: &phase },
        };
        (slot.callback)(event)
    }));
    i32::from(outcome.unwrap_or(true))
}

/// Builds a plan from an ONNX file (minutes on the first run). `progress`
/// returns `false` to cancel the build at its next step.
pub fn build_plan(
    onnx: &Path,
    options: &BuildOptions,
    progress: &mut ProgressFn<'_>,
) -> Result<Plan> {
    init();
    set_device_for_thread()?;
    let onnx = CString::new(onnx.as_os_str().as_bytes())
        .map_err(|_| Error::new("the model path contains a NUL byte"))?;
    let cache = options
        .timing_cache
        .as_ref()
        .map(|p| CString::new(p.as_os_str().as_bytes()))
        .transpose()
        .map_err(|_| Error::new("the timing cache path contains a NUL byte"))?;
    let mut slot = ProgressSlot { callback: progress };
    // SAFETY: the C strings and `slot` outlive the call; the shim does not
    // keep any of them after it returns.
    let raw = NonNull::new(unsafe {
        ffi::trt_build_plan(
            onnx.as_ptr(),
            i32::from(options.fp16),
            options.workspace_bytes,
            cache.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            Some(progress_trampoline),
            (&mut slot as *mut ProgressSlot<'_>).cast(),
        )
    })
    .ok_or_else(last_error)?;
    Ok(Plan { raw })
}
