//! GStreamer capture: a launch string ending in
//! `appsink name=frames` becomes a `FrameSource`.
//!
//! On the Jetson the string is `launch::capture_launch`
//! (nvarguscamerasrc into NVMM). Pulled buffers whose memory is a
//! dmabuf are wrapped **by reference** — the same fd the rest of
//! the pipeline asserts on. Buffers in plain system memory (a dev
//! PC's `videotestsrc`, where there is no NVMM and no production
//! path) are copied once into a pooled shm file so the fd contract
//! still holds; every such copy is counted in `copied_frames`, so
//! on the device a nonzero count is a loud zero-copy regression,
//! never a silent one.

use edge_core::{
    Error, FrameMemory, FrameMeta, FramePoll, FrameSource, HwBuffer, HwFrame, Layout, PixelFormat,
    Result, SourceConfig,
};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::any::Any;
use std::fs::File;
use std::io::Write;
use std::os::fd::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ------------------------------------------------------------ buffers

type FilePool = Arc<Mutex<Vec<File>>>;

fn make_pool(pool_size: usize, frame_bytes: usize) -> Result<FilePool> {
    let mut files = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        let path = PathBuf::from(format!(
            "/dev/shm/edge-vision-gst-{}-{i}",
            std::process::id()
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        let _ = std::fs::remove_file(&path);
        file.set_len(frame_bytes as u64)?;
        files.push(file);
    }
    Ok(Arc::new(Mutex::new(files)))
}

pub enum GstFrameBuffer {
    /// The GStreamer buffer itself; pixels never left the dmabuf.
    DmaBuf {
        buffer: gst::Buffer,
        fd: RawFd,
        memory: FrameMemory,
    },
    /// PC-dev fallback: one counted copy into a pooled shm file.
    Copied { file: Option<File>, pool: FilePool },
}

impl HwBuffer for GstFrameBuffer {
    fn memory(&self) -> FrameMemory {
        match self {
            Self::DmaBuf { memory, .. } => *memory,
            Self::Copied { .. } => FrameMemory::DmaBuf,
        }
    }

    fn dmabuf_fd(&self) -> RawFd {
        match self {
            Self::DmaBuf { fd, .. } => *fd,
            Self::Copied { file, .. } => file.as_ref().map_or(-1, AsRawFd::as_raw_fd),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl GstFrameBuffer {
    /// The wrapped GStreamer buffer, when no copy happened.
    /// Later Jetson stages (CUDA pre-process, NVENC appsrc) work
    /// from this inside the backend.
    pub fn gst_buffer(&self) -> Option<&gst::Buffer> {
        match self {
            Self::DmaBuf { buffer, .. } => Some(buffer),
            Self::Copied { .. } => None,
        }
    }
}

impl Drop for GstFrameBuffer {
    fn drop(&mut self) {
        if let Self::Copied { file, pool } = self {
            if let Some(file) = file.take() {
                if let Ok(mut pool) = pool.lock() {
                    pool.push(file);
                }
            }
        }
    }
}

// ------------------------------------------------------------ source

pub struct GstLaunchSource {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    cfg: SourceConfig,
    memory: FrameMemory,
    seq: u64,
    pool: FilePool,
    copied_frames: Arc<AtomicU64>,
}

impl GstLaunchSource {
    pub fn new(launch: &str, cfg: SourceConfig, memory: FrameMemory) -> Result<Self> {
        gst::init().map_err(|e| Error::Platform(format!("gstreamer init: {e}")))?;
        let pipeline = gst::parse::launch(launch)
            .map_err(|e| Error::Platform(format!("parsing launch string: {e}")))?
            .downcast::<gst::Pipeline>()
            .map_err(|_| Error::Platform("launch string did not build a pipeline".into()))?;
        let appsink = pipeline
            .by_name("frames")
            .and_then(|el| el.downcast::<gst_app::AppSink>().ok())
            .ok_or_else(|| Error::Platform("launch string has no appsink named `frames`".into()))?;
        let pool = make_pool(
            cfg.pool_size,
            PixelFormat::Nv12.frame_bytes(cfg.width, cfg.height),
        )?;
        Ok(Self {
            pipeline,
            appsink,
            cfg,
            memory,
            seq: 0,
            pool,
            copied_frames: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Frames that took the PC-dev copy fallback. Must stay 0 on
    /// the Jetson; anything else is a zero-copy regression.
    pub fn copied_frames(&self) -> u64 {
        self.copied_frames.load(Ordering::Relaxed)
    }

    fn wrap_buffer(&mut self, buffer: gst::Buffer) -> Result<Arc<dyn HwBuffer>> {
        if buffer.n_memory() > 0 {
            let mem = buffer.peek_memory(0);
            if let Some(dmabuf) = mem.downcast_memory_ref::<gstreamer_allocators::DmaBufMemory>() {
                return Ok(Arc::new(GstFrameBuffer::DmaBuf {
                    fd: dmabuf.fd(),
                    buffer,
                    memory: self.memory,
                }));
            }
        }
        // Fallback: copy into the shm pool (dev PCs only).
        let mut file = self
            .pool
            .lock()
            .expect("gst pool mutex poisoned")
            .pop()
            .ok_or_else(|| Error::Platform("gst source fallback pool exhausted".into()))?;
        {
            let map = buffer
                .map_readable()
                .map_err(|e| Error::Platform(format!("mapping gst buffer: {e}")))?;
            use std::io::Seek;
            file.seek(std::io::SeekFrom::Start(0))?;
            file.write_all(map.as_slice())?;
        }
        self.copied_frames.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::new(GstFrameBuffer::Copied {
            file: Some(file),
            pool: Arc::clone(&self.pool),
        }))
    }
}

impl FrameSource for GstLaunchSource {
    fn start(&mut self) -> Result<()> {
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| Error::Platform(format!("starting capture pipeline: {e}")))?;
        Ok(())
    }

    fn next_frame(&mut self, timeout: Duration) -> Result<FramePoll> {
        let sample = match self
            .appsink
            .try_pull_sample(gst::ClockTime::from_nseconds(timeout.as_nanos() as u64))
        {
            Some(sample) => sample,
            None if self.appsink.is_eos() => return Ok(FramePoll::EndOfStream),
            None => return Ok(FramePoll::Timeout),
        };
        let buffer = sample
            .buffer_owned()
            .ok_or_else(|| Error::Platform("appsink sample without buffer".into()))?;
        let pts = buffer.pts().map_or_else(
            || Duration::from_secs_f64(self.seq as f64 / f64::from(self.cfg.fps.max(1))),
            |t| Duration::from_nanos(t.nseconds()),
        );
        let buf = self.wrap_buffer(buffer)?;
        let meta = FrameMeta {
            seq: self.seq,
            pts,
            width: self.cfg.width,
            height: self.cfg.height,
            format: PixelFormat::Nv12,
            layout: Layout::PitchLinear,
            camera_id: self.cfg.camera_id,
            dmabuf_fd: buf.dmabuf_fd(),
        };
        self.seq += 1;
        Ok(FramePoll::Frame(HwFrame::new(meta, buf)))
    }

    fn stop(&mut self) -> Result<()> {
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| Error::Platform(format!("stopping capture pipeline: {e}")))?;
        Ok(())
    }

    fn memory(&self) -> FrameMemory {
        self.memory
    }
}

impl Drop for GstLaunchSource {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}
