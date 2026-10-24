//! GStreamer capture: a launch string ending in
//! `appsink name=frames` becomes a `FrameSource`.
//!
//! On the Jetson the string is `launch::capture_launch`
//! (nvarguscamerasrc into NVMM). Pulled buffers whose memory is a
//! dmabuf are wrapped **by reference** — the same fd the rest of
//! the pipeline asserts on. NVMM buffers are not upstream
//! GstDmaBufMemory, so an NVMM-declared source **fails loudly**
//! until the NvBufSurface branch (gst-nvmm `nvmmalloc`) is wired
//! in M1 — it never copies. Only a non-NVMM source (a dev PC's
//! `videotestsrc`, where there is no production path) takes the
//! counted shm-copy fallback, and only for buffers that are
//! exactly one tight NV12 frame.

use edge_core::{
    Error, FrameMemory, FrameMeta, FramePoll, FrameSource, HwBuffer, HwFrame, Layout, PixelFormat,
    PlaneDesc, PlaneLayout, Result, SourceConfig, TimeoutReason,
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

static POOL_SEQ: AtomicU64 = AtomicU64::new(0);

fn make_pool(pool_size: usize, frame_bytes: usize) -> Result<FilePool> {
    let pool_id = POOL_SEQ.fetch_add(1, Ordering::Relaxed);
    let mut files = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        let path = PathBuf::from(format!(
            "/dev/shm/edge-vision-gst-{}-{pool_id}-{i}",
            std::process::id()
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
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
    /// NVMM surface: the fd/layout/planes come from `NvBufSurface`,
    /// never from a guessed tight stride.
    Nvmm {
        buffer: gst::Buffer,
        fd: RawFd,
        layout: Layout,
        planes: PlaneLayout,
        width: u32,
        height: u32,
    },
    /// PC-dev fallback: one counted copy into a pooled shm file.
    Copied { file: Option<File>, pool: FilePool },
}

impl HwBuffer for GstFrameBuffer {
    fn memory(&self) -> FrameMemory {
        match self {
            Self::DmaBuf { memory, .. } => *memory,
            Self::Nvmm { .. } => FrameMemory::Nvmm,
            // Honest tag: this is a CPU copy into shm, not a dmabuf
            // and not zero-copy. It exists only for dev-PC pipelines.
            Self::Copied { .. } => FrameMemory::Other("shm-copy"),
        }
    }

    fn dmabuf_fd(&self) -> RawFd {
        match self {
            Self::DmaBuf { fd, .. } | Self::Nvmm { fd, .. } => *fd,
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
            Self::DmaBuf { buffer, .. } | Self::Nvmm { buffer, .. } => Some(buffer),
            Self::Copied { .. } => None,
        }
    }

    /// Layout/plane geometry read from the NvBufSurface, when this is
    /// an NVMM frame. `None` for the host fallback.
    pub fn nvmm_info(&self) -> Option<(Layout, PlaneLayout, u32, u32)> {
        match self {
            Self::Nvmm {
                layout,
                planes,
                width,
                height,
                ..
            } => Some((*layout, *planes, *width, *height)),
            _ => None,
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
    layout: Layout,
    seq: u64,
    /// PC-only fallback pool. `None` for an NVMM source: on the
    /// device the copy fallback must never run, so creating ~19 MB of
    /// `/dev/shm` files at startup would only waste RAM and can fail
    /// under the sandbox.
    pool: Option<FilePool>,
    copied_frames: Arc<AtomicU64>,
    /// An Argus camera's EOS is ambiguous: the sensor may have
    /// failed, or `nvargus-daemon` may have lost the session. Treat it
    /// as a fault so systemd restarts the process, never as a clean
    /// end of stream (which exits 0 and recovers nothing).
    argus: bool,
    /// Set when Argus EOS may mean a lost daemon. `set_state(Null)`
    /// can segfault inside the Argus client after a daemon restart, so
    /// teardown is skipped and the process exits instead.
    skip_teardown: bool,
}

impl GstLaunchSource {
    pub fn new(
        launch: &str,
        cfg: SourceConfig,
        memory: FrameMemory,
        layout: Layout,
    ) -> Result<Self> {
        gst::init().map_err(|e| Error::Platform(format!("gstreamer init: {e}")))?;
        let pipeline = gst::parse::launch(launch)
            .map_err(|e| Error::Platform(format!("parsing launch string: {e}")))?
            .downcast::<gst::Pipeline>()
            .map_err(|_| Error::Platform("launch string did not build a pipeline".into()))?;
        let appsink = pipeline
            .by_name("frames")
            .and_then(|el| el.downcast::<gst_app::AppSink>().ok())
            .ok_or_else(|| Error::Platform("launch string has no appsink named `frames`".into()))?;
        if cfg.kind == "file" {
            let path = cfg
                .path
                .as_ref()
                .ok_or_else(|| Error::invalid("source.path", "required for file source"))?;
            let filesrc = pipeline.by_name("source").ok_or_else(|| {
                Error::Platform("file launch string has no filesrc named `source`".into())
            })?;
            filesrc.set_property("location", path.to_string_lossy().as_ref());
        }
        let pool = if memory == FrameMemory::Nvmm {
            None
        } else {
            Some(make_pool(
                cfg.pool_size,
                PixelFormat::Nv12.frame_bytes(cfg.width, cfg.height),
            )?)
        };
        let argus = cfg.kind == "argus";
        Ok(Self {
            pipeline,
            appsink,
            cfg,
            memory,
            layout,
            seq: 0,
            pool,
            copied_frames: Arc::new(AtomicU64::new(0)),
            argus,
            skip_teardown: false,
        })
    }

    /// Frames that took the PC-dev copy fallback. Must stay 0 on
    /// the Jetson; anything else is a zero-copy regression.
    pub fn copied_frames(&self) -> u64 {
        self.copied_frames.load(Ordering::Relaxed)
    }

    /// Wrap a pulled buffer. `Ok(None)` means the fallback pool
    /// is exhausted and the frame is dropped (caller reports
    /// `Timeout`); it is never an error and never blocks.
    fn wrap_buffer(&mut self, buffer: gst::Buffer) -> Result<Option<Arc<dyn HwBuffer>>> {
        if buffer.n_memory() > 0 {
            let mem = buffer.peek_memory(0);
            if let Some(dmabuf) = mem.downcast_memory_ref::<gstreamer_allocators::DmaBufMemory>() {
                return Ok(Some(Arc::new(GstFrameBuffer::DmaBuf {
                    fd: dmabuf.fd(),
                    buffer,
                    memory: self.memory,
                })));
            }
        }
        // nvarguscamerasrc's NVMM memory is not an upstream
        // GstDmaBufMemory. `gst_buffer_map(READ)` exposes an
        // `NvBufSurface*` in the mapped bytes; read its fd, layout
        // and plane params (T2), never the block-linear pixels.
        if self.memory == FrameMemory::Nvmm {
            let map = buffer.map_readable().map_err(|e| {
                Error::Platform(format!(
                    "mapping NVMM buffer for NvBufSurface: {e}; refusing to copy"
                ))
            })?;
            let info = nvbuf_sys::surface_from_bytes(map.as_slice()).map_err(|e| {
                Error::Platform(format!(
                    "NVMM buffer is not a readable NvBufSurface ({e}); refusing to copy"
                ))
            })?;
            drop(map);
            let layout = match info.layout {
                nvbuf_sys::SurfaceLayout::Pitch => Layout::PitchLinear,
                nvbuf_sys::SurfaceLayout::BlockLinear => Layout::BlockLinear,
            };
            let planes: Vec<PlaneDesc> = info
                .planes()
                .map(|p| PlaneDesc {
                    offset: p.offset,
                    pitch: p.pitch,
                    row_bytes: p.row_bytes,
                    rows: p.rows,
                })
                .collect();
            return Ok(Some(Arc::new(GstFrameBuffer::Nvmm {
                buffer,
                fd: info.fd,
                layout,
                planes: PlaneLayout::from_planes(&planes),
                width: info.width,
                height: info.height,
            })));
        }
        // Fallback: copy into the shm pool (dev PCs only). The map
        // must be exactly one tight NV12 frame; anything else
        // (padding, block-linear) is not a frame we may claim.
        let expected = PixelFormat::Nv12.frame_bytes(self.cfg.width, self.cfg.height);
        let Some(pool) = self.pool.as_ref() else {
            return Err(Error::Platform(
                "no copy-fallback pool for an NVMM source; refusing to copy".into(),
            ));
        };
        let Some(mut file) = pool.lock().expect("gst pool mutex poisoned").pop() else {
            return Ok(None);
        };
        // A failed copy must return the pooled file: leaking one per
        // bad buffer would starve the fallback pool after `pool_size`
        // format errors and turn a clear error into a stop.
        let copied = (|| -> Result<()> {
            let map = buffer
                .map_readable()
                .map_err(|e| Error::Platform(format!("mapping gst buffer: {e}")))?;
            if map.size() != expected {
                return Err(Error::Platform(format!(
                    "gst buffer is {} bytes, expected {expected} (tight NV12 {}x{}); \
                     padded or block-linear buffers need plane metadata, not a raw copy",
                    map.size(),
                    self.cfg.width,
                    self.cfg.height
                )));
            }
            use std::io::Seek;
            file.seek(std::io::SeekFrom::Start(0))?;
            file.write_all(map.as_slice())?;
            Ok(())
        })();
        if let Err(e) = copied {
            pool.lock().expect("gst pool mutex poisoned").push(file);
            return Err(e);
        }
        self.copied_frames.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Arc::new(GstFrameBuffer::Copied {
            file: Some(file),
            pool: Arc::clone(pool),
        })))
    }

    /// Drain pending bus Error/Eos messages. A runtime pipeline
    /// error (camera loss, decoder failure) must surface as `Err`,
    /// not loop forever as `Timeout`.
    fn poll_bus(&mut self) -> Result<Option<FramePoll>> {
        let Some(bus) = self.pipeline.bus() else {
            return Ok(None);
        };
        // `pop_filtered`, not `timed_pop_filtered(ZERO, ..)`: gstreamer-rs
        // 0.25 computes `total - elapsed` first, and a zero budget is
        // already spent by then, so it returned None without ever
        // looking at the bus.
        while let Some(msg) = bus.pop_filtered(&[gst::MessageType::Error, gst::MessageType::Eos]) {
            match msg.view() {
                gst::MessageView::Error(e) => {
                    return Err(Error::Platform(format!(
                        "gstreamer pipeline error from {:?}: {} ({:?})",
                        msg.src().map(|s| s.path_string()),
                        e.error(),
                        e.debug()
                    )));
                }
                gst::MessageView::Eos(_) => {
                    return if self.argus {
                        self.skip_teardown = true;
                        Err(Error::CaptureServiceLost(
                            "Argus ended with a bare EOS; sensor or nvargus-daemon session lost"
                                .into(),
                        ))
                    } else {
                        Ok(Some(FramePoll::EndOfStream))
                    };
                }
                _ => {}
            }
        }
        Ok(None)
    }
}

impl GstLaunchSource {
    /// Apply exposure/gain to `nvarguscamerasrc` after the pipeline is
    /// PLAYING. The launch string never carries these ranges (Nano
    /// black-frame regression); a missing property is a configuration
    /// error, not something to ignore.
    fn apply_sensor_ranges(&self) -> Result<()> {
        if self.cfg.kind != "argus" {
            return Ok(());
        }
        let Some(camsrc) = self.pipeline.by_name("camsrc") else {
            return Ok(());
        };
        let set = |name: &str, value: String| -> Result<()> {
            if camsrc.find_property(name).is_none() {
                return Err(Error::Platform(format!(
                    "capture element does not expose `{name}`"
                )));
            }
            camsrc.set_property_from_str(name, &value);
            Ok(())
        };
        if let Some(us) = self.cfg.exposure_us {
            let ns = u64::from(us) * 1000;
            set("exposuretimerange", format!("{ns} {ns}"))?;
        }
        if self.cfg.gain_min.is_some() || self.cfg.gain_max.is_some() {
            set(
                "gainrange",
                format!(
                    "{:.1} {:.1}",
                    self.cfg.gain_min.unwrap_or(1.0),
                    self.cfg.gain_max.unwrap_or(16.0)
                ),
            )?;
        }
        Ok(())
    }
}

impl FrameSource for GstLaunchSource {
    fn start(&mut self) -> Result<()> {
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| Error::Platform(format!("starting capture pipeline: {e}")))?;
        // Wait for PLAYING before touching the sensor properties: on
        // the Nano, ranges set at parse time produced black frames.
        self.pipeline
            .state(gst::ClockTime::from_seconds(5))
            .0
            .map_err(|e| Error::Platform(format!("waiting for capture PLAYING: {e}")))?;
        self.apply_sensor_ranges()
    }

    fn next_frame(&mut self, timeout: Duration) -> Result<FramePoll> {
        if let Some(poll) = self.poll_bus()? {
            return Ok(poll);
        }
        let sample = match self
            .appsink
            .try_pull_sample(gst::ClockTime::from_nseconds(timeout.as_nanos() as u64))
        {
            Some(sample) => sample,
            None => {
                if let Some(poll) = self.poll_bus()? {
                    return Ok(poll);
                }
                if self.appsink.is_eos() {
                    return if self.argus {
                        self.skip_teardown = true;
                        Err(Error::CaptureServiceLost(
                            "Argus ended with a bare EOS; sensor or nvargus-daemon session lost"
                                .into(),
                        ))
                    } else {
                        Ok(FramePoll::EndOfStream)
                    };
                }
                return Ok(FramePoll::Timeout(TimeoutReason::NoFrame));
            }
        };
        let buffer = sample
            .buffer_owned()
            .ok_or_else(|| Error::Platform("appsink sample without buffer".into()))?;
        let pts = buffer.pts().map_or_else(
            || Duration::from_secs_f64(self.seq as f64 / f64::from(self.cfg.fps.max(1))),
            |t| Duration::from_nanos(t.nseconds()),
        );
        let Some(buf) = self.wrap_buffer(buffer)? else {
            // Fallback pool exhausted: this frame is dropped.
            return Ok(FramePoll::Timeout(TimeoutReason::PoolExhausted));
        };
        let (layout, planes, width, height) = buf
            .as_any()
            .downcast_ref::<GstFrameBuffer>()
            .and_then(|b| b.nvmm_info())
            .unwrap_or_else(|| {
                // Host fallback: only the pitch-linear copy has a
                // real tight layout; never claim one for block-linear.
                let planes = if self.layout == Layout::BlockLinear {
                    PlaneLayout::default()
                } else {
                    PixelFormat::Nv12.plane_layout(self.cfg.width, self.cfg.height)
                };
                (self.layout, planes, self.cfg.width, self.cfg.height)
            });
        let meta = FrameMeta {
            seq: self.seq,
            pts,
            width,
            height,
            format: PixelFormat::Nv12,
            layout,
            planes,
            camera_id: self.cfg.camera_id,
            dmabuf_fd: buf.dmabuf_fd(),
        };
        self.seq += 1;
        Ok(FramePoll::Frame(HwFrame::new(meta, buf)))
    }

    fn stop(&mut self) -> Result<()> {
        if self.skip_teardown {
            // Leak the pipeline until exit: tearing it down after a
            // lost Argus daemon can segfault in C++.
            return Ok(());
        }
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| Error::Platform(format!("stopping capture pipeline: {e}")))?;
        Ok(())
    }

    fn memory(&self) -> FrameMemory {
        self.memory
    }

    fn layout(&self) -> Layout {
        self.layout
    }

    fn pool_available(&self) -> Option<usize> {
        self.pool
            .as_ref()
            .and_then(|pool| pool.lock().ok().map(|pool| pool.len()))
    }
}

impl Drop for GstLaunchSource {
    fn drop(&mut self) {
        if !self.skip_teardown {
            let _ = self.pipeline.set_state(gst::State::Null);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file_cfg(path: PathBuf) -> SourceConfig {
        SourceConfig {
            kind: "file".into(),
            path: Some(path),
            camera_id: 0,
            width: 64,
            height: 64,
            fps: 30,
            exposure_us: None,
            gain_min: None,
            gain_max: None,
            paced: true,
            pool_size: 2,
            argus_pool: None,
            frames: None,
        }
    }

    /// R7: a path with spaces or `!` must be a property value, never
    /// syntax in the launch string.
    #[test]
    fn file_source_sets_location_as_a_property() {
        gst::init().expect("gst init");
        if gst::ElementFactory::find("filesrc").is_none()
            || gst::ElementFactory::find("decodebin").is_none()
        {
            eprintln!("skipping: filesrc/decodebin not available");
            return;
        }
        let path = PathBuf::from("/tmp/clip with a space !.mp4");
        let launch = "filesrc name=source ! decodebin ! videoconvert \
            ! video/x-raw,format=NV12,width=64,height=64 \
            ! appsink name=frames max-buffers=2 drop=true";
        let source = GstLaunchSource::new(
            launch,
            file_cfg(path.clone()),
            FrameMemory::DmaBuf,
            Layout::PitchLinear,
        )
        .expect("parse launch");
        let filesrc = source.pipeline.by_name("source").expect("named filesrc");
        let location: String = filesrc.property("location");
        assert_eq!(location, path.to_string_lossy());
    }
}
