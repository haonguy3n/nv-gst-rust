//! Simulator backend: runs the full pipeline on any laptop and in CI
//! with no camera, GPU or model file.
//!
//! It exists from day one as a second backend so the core traits
//! cannot quietly leak Jetson assumptions. Frames are backed by
//! anonymous files in `/dev/shm` (unlinked immediately), which give a
//! real fd to honour the `HwBuffer::dmabuf_fd` contract and a pooled,
//! drop-returns-to-pool lifecycle like the NVMM pool on Jetson.
//! The default engine is deterministic: it derives detections from
//! the input tensor the preprocessor filled, so golden tests are
//! stable. With `--features onnx` and `model.path` set, the same
//! `InferenceEngine` trait is backed by ONNX Runtime instead.

#![forbid(unsafe_code)]

#[cfg(feature = "onnx")]
mod onnx;

use edge_core::{
    DType, Detection, Detections, Encoder, EncoderConfig, EngineState, Error, FrameMemory,
    FrameMeta, FramePoll, FrameSource, GpuStream, HwBuffer, HwFrame, InferenceEngine,
    InputColorspace, Layout, ModelConfig, Overlay, PixelFormat, Platform, PlatformCaps,
    PostProcessor, PreConfig, Precision, Preprocessor, Result, SourceConfig, Tensor, TensorBuffer,
    TensorSpec,
};
use std::any::Any;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------- pool

type FilePool = Arc<Mutex<Vec<File>>>;

static POOL_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn make_pool(pool_size: usize, frame_bytes: usize) -> Result<FilePool> {
    let pool_id = POOL_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut files = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        let path = PathBuf::from(format!(
            "/dev/shm/edge-vision-sim-{}-{pool_id}-{i}",
            std::process::id()
        ));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        // Unlink now: the fd stays valid, nothing is left on disk.
        let _ = std::fs::remove_file(&path);
        file.set_len(frame_bytes as u64)?;
        files.push(file);
    }
    Ok(Arc::new(Mutex::new(files)))
}

/// Simulated hardware buffer. Dropping it returns the file to the pool.
pub struct SimBuffer {
    file: Option<File>,
    pool: FilePool,
}

impl SimBuffer {
    /// Backend-internal read of the pooled bytes. Only sim code
    /// downcasts to this type; the core API still gives the rest of
    /// the pipeline no pixel access.
    fn read_bytes(&self, out: &mut Vec<u8>) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| Error::Platform("sim buffer already returned to pool".into()))?;
        let len = file.metadata()?.len() as usize;
        out.resize(len, 0);
        file.read_exact_at(out, 0)?;
        Ok(())
    }
}

impl HwBuffer for SimBuffer {
    fn memory(&self) -> FrameMemory {
        FrameMemory::DmaBuf
    }
    fn dmabuf_fd(&self) -> RawFd {
        self.file.as_ref().map_or(-1, AsRawFd::as_raw_fd)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for SimBuffer {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            if let Ok(mut pool) = self.pool.lock() {
                pool.push(file);
            }
        }
    }
}

// ---------------------------------------------------------------- source

enum SourceKind {
    Synthetic { frames: Option<u64> },
    File { reader: std::io::BufReader<File> },
}

pub struct SimSource {
    cfg: SourceConfig,
    kind: SourceKind,
    pool: FilePool,
    seq: u64,
    running: bool,
    started: Option<std::time::Instant>,
    scratch: Vec<u8>,
}

impl FrameSource for SimSource {
    fn start(&mut self) -> Result<()> {
        self.running = true;
        self.started = Some(std::time::Instant::now());
        Ok(())
    }

    fn next_frame(&mut self, timeout: Duration) -> Result<FramePoll> {
        if !self.running {
            return Ok(FramePoll::EndOfStream);
        }
        if let SourceKind::Synthetic { frames } = &self.kind {
            if frames.is_some_and(|limit| self.seq >= limit) {
                return Ok(FramePoll::EndOfStream);
            }
        }
        // Pace like a real sensor at the configured fps, so the
        // pipeline's backpressure sees realistic arrival times
        // instead of an instant burst. The wait honours the
        // caller's timeout: a frame due later than that is a
        // Timeout, exactly like a real sensor between frames.
        if let Some(started) = self.started.filter(|_| self.cfg.paced) {
            let due =
                started + Duration::from_secs_f64(self.seq as f64 / f64::from(self.cfg.fps.max(1)));
            let now = std::time::Instant::now();
            if due > now {
                let wait = due - now;
                if wait > timeout {
                    std::thread::sleep(timeout);
                    return Ok(FramePoll::Timeout);
                }
                std::thread::sleep(wait);
            }
        }
        // Pool exhaustion waits up to `timeout` for a buffer to be
        // returned, then reports Timeout instead of deadlocking:
        // the caller sees a timeout and the frame is counted.
        let deadline = std::time::Instant::now() + timeout;
        let mut file = loop {
            if let Some(f) = self.pool.lock().expect("sim pool mutex poisoned").pop() {
                break f;
            }
            if std::time::Instant::now() >= deadline {
                return Ok(FramePoll::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        };

        let frame_bytes = PixelFormat::Nv12.frame_bytes(self.cfg.width, self.cfg.height);
        if matches!(self.kind, SourceKind::Synthetic { .. }) {
            // A moving gradient, so the real preprocessor has
            // non-trivial pixels to work on (and golden tests stay
            // deterministic per sequence number).
            let (w, h) = (self.cfg.width as usize, self.cfg.height as usize);
            self.scratch.resize(frame_bytes, 128);
            for y in 0..h {
                for x in 0..w {
                    self.scratch[y * w + x] = (x + 2 * y + 37 * self.seq as usize) as u8;
                }
            }
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&self.scratch)?;
        }
        if let SourceKind::File { reader } = &mut self.kind {
            // Dev backend only: pixels arrive from a file, so one read
            // into the pooled buffer is unavoidable and intended.
            self.scratch.resize(frame_bytes, 0);
            match reader.read_exact(&mut self.scratch) {
                Ok(()) => {
                    file.seek(SeekFrom::Start(0))?;
                    file.write_all(&self.scratch)?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    if let Ok(mut pool) = self.pool.lock() {
                        pool.push(file);
                    }
                    return Ok(FramePoll::EndOfStream);
                }
                Err(e) => return Err(e.into()),
            }
        }

        let fd = file.as_raw_fd();
        let meta = FrameMeta {
            seq: self.seq,
            pts: Duration::from_secs_f64(self.seq as f64 / f64::from(self.cfg.fps.max(1))),
            width: self.cfg.width,
            height: self.cfg.height,
            format: PixelFormat::Nv12,
            layout: Layout::PitchLinear,
            camera_id: self.cfg.camera_id,
            dmabuf_fd: fd,
        };
        self.seq += 1;
        let buf: Arc<dyn HwBuffer> = Arc::new(SimBuffer {
            file: Some(file),
            pool: Arc::clone(&self.pool),
        });
        Ok(FramePoll::Frame(HwFrame::new(meta, buf)))
    }

    fn stop(&mut self) -> Result<()> {
        self.running = false;
        Ok(())
    }

    fn memory(&self) -> FrameMemory {
        FrameMemory::DmaBuf
    }

    fn layout(&self) -> Layout {
        Layout::PitchLinear
    }

    fn pool_available(&self) -> Option<usize> {
        self.pool.lock().ok().map(|pool| pool.len())
    }
}

// ---------------------------------------------------------------- tensors

/// Host `f32` tensor backing for the simulator.
pub struct HostTensor {
    data: Vec<f32>,
}

impl HostTensor {
    fn zeros(elements: usize) -> Box<dyn TensorBuffer> {
        Box::new(Self {
            data: vec![0.0; elements],
        })
    }

    #[cfg(feature = "onnx")]
    pub(crate) fn from_vec(data: Vec<f32>) -> Box<dyn TensorBuffer> {
        Box::new(Self { data })
    }

    /// Overwrite contents in place, reusing the allocation.
    #[cfg(feature = "onnx")]
    pub(crate) fn replace(&mut self, src: &[f32]) {
        self.data.clear();
        self.data.extend_from_slice(src);
    }
}

impl TensorBuffer for HostTensor {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_f32_slice(&self) -> Option<&[f32]> {
        Some(&self.data)
    }
    fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        Some(&mut self.data)
    }
}

// ---------------------------------------------------------------- pre / engine

/// Real pre-processing on the CPU: NV12 -> letterbox resize ->
/// channel order -> normalise, written straight into the engine's
/// input binding. This is what lets the sim run real models on
/// recorded frames; hardware backends do the same math in one
/// fused GPU/VIC pass without touching the pixels on the CPU.
pub struct SimPreprocessor {
    cfg: PreConfig,
    scratch: Vec<u8>,
}

/// NV12 sample to RGB bytes under the configured matrix/range.
/// Coefficients are the standard BT.601/BT.709 limited- and
/// full-range (JFIF-style) conversions.
fn nv12_rgb(cs: InputColorspace, y: f32, u: f32, v: f32) -> (f32, f32, f32) {
    let (u, v) = (u - 128.0, v - 128.0);
    let (r, g, b) = match cs {
        InputColorspace::Bt601Limited => {
            let y = 1.164 * (y - 16.0);
            (y + 1.596 * v, y - 0.813 * v - 0.391 * u, y + 2.018 * u)
        }
        InputColorspace::Bt709Limited => {
            let y = 1.164 * (y - 16.0);
            (y + 1.793 * v, y - 0.533 * v - 0.213 * u, y + 2.115 * u)
        }
        InputColorspace::Bt601Full => (y + 1.402 * v, y - 0.714 * v - 0.344 * u, y + 1.772 * u),
        InputColorspace::Bt709Full => (y + 1.5748 * v, y - 0.4681 * v - 0.1873 * u, y + 1.8556 * u),
    };
    (
        r.clamp(0.0, 255.0),
        g.clamp(0.0, 255.0),
        b.clamp(0.0, 255.0),
    )
}

impl Preprocessor for SimPreprocessor {
    fn run(&mut self, frame: &HwFrame, dst: &mut Tensor, _stream: &GpuStream) -> Result<()> {
        if self.cfg.layout != Layout::PitchLinear {
            return Err(Error::Unsupported(format!(
                "sim preprocessor reads pitch-linear NV12 only, got {}",
                self.cfg.layout
            )));
        }
        let sim = frame
            .buf
            .as_any()
            .downcast_ref::<SimBuffer>()
            .ok_or_else(|| Error::Platform("sim preprocessor got a foreign HwBuffer".into()))?;
        sim.read_bytes(&mut self.scratch)?;

        let (sw, sh) = (frame.meta().width as usize, frame.meta().height as usize);
        let (dw, dh) = (self.cfg.dst_width as usize, self.cfg.dst_height as usize);
        if self.scratch.len()
            < PixelFormat::Nv12.frame_bytes(frame.meta().width, frame.meta().height)
        {
            return Err(Error::Platform(
                "sim preprocessor: frame buffer smaller than NV12 frame".into(),
            ));
        }
        let out = dst.as_f32_slice_mut().ok_or_else(|| {
            Error::Platform("sim preprocessor needs a host-visible F32 input tensor".into())
        })?;
        if out.len() < 3 * dw * dh {
            return Err(Error::Platform(format!(
                "input tensor too small: {} < {}",
                out.len(),
                3 * dw * dh
            )));
        }

        let norm = edge_core::normalize_factor(self.cfg.normalize);
        out[..3 * dw * dh].fill(self.cfg.pad_value as f32 * norm);

        let lb = edge_core::Letterbox::new(sw as u32, sh as u32, dw as u32, dh as u32);
        let scale = lb.scale;
        let (rw, rh) = (lb.resized_width as usize, lb.resized_height as usize);
        let (ox, oy) = (lb.pad_x as usize, lb.pad_y as usize);
        let (y_plane, uv_plane) = self.scratch.split_at(sw * sh);

        let sample_y = |x: usize, y: usize| y_plane[(y.min(sh - 1)) * sw + x.min(sw - 1)] as f32;
        let sample_uv = |x: usize, y: usize| {
            let cx = (x / 2).min(sw / 2 - 1);
            let cy = (y / 2).min(sh / 2 - 1);
            let i = cy * sw + cx * 2;
            (uv_plane[i] as f32, uv_plane[i + 1] as f32)
        };

        for ry in 0..rh {
            // Bilinear source position for this resized row.
            let sy = (ry as f32 + 0.5) / scale - 0.5;
            let y0 = sy.floor().max(0.0) as usize;
            let fy = (sy - y0 as f32).clamp(0.0, 1.0);
            for rx in 0..rw {
                let sx = (rx as f32 + 0.5) / scale - 0.5;
                let x0 = sx.floor().max(0.0) as usize;
                let fx = (sx - x0 as f32).clamp(0.0, 1.0);
                let yv = sample_y(x0, y0) * (1.0 - fx) * (1.0 - fy)
                    + sample_y(x0 + 1, y0) * fx * (1.0 - fy)
                    + sample_y(x0, y0 + 1) * (1.0 - fx) * fy
                    + sample_y(x0 + 1, y0 + 1) * fx * fy;
                let (u, v) = sample_uv(x0, y0);
                let (r, g, b) = nv12_rgb(self.cfg.colorspace, yv, u, v);
                let (c0, c1, c2) = edge_core::order_channels(self.cfg.color, r, g, b);
                let (dx, dy) = (ox + rx, oy + ry);
                out[dy * dw + dx] = c0 * norm;
                out[dw * dh + dy * dw + dx] = c1 * norm;
                out[2 * dw * dh + dy * dw + dx] = c2 * norm;
            }
        }
        Ok(())
    }
}

/// Number of candidate rows the sim engine emits; each row is
/// `[x, y, w, h, score, class]` in normalised coordinates.
pub const SIM_CANDIDATES: usize = 4;
pub const SIM_ROW: usize = 6;

pub struct SimEngine {
    inputs: Vec<Tensor>,
    outputs: Vec<Tensor>,
    calls: u64,
}

impl SimEngine {
    fn new(cfg: &ModelConfig) -> Self {
        let input_elems = 3 * (cfg.input_width as usize) * (cfg.input_height as usize);
        let inputs = vec![Tensor::new(
            TensorSpec::new(
                "images",
                vec![1, 3, cfg.input_height as usize, cfg.input_width as usize],
                DType::F32,
            ),
            HostTensor::zeros(input_elems),
        )];
        let outputs = vec![Tensor::new(
            TensorSpec::new("detections", vec![SIM_CANDIDATES, SIM_ROW], DType::F32),
            HostTensor::zeros(SIM_CANDIDATES * SIM_ROW),
        )];
        Self {
            inputs,
            outputs,
            calls: 0,
        }
    }
}

impl InferenceEngine for SimEngine {
    fn input_bindings(&mut self) -> &mut [Tensor] {
        &mut self.inputs
    }
    fn output_bindings(&self) -> &[Tensor] {
        &self.outputs
    }
    fn infer(&mut self, _stream: &GpuStream) -> Result<()> {
        let score = self.inputs[0]
            .as_f32_slice()
            .and_then(|s| s.first().copied())
            .unwrap_or(0.0);
        let out = &mut self.outputs[0];
        let data = out
            .as_f32_slice_mut()
            .ok_or_else(|| Error::Platform("sim engine output tensor is not writable".into()))?;
        let drift = (self.calls % 20) as f32 * 0.01;
        // Candidate 0: real detection. Candidate 1: below any sane
        // threshold, so post-processing filtering is exercised.
        data[0..SIM_ROW].copy_from_slice(&[0.10 + drift, 0.20, 0.30, 0.25, score, 0.0]);
        data[SIM_ROW..2 * SIM_ROW].copy_from_slice(&[0.55, 0.50, 0.20, 0.20, 0.05, 1.0]);
        for row in data[2 * SIM_ROW..].chunks_mut(SIM_ROW) {
            row.fill(0.0);
        }
        self.calls += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------- post / overlay

/// Platform-neutral decoder for the sim engine's row format.
/// A YOLO decoder + NMS in a shared `postproc` crate replaces this
/// once a real model lands.
pub struct SimPostProcessor {
    score_threshold: f32,
}

impl SimPostProcessor {
    pub fn new(score_threshold: f32) -> Self {
        Self { score_threshold }
    }
}

impl PostProcessor for SimPostProcessor {
    fn decode(&mut self, outputs: &[Tensor], meta: &FrameMeta, out: &mut Detections) -> Result<()> {
        out.clear(meta);
        let Some(tensor) = outputs.first() else {
            return Ok(());
        };
        let Some(data) = tensor.as_f32_slice() else {
            return Err(Error::Platform(
                "sim postprocessor needs a host-visible F32 output tensor".into(),
            ));
        };
        for row in data.as_chunks::<SIM_ROW>().0 {
            let score = row[4];
            if score >= self.score_threshold {
                out.push(Detection {
                    x: row[0],
                    y: row[1],
                    width: row[2],
                    height: row[3],
                    score,
                    class_id: row[5] as u32,
                });
            }
        }
        Ok(())
    }
}

pub struct SimOverlay;

impl Overlay for SimOverlay {
    fn draw(&mut self, _frame: &HwFrame, det: &Detections, _stream: &GpuStream) -> Result<()> {
        // The core API gives no CPU pixel access, and the sim has no
        // GPU, so drawing is a deliberate no-op here. Metadata still
        // reaches the sink, which is what sim tests assert on.
        let _ = det.len();
        Ok(())
    }
}

// ---------------------------------------------------------------- platform

#[derive(Debug, Default)]
pub struct SimPlatform;

impl Platform for SimPlatform {
    fn name(&self) -> &'static str {
        "sim"
    }

    fn caps(&self) -> PlatformCaps {
        PlatformCaps {
            frame_memory: FrameMemory::DmaBuf,
            formats: vec![PixelFormat::Nv12],
            max_width: 3840,
            max_height: 2160,
            codecs: Vec::new(),
            precisions: vec![Precision::Fp32],
            devices: vec!["cpu".into()],
            has_hw_encoder: false,
            cpu_encode: false,
        }
    }

    fn source(&self, cfg: &SourceConfig) -> Result<Box<dyn FrameSource>> {
        let kind = match cfg.kind.as_str() {
            "synthetic" => SourceKind::Synthetic { frames: cfg.frames },
            "file" => {
                let path = cfg
                    .path
                    .clone()
                    .ok_or_else(|| Error::invalid("source.path", "required for file source"))?;
                SourceKind::File {
                    reader: std::io::BufReader::new(File::open(path)?),
                }
            }
            other => {
                return Err(Error::Unsupported(format!(
                    "sim source kind `{other}` (use \"synthetic\" or \"file\")"
                )))
            }
        };
        let pool = make_pool(
            cfg.pool_size,
            PixelFormat::Nv12.frame_bytes(cfg.width, cfg.height),
        )?;
        Ok(Box::new(SimSource {
            cfg: cfg.clone(),
            kind,
            pool,
            seq: 0,
            running: false,
            started: None,
            scratch: Vec::new(),
        }))
    }

    fn preprocessor(&self, cfg: &PreConfig) -> Result<Box<dyn Preprocessor>> {
        Ok(Box::new(SimPreprocessor {
            cfg: cfg.clone(),
            scratch: Vec::new(),
        }))
    }

    fn engine(&self, cfg: &ModelConfig) -> Result<EngineState> {
        match &cfg.path {
            None => Ok(EngineState::Ready(Box::new(SimEngine::new(cfg)))),
            Some(path) => {
                #[cfg(feature = "onnx")]
                {
                    Ok(EngineState::Ready(Box::new(onnx::OnnxEngine::load(
                        path, cfg,
                    )?)))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(Error::Unsupported(format!(
                        "model.path = {} requires building platform-sim \
                         with --features onnx (or clear model.path to use \
                         the deterministic sim engine)",
                        path.display()
                    )))
                }
            }
        }
    }

    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        Ok(Box::new(SimOverlay))
    }

    fn encoder(&self, _cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>> {
        // Like Orin Nano: no hardware encoder, metadata only.
        Ok(None)
    }
}

/// Sim platform whose engine takes `ready_after` to become
/// available, modelling a TensorRT plan build on first boot.
/// Everything except `engine()` delegates to [`SimPlatform`]; it
/// exists so the runner's warming path (video flows, detections
/// empty, then inference joins) is testable without hardware.
pub struct DelayedSimPlatform {
    pub ready_after: Duration,
}

struct SimWarmup {
    ready_at: std::time::Instant,
    cfg: ModelConfig,
}

impl edge_core::EngineWarmup for SimWarmup {
    fn poll_ready(&mut self) -> Result<Option<Box<dyn InferenceEngine>>> {
        if std::time::Instant::now() < self.ready_at {
            return Ok(None);
        }
        match SimPlatform.engine(&self.cfg)? {
            EngineState::Ready(engine) => Ok(Some(engine)),
            EngineState::Warming(_) => Ok(None),
        }
    }
}

impl Platform for DelayedSimPlatform {
    fn name(&self) -> &'static str {
        "sim-delayed"
    }
    fn caps(&self) -> PlatformCaps {
        SimPlatform.caps()
    }
    fn source(&self, cfg: &SourceConfig) -> Result<Box<dyn FrameSource>> {
        SimPlatform.source(cfg)
    }
    fn preprocessor(&self, cfg: &PreConfig) -> Result<Box<dyn Preprocessor>> {
        SimPlatform.preprocessor(cfg)
    }
    fn engine(&self, cfg: &ModelConfig) -> Result<EngineState> {
        Ok(EngineState::Warming(Box::new(SimWarmup {
            ready_at: std::time::Instant::now() + self.ready_after,
            cfg: cfg.clone(),
        })))
    }
    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        SimPlatform.overlay()
    }
    fn encoder(&self, cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>> {
        SimPlatform.encoder(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_cfg(pool_size: usize, frames: u64) -> SourceConfig {
        SourceConfig {
            kind: "synthetic".into(),
            path: None,
            camera_id: 0,
            width: 64,
            height: 48,
            fps: 30,
            exposure_us: None,
            gain_min: None,
            gain_max: None,
            paced: true,
            pool_size,
            frames: Some(frames),
        }
    }

    #[test]
    fn sequence_and_pts_are_monotonic() {
        let platform = SimPlatform;
        let mut src = platform.source(&synthetic_cfg(6, 5)).unwrap();
        src.start().unwrap();
        let mut last_pts = Duration::ZERO;
        for expected_seq in 0..5 {
            let FramePoll::Frame(frame) = src.next_frame(Duration::from_secs(2)).unwrap() else {
                panic!("expected a frame");
            };
            assert_eq!(frame.meta().seq, expected_seq);
            assert!(frame.meta().pts >= last_pts);
            assert_eq!(frame.meta().dmabuf_fd, frame.buf.dmabuf_fd());
            last_pts = frame.meta().pts;
        }
        assert!(matches!(
            src.next_frame(Duration::from_secs(2)).unwrap(),
            FramePoll::EndOfStream
        ));
        src.stop().unwrap();
    }

    #[test]
    fn pool_exhaustion_times_out_instead_of_deadlocking() {
        let platform = SimPlatform;
        let mut src = platform.source(&synthetic_cfg(3, 100)).unwrap();
        src.start().unwrap();
        // Hold every pooled buffer: the next call must not block.
        let held: Vec<_> = (0..3)
            .map(|_| match src.next_frame(Duration::from_secs(2)).unwrap() {
                FramePoll::Frame(f) => f,
                _ => panic!("expected a frame"),
            })
            .collect();
        assert!(matches!(
            src.next_frame(Duration::from_millis(1)).unwrap(),
            FramePoll::Timeout
        ));
        drop(held);
        // Buffers returned on drop, so capture resumes.
        assert!(matches!(
            src.next_frame(Duration::from_secs(2)).unwrap(),
            FramePoll::Frame(_)
        ));
        src.stop().unwrap();
    }
}
