//! Simulator backend: runs the full pipeline on any laptop and in CI
//! with no camera, GPU or model file.
//!
//! It exists from day one as a second backend so the core traits
//! cannot quietly leak Jetson assumptions. Frames are backed by
//! anonymous files in `/dev/shm` (unlinked immediately), which give a
//! real fd to honour the `HwBuffer::dmabuf_fd` contract and a pooled,
//! drop-returns-to-pool lifecycle like the NVMM pool on Jetson.
//! The engine is deterministic: it derives detections from the input
//! tensor the preprocessor filled, so golden tests are stable.
//!
//! Not yet done (needs a model + clip in the repo, tracked in
//! README): a real ONNX Runtime engine behind the same
//! `InferenceEngine` trait.

#![forbid(unsafe_code)]

use edge_core::{
    DType, Detection, Detections, Encoder, EncoderConfig, Error, FrameMemory, FrameMeta,
    FrameSource, GpuStream, HwBuffer, HwFrame, InferenceEngine, Layout, ModelConfig, Overlay,
    PixelFormat, Platform, PlatformCaps, PostProcessor, PreConfig, Preprocessor, Result,
    SourceConfig, Tensor, TensorBuffer, TensorSpec,
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

fn make_pool(pool_size: usize, frame_bytes: usize) -> Result<FilePool> {
    let mut files = Vec::with_capacity(pool_size);
    for i in 0..pool_size {
        let path = PathBuf::from(format!(
            "/dev/shm/edge-vision-sim-{}-{i}",
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
}

impl FrameSource for SimSource {
    fn start(&mut self) -> Result<()> {
        self.running = true;
        self.started = Some(std::time::Instant::now());
        Ok(())
    }

    fn next_frame(&mut self, _timeout: Duration) -> Result<Option<HwFrame>> {
        if !self.running {
            return Ok(None);
        }
        if let SourceKind::Synthetic { frames } = &self.kind {
            if frames.is_some_and(|limit| self.seq >= limit) {
                return Ok(None);
            }
        }
        // Pace like a real sensor at the configured fps, so the
        // pipeline's backpressure sees realistic arrival times
        // instead of an instant burst.
        if let Some(started) = self.started {
            let due =
                started + Duration::from_secs_f64(self.seq as f64 / f64::from(self.cfg.fps.max(1)));
            let now = std::time::Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
        }
        // Pool exhaustion drops instead of deadlocking: the caller
        // sees a timeout and the frame is counted by the pipeline.
        let mut file = {
            let mut pool = self.pool.lock().expect("sim pool mutex poisoned");
            match pool.pop() {
                Some(f) => f,
                None => return Ok(None),
            }
        };

        let frame_bytes = PixelFormat::Nv12.frame_bytes(self.cfg.width, self.cfg.height);
        if let SourceKind::File { reader } = &mut self.kind {
            // Dev backend only: pixels arrive from a file, so one read
            // into the pooled buffer is unavoidable and intended.
            let mut chunk = vec![0u8; frame_bytes];
            match reader.read_exact(&mut chunk) {
                Ok(()) => {
                    file.seek(SeekFrom::Start(0))?;
                    file.write_all(&chunk)?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    if let Ok(mut pool) = self.pool.lock() {
                        pool.push(file);
                    }
                    return Ok(None);
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
        Ok(Some(HwFrame::new(meta, buf)))
    }

    fn stop(&mut self) -> Result<()> {
        self.running = false;
        Ok(())
    }

    fn memory(&self) -> FrameMemory {
        FrameMemory::DmaBuf
    }
}

// ---------------------------------------------------------------- tensors

/// Host `f32` tensor backing for the simulator.
pub struct HostTensor {
    data: Vec<f32>,
}

impl HostTensor {
    fn zeros(elements: usize) -> Arc<dyn TensorBuffer> {
        Arc::new(Self {
            data: vec![0.0; elements],
        })
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

pub struct SimPreprocessor {
    dst: (u32, u32),
}

impl Preprocessor for SimPreprocessor {
    fn run(&mut self, frame: &HwFrame, dst: &mut Tensor, _stream: &GpuStream) -> Result<()> {
        // No pixel access through the core API (by design), so the sim
        // fills the input binding with a deterministic pattern keyed
        // by the frame sequence. A hardware backend replaces this
        // with a fused GPU kernel reading the surface in place.
        let expected = (self.dst.0 as usize) * (self.dst.1 as usize) * 3;
        let slice = dst.as_f32_slice_mut().ok_or_else(|| {
            Error::Platform("sim preprocessor needs a host-visible F32 input tensor".into())
        })?;
        if slice.len() < expected {
            return Err(Error::Platform(format!(
                "input tensor too small: {} < {expected}",
                slice.len()
            )));
        }
        let v = 0.5 + (frame.meta.seq % 10) as f32 / 20.0; // 0.50..=0.95
        slice[..expected].fill(v);
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
    fn decode(&self, outputs: &[Tensor], meta: &FrameMeta, out: &mut Detections) -> Result<()> {
        out.clear(meta.seq);
        let Some(tensor) = outputs.first() else {
            return Ok(());
        };
        let Some(data) = tensor.as_f32_slice() else {
            return Err(Error::Platform(
                "sim postprocessor needs a host-visible F32 output tensor".into(),
            ));
        };
        for row in data.chunks_exact(SIM_ROW) {
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
            has_hw_encoder: false,
            accelerators: vec!["cpu-sim".into()],
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
        }))
    }

    fn preprocessor(&self, cfg: &PreConfig) -> Result<Box<dyn Preprocessor>> {
        Ok(Box::new(SimPreprocessor {
            dst: (cfg.dst_width, cfg.dst_height),
        }))
    }

    fn engine(&self, cfg: &ModelConfig) -> Result<Box<dyn InferenceEngine>> {
        Ok(Box::new(SimEngine::new(cfg)))
    }

    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        Ok(Box::new(SimOverlay))
    }

    fn encoder(&self, _cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>> {
        // Like Orin Nano: no hardware encoder, metadata only.
        Ok(None)
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
            let frame = src
                .next_frame(Duration::from_millis(10))
                .unwrap()
                .expect("frame");
            assert_eq!(frame.meta.seq, expected_seq);
            assert!(frame.meta.pts >= last_pts);
            assert_eq!(frame.meta.dmabuf_fd, frame.buf.dmabuf_fd());
            last_pts = frame.meta.pts;
        }
        assert!(src.next_frame(Duration::from_millis(10)).unwrap().is_none());
        src.stop().unwrap();
    }

    #[test]
    fn pool_exhaustion_returns_none_instead_of_deadlocking() {
        let platform = SimPlatform;
        let mut src = platform.source(&synthetic_cfg(3, 100)).unwrap();
        src.start().unwrap();
        // Hold every pooled buffer: the next call must not block.
        let held: Vec<_> = (0..3)
            .map(|_| src.next_frame(Duration::from_millis(1)).unwrap().unwrap())
            .collect();
        assert!(src.next_frame(Duration::from_millis(1)).unwrap().is_none());
        drop(held);
        // Buffers returned on drop, so capture resumes.
        assert!(src.next_frame(Duration::from_millis(1)).unwrap().is_some());
        src.stop().unwrap();
    }
}
