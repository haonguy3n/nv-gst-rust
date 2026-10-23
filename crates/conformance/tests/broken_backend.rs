//! The suite is only worth keeping if it fails on a broken backend.
//! These stubs violate the contract on purpose.

use edge_core::{
    Codec, Encoder, EncoderConfig, Error, FrameMemory, FrameMeta, FramePoll, FrameSource, HwBuffer,
    HwFrame, InferenceEngine, Layout, ModelConfig, Overlay, PixelFormat, Platform, PlatformCaps,
    PreConfig, Preprocessor, Result, SourceConfig,
};
use std::any::Any;
use std::os::fd::RawFd;
use std::sync::Arc;
use std::time::Duration;

struct StubBuffer;
impl HwBuffer for StubBuffer {
    fn memory(&self) -> FrameMemory {
        FrameMemory::DmaBuf
    }
    fn dmabuf_fd(&self) -> RawFd {
        0
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Hands out frames whose sequence never increases.
struct StubSource;
impl FrameSource for StubSource {
    fn start(&mut self) -> Result<()> {
        Ok(())
    }
    fn next_frame(&mut self, _timeout: Duration) -> Result<FramePoll> {
        Ok(FramePoll::Frame(HwFrame::new(
            FrameMeta {
                seq: 0, // the violation under test
                pts: Duration::ZERO,
                width: 64,
                height: 64,
                format: PixelFormat::Nv12,
                layout: Layout::PitchLinear,
                camera_id: 0,
                dmabuf_fd: 0,
            },
            Arc::new(StubBuffer),
        )))
    }
    fn stop(&mut self) -> Result<()> {
        Ok(())
    }
    fn memory(&self) -> FrameMemory {
        FrameMemory::DmaBuf
    }
}

/// Advertises a hardware encoder it does not have.
struct BrokenPlatform;
impl Platform for BrokenPlatform {
    fn name(&self) -> &'static str {
        "broken"
    }
    fn caps(&self) -> PlatformCaps {
        PlatformCaps {
            frame_memory: FrameMemory::DmaBuf,
            formats: vec![PixelFormat::Nv12],
            max_width: 1920,
            max_height: 1080,
            codecs: vec![Codec::H264],
            has_hw_encoder: true, // the lie under test
            cpu_encode: false,
            accelerators: Vec::new(),
        }
    }
    fn source(&self, _cfg: &SourceConfig) -> Result<Box<dyn FrameSource>> {
        Ok(Box::new(StubSource))
    }
    fn preprocessor(&self, _cfg: &PreConfig) -> Result<Box<dyn Preprocessor>> {
        Err(Error::Unsupported("stub".into()))
    }
    fn engine(&self, _cfg: &ModelConfig) -> Result<Box<dyn InferenceEngine>> {
        Err(Error::Unsupported("stub".into()))
    }
    fn overlay(&self) -> Result<Box<dyn Overlay>> {
        Err(Error::Unsupported("stub".into()))
    }
    fn encoder(&self, _cfg: &EncoderConfig) -> Result<Option<Box<dyn Encoder>>> {
        Ok(None)
    }
}

fn source_cfg() -> SourceConfig {
    SourceConfig {
        kind: "stub".into(),
        path: None,
        camera_id: 0,
        width: 64,
        height: 64,
        fps: 30,
        paced: true,
        pool_size: 4,
        frames: None,
    }
}

#[test]
#[should_panic(expected = "seq must strictly increase")]
fn suite_catches_non_increasing_sequence() {
    edge_conformance::check_frame_order_and_timestamps(&BrokenPlatform, &source_cfg());
}

#[test]
#[should_panic(expected = "encoder() must return Some exactly when")]
fn suite_catches_encoder_caps_lie() {
    edge_conformance::check_encoder_option_matches_caps(&BrokenPlatform);
}
