//! Platform-neutral sinks.
//!
//! RTSP/WebRTC serving (GStreamer) belongs here too, but needs a
//! hardware encoder feeding it, so it lands with the Jetson backend.
//! What every profile can use today: JSONL metadata (the Orin Nano
//! "detections only" output, and golden-test material), raw file
//! video, fan-out, and a counting sink for tests and benches.

#![forbid(unsafe_code)]

use edge_core::{Detections, EncodedPacket, Error, Result, Sink};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

// ------------------------------------------------------------------ JSONL

/// Detections as JSON lines: `{"frame_seq":N,"items":[...]}` per line.
pub struct JsonlMetaSink<W: Write + Send> {
    writer: W,
}

impl JsonlMetaSink<std::io::BufWriter<std::fs::File>> {
    pub fn create(path: &Path) -> Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self::new(std::io::BufWriter::new(file)))
    }
}

impl<W: Write + Send> JsonlMetaSink<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: Write + Send> Sink for JsonlMetaSink<W> {
    fn push_video(&mut self, _pkt: EncodedPacket) -> Result<()> {
        Err(Error::Unsupported(
            "JsonlMetaSink is metadata-only; pair an encoder with a video sink".into(),
        ))
    }

    fn push_meta(&mut self, det: &Detections) -> Result<()> {
        serde_json::to_writer(&mut self.writer, det)
            .map_err(|e| Error::Platform(format!("serialising detections: {e}")))?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }
}

// ------------------------------------------------------------------ file

/// Raw encoded bitstream to a file (elementary stream; no muxing).
pub struct FileVideoSink {
    writer: std::io::BufWriter<std::fs::File>,
    pub packets: u64,
    pub bytes: u64,
}

impl FileVideoSink {
    pub fn create(path: &Path) -> Result<Self> {
        Ok(Self {
            writer: std::io::BufWriter::new(std::fs::File::create(path)?),
            packets: 0,
            bytes: 0,
        })
    }
}

impl Sink for FileVideoSink {
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()> {
        self.writer.write_all(&pkt.data)?;
        self.packets += 1;
        self.bytes += pkt.data.len() as u64;
        Ok(())
    }
}

// ------------------------------------------------------------------ fan-out

/// Forwards every packet and metadata frame to all child sinks.
/// Fail-fast: the first child error aborts the push.
pub struct MultiSink {
    sinks: Vec<Box<dyn Sink>>,
}

impl MultiSink {
    pub fn new(sinks: Vec<Box<dyn Sink>>) -> Self {
        Self { sinks }
    }
}

impl Sink for MultiSink {
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()> {
        let mut iter = self.sinks.iter_mut().peekable();
        while let Some(sink) = iter.next() {
            if iter.peek().is_some() {
                sink.push_video(pkt.clone())?;
            } else {
                sink.push_video(pkt)?;
                break;
            }
        }
        Ok(())
    }

    fn push_meta(&mut self, det: &Detections) -> Result<()> {
        for sink in &mut self.sinks {
            sink.push_meta(det)?;
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ counting

#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    pub meta_frames: u64,
    pub detections: u64,
    pub video_packets: u64,
    pub video_bytes: u64,
}

/// Counts everything and keeps nothing. The shared `Counts` handle
/// stays readable after the sink is boxed into a pipeline.
pub struct CountingSink {
    counts: Arc<Mutex<Counts>>,
}

impl CountingSink {
    pub fn new(counts: Arc<Mutex<Counts>>) -> Self {
        Self { counts }
    }
}

impl Sink for CountingSink {
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()> {
        let mut c = self.counts.lock().expect("counts mutex poisoned");
        c.video_packets += 1;
        c.video_bytes += pkt.data.len() as u64;
        Ok(())
    }

    fn push_meta(&mut self, det: &Detections) -> Result<()> {
        let mut c = self.counts.lock().expect("counts mutex poisoned");
        c.meta_frames += 1;
        c.detections += det.len() as u64;
        Ok(())
    }
}

/// Discards everything. For pure throughput benches.
pub struct NullSink;

impl Sink for NullSink {
    fn push_video(&mut self, _pkt: EncodedPacket) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::Detection;
    use std::time::Duration;

    fn detections(seq: u64) -> Detections {
        let mut d = Detections::with_capacity(2);
        d.clear(seq);
        d.push(Detection {
            x: 0.375,
            y: 0.375,
            width: 0.25,
            height: 0.25,
            score: 0.9,
            class_id: 1,
        });
        d
    }

    #[test]
    fn jsonl_sink_writes_one_valid_json_object_per_line() {
        let mut sink = JsonlMetaSink::new(Vec::new());
        sink.push_meta(&detections(3)).unwrap();
        sink.push_meta(&detections(4)).unwrap();
        let text = String::from_utf8(sink.into_inner()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v["frame_seq"], 3);
        assert_eq!(v["items"][0]["class_id"], 1);
        assert!((v["items"][0]["score"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn jsonl_sink_rejects_video() {
        let mut sink = JsonlMetaSink::new(Vec::new());
        let pkt = EncodedPacket {
            pts: Duration::ZERO,
            keyframe: true,
            data: vec![0u8; 4],
        };
        assert!(sink.push_video(pkt).is_err());
    }

    #[test]
    fn multi_sink_fans_out_to_every_child() {
        let counts_a = Arc::new(Mutex::new(Counts::default()));
        let counts_b = Arc::new(Mutex::new(Counts::default()));
        let mut multi = MultiSink::new(vec![
            Box::new(CountingSink::new(Arc::clone(&counts_a))),
            Box::new(CountingSink::new(Arc::clone(&counts_b))),
        ]);
        multi.push_meta(&detections(0)).unwrap();
        multi
            .push_video(EncodedPacket {
                pts: Duration::ZERO,
                keyframe: false,
                data: vec![1u8; 10],
            })
            .unwrap();
        for counts in [&counts_a, &counts_b] {
            let c = *counts.lock().unwrap();
            assert_eq!(c.meta_frames, 1);
            assert_eq!(c.detections, 1);
            assert_eq!(c.video_packets, 1);
            assert_eq!(c.video_bytes, 10);
        }
    }

    #[test]
    fn file_sink_writes_packet_bytes_verbatim() {
        let dir = std::env::temp_dir().join(format!("edge-sinks-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.h264");
        let mut sink = FileVideoSink::create(&path).unwrap();
        for payload in [&[0x00, 0x00, 0x01, 0x67][..], &[0x00, 0x00, 0x01, 0x65][..]] {
            sink.push_video(EncodedPacket {
                pts: Duration::ZERO,
                keyframe: true,
                data: payload.to_vec(),
            })
            .unwrap();
        }
        assert_eq!(sink.packets, 2);
        assert_eq!(sink.bytes, 8);
        drop(sink);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes, vec![0, 0, 1, 0x67, 0, 0, 1, 0x65]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
