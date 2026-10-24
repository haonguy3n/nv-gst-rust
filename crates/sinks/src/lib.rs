//! Platform-neutral sinks.
//!
//! RTSP/WebRTC serving (GStreamer) belongs here too, but needs a
//! hardware encoder feeding it, so it lands with the Jetson backend.
//! What every profile can use today: JSONL metadata (the Orin Nano
//! "detections only" output, and golden-test material), raw file
//! video, fan-out, and a counting sink for tests and benches.

#![forbid(unsafe_code)]

#[cfg(feature = "rtsp")]
pub mod rtsp;
pub mod web_ui;
pub use web_ui::{WebUi, WebUiSink};

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
        // A metadata sink in a fan-out next to a video sink is a
        // legitimate topology; ignoring video here (documented)
        // beats killing an otherwise healthy run on packet one.
        Ok(())
    }

    fn push_meta(&mut self, det: &Detections) -> Result<()> {
        serde_json::to_writer(&mut self.writer, det)
            .map_err(|e| Error::Platform(format!("serialising detections: {e}")))?;
        self.writer.write_all(b"\n")?;
        // Lines are small and consumers tail this: flush per frame.
        self.writer.flush()?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.flush()?;
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
        if pkt.keyframe {
            self.writer.flush()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

// ------------------------------------------------------------------ tcp

/// Raw Annex-B H.264 over TCP, one viewer at a time
/// (`ffplay -f h264 tcp://<cam>:8555`). A client problem is never a
/// run error: the client is dropped with one log line. A new client
/// starts at the next keyframe (x264 repeats SPS/PPS in every IDR).
pub struct TcpH264Sink {
    listener: std::net::TcpListener,
    client: Option<std::net::TcpStream>,
    synced: bool,
}

impl TcpH264Sink {
    /// Writes to a stalled viewer give up after this long.
    const WRITE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);
    /// Total time one packet may take; the timeout above is per syscall,
    /// so a viewer trickling bytes would otherwise hold the writer for
    /// minutes.
    const PACKET_DEADLINE: std::time::Duration = std::time::Duration::from_secs(1);

    pub fn bind(addr: std::net::SocketAddr) -> Result<Self> {
        let listener = std::net::TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        Ok(Self {
            listener,
            client: None,
            synced: false,
        })
    }

    pub fn local_addr(&self) -> Result<std::net::SocketAddr> {
        Ok(self.listener.local_addr()?)
    }

    /// Takes a pending connection; a newer viewer replaces the old one.
    fn accept_pending(&mut self) {
        loop {
            match self.listener.accept() {
                Ok((stream, peer)) => {
                    let ready = stream
                        .set_nonblocking(false)
                        .and_then(|()| stream.set_write_timeout(Some(Self::WRITE_TIMEOUT)))
                        .and_then(|()| stream.set_nodelay(true));
                    if let Err(e) = ready {
                        log::warn!("video_tcp: dropping {peer}: socket setup: {e}");
                        continue;
                    }
                    log::info!("video_tcp: viewer {peer} connected; waiting for a keyframe");
                    self.client = Some(stream);
                    self.synced = false;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return,
                Err(e) => {
                    log::error!("video_tcp: accept failed: {e}");
                    return;
                }
            }
        }
    }
}

impl Sink for TcpH264Sink {
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()> {
        self.accept_pending();
        if self.client.is_none() || (!self.synced && !pkt.keyframe) {
            return Ok(());
        }
        self.synced = true;
        if let Some(client) = &mut self.client {
            if let Err(e) = write_within(client, &pkt.data, Self::PACKET_DEADLINE) {
                log::warn!("video_tcp: viewer dropped: {e}");
                self.client = None;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if let Some(client) = self.client.take() {
            // Unblocks a viewer-side stall; the peer sees a clean close.
            let _ = client.shutdown(std::net::Shutdown::Both);
        }
        Ok(())
    }
}

/// `write_all` with one total deadline for the whole buffer.
pub(crate) fn write_within(
    out: &mut impl Write,
    mut data: &[u8],
    deadline: std::time::Duration,
) -> std::io::Result<()> {
    use std::io::ErrorKind;
    let start = std::time::Instant::now();
    while !data.is_empty() {
        match out.write(data) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => data = &data[n..],
            Err(e) if matches!(e.kind(), ErrorKind::Interrupted) => {}
            Err(e) if !matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return Err(e);
            }
            Err(_) => {}
        }
        if !data.is_empty() && start.elapsed() >= deadline {
            return Err(ErrorKind::TimedOut.into());
        }
    }
    Ok(())
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

    fn finish(&mut self) -> Result<()> {
        // Finish every child; the first error wins.
        let mut first: Option<Error> = None;
        for sink in &mut self.sinks {
            if let Err(e) = sink.finish() {
                if first.is_none() {
                    first = Some(e);
                }
            }
        }
        first.map_or(Ok(()), Err)
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
        d.clear(&edge_core::FrameMeta {
            seq,
            pts: Duration::from_millis(33 * seq),
            width: 64,
            height: 64,
            format: edge_core::PixelFormat::Nv12,
            layout: edge_core::Layout::PitchLinear,
            planes: edge_core::PixelFormat::Nv12.plane_layout(64, 64),
            camera_id: 2,
            dmabuf_fd: -1,
        });
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

    fn packet(keyframe: bool, tag: u8) -> EncodedPacket {
        EncodedPacket::new(Duration::ZERO, keyframe, vec![tag; 4])
    }

    fn read_until_closed(mut client: std::net::TcpStream) -> Vec<u8> {
        use std::io::Read;
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut got = Vec::new();
        let _ = client.read_to_end(&mut got);
        got
    }

    /// Pushes until the listener has accepted the client (the connect
    /// completes asynchronously); returns once a keyframe was written.
    fn push_until_served(sink: &mut TcpH264Sink) {
        for _ in 0..200 {
            sink.push_video(packet(false, 1)).unwrap();
            if sink.client.is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("client was never accepted");
    }

    #[test]
    fn tcp_sink_waits_for_a_keyframe_per_client() {
        let mut sink = TcpH264Sink::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let client = std::net::TcpStream::connect(sink.local_addr().unwrap()).unwrap();
        push_until_served(&mut sink);
        sink.push_video(packet(false, 2)).unwrap();
        sink.push_video(packet(true, 3)).unwrap();
        sink.push_video(packet(false, 4)).unwrap();
        sink.finish().unwrap();
        drop(sink);
        assert_eq!(read_until_closed(client), [3, 3, 3, 3, 4, 4, 4, 4]);
    }

    #[test]
    fn tcp_sink_survives_a_dead_client_and_serves_the_next() {
        let mut sink = TcpH264Sink::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let addr = sink.local_addr().unwrap();
        let first = std::net::TcpStream::connect(addr).unwrap();
        push_until_served(&mut sink);
        drop(first);
        // Writes to a closed peer fail after the RST arrives; never an Err.
        for _ in 0..50 {
            sink.push_video(packet(true, 5)).unwrap();
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(sink.client.is_none(), "dead client must be dropped");
        let second = std::net::TcpStream::connect(addr).unwrap();
        push_until_served(&mut sink);
        sink.push_video(packet(true, 6)).unwrap();
        drop(sink);
        assert_eq!(read_until_closed(second), [6, 6, 6, 6]);
    }

    #[test]
    fn tcp_sink_drops_a_viewer_that_stalls_inside_one_packet() {
        let mut sink = TcpH264Sink::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        // Connected but never reading: a packet larger than the socket
        // buffers cannot be written.
        let _stalled = std::net::TcpStream::connect(sink.local_addr().unwrap()).unwrap();
        push_until_served(&mut sink);
        let start = std::time::Instant::now();
        sink.push_video(EncodedPacket::new(Duration::ZERO, true, vec![7; 64 << 20]))
            .unwrap();
        assert!(start.elapsed() < Duration::from_millis(1900), "bounded");
        assert!(sink.client.is_none(), "stalled viewer must be dropped");
        sink.finish().unwrap();
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
    fn jsonl_lines_are_flushed_and_attributed_before_drop() {
        let dir = std::env::temp_dir().join(format!("edge-sinks-flush-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta.jsonl");
        let mut sink = JsonlMetaSink::create(&path).unwrap();
        sink.push_meta(&detections(7)).unwrap();
        // No finish(), no drop: the line must already be readable.
        let text = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(v["frame_seq"], 7);
        assert_eq!(v["camera_id"], 2);
        // pts is one nanosecond count for joining, not an object.
        assert_eq!(v["pts_ns"], 33 * 7 * 1_000_000);
        assert!(v.get("pts").is_none(), "no nested pts object: {text}");
        sink.finish().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn jsonl_sink_ignores_video_so_fanout_topologies_work() {
        let mut sink = JsonlMetaSink::new(Vec::new());
        let pkt = EncodedPacket {
            pts: Duration::ZERO,
            keyframe: true,
            data: vec![0u8; 4].into(),
        };
        sink.push_video(pkt)
            .expect("video is ignored, not an error");
        assert!(sink.into_inner().is_empty(), "nothing was written");
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
                data: vec![1u8; 10].into(),
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
                data: payload.to_vec().into(),
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

// ---------------------------------------------------------------- async

/// Moves a sink's blocking writes onto a bounded writer thread.
///
/// The processing thread only queues a packet/metadata frame; file
/// flushes and slow disks happen on `sink-writer`. The processing
/// thread holds capture buffers, so `push_*` never blocks: when the
/// queue is full the packet/frame is dropped and counted in
/// [`AsyncDrops`]. After a dropped video packet, video is held back
/// until the next keyframe so a viewer never decodes a torn GOP.
/// Only `finish` waits (for room to queue the stop request).
pub struct AsyncSink {
    tx: Option<std::sync::mpsc::SyncSender<Command>>,
    handle: Option<std::thread::JoinHandle<Result<()>>>,
    drops: Arc<AsyncDrops>,
    failure: Arc<Failure>,
    await_keyframe: bool,
}

/// The inner sink's first error, visible to the pushing side.
#[derive(Default)]
struct Failure {
    failed: std::sync::atomic::AtomicBool,
    message: Mutex<String>,
}

impl Failure {
    fn record(&self, e: &Error) {
        use std::sync::atomic::Ordering::{Acquire, Release};
        if !self.failed.load(Acquire) {
            *self.message.lock().expect("failure mutex poisoned") = e.to_string();
            self.failed.store(true, Release);
        }
    }

    fn check(&self) -> Result<()> {
        if self.failed.load(std::sync::atomic::Ordering::Acquire) {
            let msg = self.message.lock().expect("failure mutex poisoned");
            return Err(Error::Platform(format!("sink-writer failed: {msg}")));
        }
        Ok(())
    }
}

/// Packets dropped by an [`AsyncSink`] because its queue was full.
#[derive(Debug, Default)]
pub struct AsyncDrops {
    pub video: std::sync::atomic::AtomicU64,
    pub meta: std::sync::atomic::AtomicU64,
}

enum Command {
    Video(EncodedPacket),
    Meta(edge_core::Detections),
    Finish,
}

impl AsyncSink {
    pub fn new(mut inner: Box<dyn Sink>, capacity: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Command>(capacity.max(1));
        let failure = Arc::new(Failure::default());
        let writer_failure = Arc::clone(&failure);
        let handle = std::thread::Builder::new()
            .name("sink-writer".into())
            .spawn(move || -> Result<()> {
                let mut first: Option<Error> = None;
                let mut finished = false;
                while let Ok(cmd) = rx.recv() {
                    let is_finish = matches!(cmd, Command::Finish);
                    let result = match cmd {
                        Command::Video(pkt) => inner.push_video(pkt),
                        Command::Meta(det) => inner.push_meta(&det),
                        Command::Finish => inner.finish(),
                    };
                    if let Err(e) = result {
                        writer_failure.record(&e);
                        if first.is_none() {
                            first = Some(e);
                        }
                    }
                    if is_finish {
                        finished = true;
                        break;
                    }
                }
                // Channel closed without an explicit Finish (Drop path).
                if !finished {
                    if let Err(e) = inner.finish() {
                        if first.is_none() {
                            first = Some(e);
                        }
                    }
                }
                first.map_or(Ok(()), Err)
            })
            .expect("spawning sink-writer thread");
        Self {
            tx: Some(tx),
            handle: Some(handle),
            drops: Arc::new(AsyncDrops::default()),
            failure,
            await_keyframe: false,
        }
    }

    /// Shared drop counters; stay readable after the sink is boxed.
    pub fn drops(&self) -> Arc<AsyncDrops> {
        Arc::clone(&self.drops)
    }

    /// Queues without blocking. `Ok(false)` means the queue was full.
    fn try_queue(&self, cmd: Command) -> Result<bool> {
        use std::sync::mpsc::TrySendError;
        self.failure.check()?;
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| Error::Platform("async sink is already finished".into()))?;
        match tx.try_send(cmd) {
            Ok(()) => Ok(true),
            Err(TrySendError::Full(_)) => Ok(false),
            Err(TrySendError::Disconnected(_)) => {
                Err(Error::Platform("sink-writer thread is gone".into()))
            }
        }
    }
}

impl Sink for AsyncSink {
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        if self.await_keyframe && !pkt.keyframe {
            self.drops.video.fetch_add(1, Relaxed);
            return Ok(());
        }
        let keyframe = pkt.keyframe;
        if self.try_queue(Command::Video(pkt))? {
            if keyframe {
                self.await_keyframe = false;
            }
        } else {
            self.await_keyframe = true;
            self.drops.video.fetch_add(1, Relaxed);
        }
        Ok(())
    }

    fn push_meta(&mut self, det: &edge_core::Detections) -> Result<()> {
        if !self.try_queue(Command::Meta(det.clone()))? {
            self.drops
                .meta
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.tx.is_some() {
            // Blocking is fine here: shutdown, and the join waits anyway.
            if let Some(tx) = self.tx.take() {
                let _ = tx.send(Command::Finish);
            }
        }
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        match handle.join() {
            Ok(result) => result,
            Err(_) => Err(Error::Platform("sink-writer thread panicked".into())),
        }
    }
}

impl Drop for AsyncSink {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[cfg(test)]
mod async_tests {
    use super::*;
    use edge_core::{Detection, Detections, FrameMeta, Layout, PixelFormat};
    use std::time::Duration;

    fn detections(seq: u64) -> Detections {
        let mut d = Detections::with_capacity(1);
        d.clear(&FrameMeta {
            seq,
            pts: Duration::from_millis(seq),
            width: 64,
            height: 64,
            format: PixelFormat::Nv12,
            layout: Layout::PitchLinear,
            planes: PixelFormat::Nv12.plane_layout(64, 64),
            camera_id: 3,
            dmabuf_fd: -1,
        });
        d.push(Detection {
            x: 0.5,
            y: 0.5,
            width: 0.1,
            height: 0.1,
            score: 0.9,
            class_id: 2,
        });
        d
    }

    #[test]
    fn async_jsonl_sink_writes_on_the_writer_thread_and_flushes() {
        let dir = std::env::temp_dir().join(format!("edge-sinks-async-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta.jsonl");
        let inner: Box<dyn Sink> = Box::new(JsonlMetaSink::create(&path).unwrap());
        let mut sink = AsyncSink::new(inner, 8);
        for seq in 0..5 {
            sink.push_meta(&detections(seq)).unwrap();
        }
        sink.finish().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn async_sink_reports_a_closed_writer() {
        let counts = Arc::new(Mutex::new(Counts::default()));
        let inner: Box<dyn Sink> = Box::new(CountingSink::new(Arc::clone(&counts)));
        let mut sink = AsyncSink::new(inner, 1);
        sink.finish().unwrap();
        assert!(sink.push_meta(&detections(1)).is_err());
    }

    struct Gate {
        release: std::sync::mpsc::Receiver<()>,
        seen: Arc<Mutex<Vec<bool>>>,
    }

    impl Sink for Gate {
        fn push_video(&mut self, pkt: EncodedPacket) -> Result<()> {
            let _ = self.release.recv();
            self.seen.lock().unwrap().push(pkt.keyframe);
            Ok(())
        }
    }

    #[test]
    fn async_sink_never_blocks_and_counts_drops_until_a_keyframe() {
        use std::sync::atomic::Ordering::Relaxed;
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let inner: Box<dyn Sink> = Box::new(Gate {
            release: release_rx,
            seen: Arc::clone(&seen),
        });
        let mut sink = AsyncSink::new(inner, 2);
        let drops = sink.drops();
        let pkt = |key| EncodedPacket::new(Duration::ZERO, key, vec![0]);
        // The writer is stuck on the first packet; two more fill the queue.
        for _ in 0..20 {
            sink.push_video(pkt(true)).unwrap();
        }
        assert!(drops.video.load(Relaxed) >= 17, "full queue must drop");
        // Non-keyframes are held back after a drop, even with room.
        let before = drops.video.load(Relaxed);
        sink.push_video(pkt(false)).unwrap();
        assert_eq!(drops.video.load(Relaxed), before + 1);
        sink.push_meta(&detections(1)).unwrap();
        sink.push_meta(&detections(2)).unwrap();
        sink.push_meta(&detections(3)).unwrap();
        // The queue is still full (the writer is stuck): all three drop.
        assert_eq!(drops.meta.load(Relaxed), 3);
        for _ in 0..30 {
            let _ = release_tx.send(());
        }
        // Room again: a keyframe re-opens the gate, then deltas flow.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            sink.push_video(pkt(true)).unwrap();
            if !sink.await_keyframe {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "gate never reopened");
            std::thread::sleep(Duration::from_millis(5));
        }
        let before = drops.video.load(Relaxed);
        for _ in 0..3 {
            sink.push_video(pkt(false)).unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(drops.video.load(Relaxed), before, "video flows again");
        sink.finish().unwrap();
        let seen = seen.lock().unwrap();
        assert!(seen.iter().filter(|&&k| !k).count() >= 1, "deltas arrived");
    }

    struct Failing {
        finishes: Arc<std::sync::atomic::AtomicU32>,
    }

    impl Sink for Failing {
        fn push_video(&mut self, _pkt: EncodedPacket) -> Result<()> {
            Err(Error::Platform("disk full".into()))
        }

        fn finish(&mut self) -> Result<()> {
            self.finishes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn async_sink_surfaces_an_inner_error_on_a_later_push() {
        let finishes = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let inner: Box<dyn Sink> = Box::new(Failing {
            finishes: Arc::clone(&finishes),
        });
        let mut sink = AsyncSink::new(inner, 4);
        let pkt = || EncodedPacket::new(Duration::ZERO, true, vec![0]);
        sink.push_video(pkt()).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let err = loop {
            match sink.push_video(pkt()) {
                Err(e) => break e,
                Ok(()) => {
                    assert!(std::time::Instant::now() < deadline, "error never surfaced");
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        };
        assert!(err.to_string().contains("disk full"), "{err}");
        assert!(sink.push_meta(&detections(1)).is_err());
        // finish keeps first-error semantics.
        let fin = sink.finish().unwrap_err();
        assert!(fin.to_string().contains("disk full"), "{fin}");
    }

    #[test]
    fn async_sink_finishes_the_inner_sink_exactly_once() {
        let finishes = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let inner: Box<dyn Sink> = Box::new(Failing {
            finishes: Arc::clone(&finishes),
        });
        let mut sink = AsyncSink::new(inner, 4);
        sink.finish().unwrap();
        drop(sink);
        assert_eq!(finishes.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
