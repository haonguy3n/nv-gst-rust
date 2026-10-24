//! Browser UI: live video, detection boxes and counters on one port.
//!
//! `GET /` serves the embedded page, `GET /ws` upgrades to a WebSocket.
//! The camera does not re-encode: every access unit of the existing
//! H.264 stream goes out as one binary message and the page decodes it
//! with WebCodecs (`VideoDecoder`).
//!
//! WebSocket messages (server to browser; the browser sends nothing):
//! - binary: 9-byte header, then one Annex-B access unit. Byte 0 is
//!   flags (bit 0 = keyframe), bytes 1..9 are the pts in microseconds
//!   (u64, big endian), the rest is the H.264 data. A viewer gets
//!   video from its first keyframe on.
//! - text `{"type":"det", "frame_seq", "camera_id", "pts_ns", "pts_us",
//!   "coords", "items":[{"x","y","width","height","score","class_id"}]}`.
//!   `x`, `y`, `width`, `height` are normalised to 0..1 of the frame,
//!   `x`/`y` the top-left corner. `pts_us` equals the video header pts.
//! - text `{"type":"stats", ...}`: whatever `publish_stats` was given.
//! - text `{"type":"error","message":"..."}` followed by a close, when
//!   the viewer limit is reached.
//!
//! Role: Sink plus server. Pattern: the accept thread only adds viewers
//! to a hand-over list; every socket write happens on the `AsyncSink`
//! writer thread, under one total deadline per packet, so a slow
//! browser is dropped instead of stalling the pipeline (as in
//! `TcpH264Sink`). Frames are written by hand (server frames are
//! unmasked, so the framing is a few bytes); `tungstenite` does the
//! handshake.

use crate::write_within;
use edge_core::{Detections, EncodedPacket, Result, Sink};
use std::io::Read;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PAGE: &str = include_str!("web_ui.html");

/// Browsers connected at once; more get an error message and a close.
pub const MAX_VIEWERS: usize = 4;
/// Connections in the request/handshake phase at once.
const MAX_HANDSHAKES: usize = 8;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
const WRITE_TIMEOUT: Duration = Duration::from_millis(200);
/// Total time one packet may take across all viewers.
const PACKET_DEADLINE: Duration = Duration::from_secs(1);
const OP_TEXT: u8 = 0x1;
const OP_BINARY: u8 = 0x2;
const OP_CLOSE: u8 = 0x8;

struct Viewer {
    stream: TcpStream,
    peer: SocketAddr,
    synced: bool,
    stats_gen: u64,
    count: Arc<AtomicUsize>,
}

impl Drop for Viewer {
    fn drop(&mut self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
        self.count.fetch_sub(1, Ordering::Relaxed);
    }
}

struct Shared {
    pending: Mutex<Vec<Viewer>>,
    viewers: Arc<AtomicUsize>,
    handshakes: AtomicUsize,
    /// Generation counter and the latest stats message.
    stats: Mutex<(u64, Arc<str>)>,
    shutdown: AtomicBool,
}

/// Server handle: keep it alive for as long as the UI should be served.
pub struct WebUi {
    shared: Arc<Shared>,
    addr: SocketAddr,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl WebUi {
    pub fn bind(addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let shared = Arc::new(Shared {
            pending: Mutex::new(Vec::new()),
            viewers: Arc::new(AtomicUsize::new(0)),
            handshakes: AtomicUsize::new(0),
            stats: Mutex::new((0, Arc::from(""))),
            shutdown: AtomicBool::new(false),
        });
        let accept_shared = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("web-ui-accept".into())
            .spawn(move || accept_loop(&listener, &accept_shared))?;
        Ok(Self {
            shared,
            addr,
            thread: Some(thread),
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// The sink to put behind `AsyncSink`. Use one per `WebUi`.
    pub fn sink(&self) -> WebUiSink {
        WebUiSink {
            shared: Arc::clone(&self.shared),
            viewers: Vec::new(),
        }
    }

    /// Latest counters, as one complete JSON object with
    /// `"type":"stats"`. Sent to the viewers by the writer thread with
    /// its next packet; only the newest message is kept.
    pub fn publish_stats(&self, json: &str) {
        let mut stats = self.shared.stats.lock().expect("stats mutex poisoned");
        stats.0 += 1;
        stats.1 = Arc::from(json);
    }
}

impl Drop for WebUi {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn accept_loop(listener: &TcpListener, shared: &Arc<Shared>) {
    while !shared.shutdown.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, peer)) => spawn_handler(stream, peer, shared),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                log::error!("web_ui: accept failed: {e}");
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

fn spawn_handler(stream: TcpStream, peer: SocketAddr, shared: &Arc<Shared>) {
    if shared.handshakes.fetch_add(1, Ordering::Relaxed) >= MAX_HANDSHAKES {
        shared.handshakes.fetch_sub(1, Ordering::Relaxed);
        return;
    }
    let handler_shared = Arc::clone(shared);
    let spawned = std::thread::Builder::new()
        .name("web-ui-conn".into())
        .spawn(move || {
            if let Err(e) = handle_connection(stream, peer, &handler_shared) {
                log::warn!("web_ui: {peer}: {e}");
            }
            handler_shared.handshakes.fetch_sub(1, Ordering::Relaxed);
        });
    if let Err(e) = spawned {
        log::error!("web_ui: cannot spawn a handler: {e}");
        shared.handshakes.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Waits (up to the request timeout) until the first request line is
/// readable, without consuming it, and returns `(method, path)`.
fn peek_request_line(stream: &TcpStream) -> std::io::Result<(String, String)> {
    let start = Instant::now();
    let mut buf = [0u8; 1024];
    loop {
        let n = stream.peek(&mut buf)?;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        if let Some(end) = buf[..n].windows(2).position(|w| w == b"\r\n") {
            let line = String::from_utf8_lossy(&buf[..end]).into_owned();
            let mut parts = line.split_whitespace();
            let method = parts.next().unwrap_or("").to_string();
            let target = parts.next().unwrap_or("");
            let path = target.split('?').next().unwrap_or("").to_string();
            return Ok((method, path));
        }
        if n == buf.len() || start.elapsed() >= REQUEST_TIMEOUT {
            return Err(std::io::ErrorKind::InvalidData.into());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn handle_connection(stream: TcpStream, peer: SocketAddr, shared: &Shared) -> std::io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(REQUEST_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    stream.set_nodelay(true)?;
    let (method, path) = peek_request_line(&stream)?;
    match (method.as_str(), path.as_str()) {
        ("GET", "/ws") => upgrade(stream, peer, shared),
        ("GET", "/" | "/index.html") => respond(stream, "200 OK", "text/html; charset=utf-8", PAGE),
        _ => respond(stream, "404 Not Found", "text/plain", "not found\n"),
    }
}

fn respond(mut stream: TcpStream, status: &str, mime: &str, body: &str) -> std::io::Result<()> {
    // Consume the request head so closing does not reset the reply.
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") && head.len() < 8192 {
        if stream.read(&mut byte)? == 0 {
            break;
        }
        head.push(byte[0]);
    }
    let reply = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    write_within(&mut stream, reply.as_bytes(), PACKET_DEADLINE)
}

fn upgrade(stream: TcpStream, peer: SocketAddr, shared: &Shared) -> std::io::Result<()> {
    let ws = tungstenite::accept(stream).map_err(std::io::Error::other)?;
    let mut raw = ws.get_ref().try_clone()?;
    drop(ws);
    if shared.viewers.fetch_add(1, Ordering::Relaxed) >= MAX_VIEWERS {
        shared.viewers.fetch_sub(1, Ordering::Relaxed);
        let msg = format!(
            r#"{{"type":"error","message":"too many viewers (max {MAX_VIEWERS}); close another tab"}}"#
        );
        let _ = write_within(
            &mut raw,
            &ws_frame(OP_TEXT, msg.as_bytes()),
            PACKET_DEADLINE,
        );
        let _ = write_within(&mut raw, &ws_frame(OP_CLOSE, &[]), PACKET_DEADLINE);
        log::warn!("web_ui: refused {peer}: {MAX_VIEWERS} viewers already");
        return Ok(());
    }
    log::info!("web_ui: viewer {peer} connected; waiting for a keyframe");
    shared
        .pending
        .lock()
        .expect("pending mutex poisoned")
        .push(Viewer {
            stream: raw,
            peer,
            synced: false,
            stats_gen: 0,
            count: Arc::clone(&shared.viewers),
        });
    Ok(())
}

/// One unmasked, unfragmented server frame.
fn ws_frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 10);
    out.push(0x80 | opcode);
    match payload.len() {
        n if n < 126 => out.push(n as u8),
        n if n <= usize::from(u16::MAX) => {
            out.push(126);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        }
        n => {
            out.push(127);
            out.extend_from_slice(&(n as u64).to_be_bytes());
        }
    }
    out.extend_from_slice(payload);
    out
}

fn video_frame(pkt: &EncodedPacket) -> Vec<u8> {
    let mut payload = Vec::with_capacity(pkt.data.len() + 9);
    payload.push(u8::from(pkt.keyframe));
    payload.extend_from_slice(&(pkt.pts.as_micros() as u64).to_be_bytes());
    payload.extend_from_slice(&pkt.data);
    ws_frame(OP_BINARY, &payload)
}

/// The `Sink` half; created by `WebUi::sink`.
pub struct WebUiSink {
    shared: Arc<Shared>,
    viewers: Vec<Viewer>,
}

impl WebUiSink {
    fn adopt_pending(&mut self) {
        let mut pending = self.shared.pending.lock().expect("pending mutex poisoned");
        self.viewers.append(&mut pending);
    }

    /// Writes `frame` to every viewer that `wanted` accepts, within one
    /// total deadline; a viewer that fails is dropped.
    fn broadcast(&mut self, frame: &[u8], wanted: impl Fn(&Viewer) -> bool, deadline: Instant) {
        self.viewers.retain_mut(|v| {
            if !wanted(v) {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            match write_within(&mut v.stream, frame, left) {
                Ok(()) => true,
                Err(e) => {
                    log::warn!("web_ui: viewer {} dropped: {e}", v.peer);
                    false
                }
            }
        });
    }

    fn send_stats(&mut self, deadline: Instant) {
        let (generation, text) = {
            let stats = self.shared.stats.lock().expect("stats mutex poisoned");
            (stats.0, Arc::clone(&stats.1))
        };
        if generation == 0 || self.viewers.iter().all(|v| v.stats_gen == generation) {
            return;
        }
        let frame = ws_frame(OP_TEXT, text.as_bytes());
        self.broadcast(&frame, |v| v.stats_gen != generation, deadline);
        for v in &mut self.viewers {
            v.stats_gen = generation;
        }
    }
}

impl Sink for WebUiSink {
    fn push_video(&mut self, pkt: EncodedPacket) -> Result<()> {
        self.adopt_pending();
        if self.viewers.is_empty() {
            return Ok(());
        }
        let deadline = Instant::now() + PACKET_DEADLINE;
        if pkt.keyframe || self.viewers.iter().any(|v| v.synced) {
            let frame = video_frame(&pkt);
            let keyframe = pkt.keyframe;
            self.broadcast(&frame, |v| v.synced || keyframe, deadline);
            for v in &mut self.viewers {
                v.synced |= keyframe;
            }
        }
        self.send_stats(deadline);
        Ok(())
    }

    fn push_meta(&mut self, det: &Detections) -> Result<()> {
        if !self.viewers.iter().any(|v| v.synced) {
            return Ok(());
        }
        let Ok(mut value) = serde_json::to_value(det) else {
            return Ok(());
        };
        if let Some(obj) = value.as_object_mut() {
            obj.insert("type".into(), "det".into());
            obj.insert("pts_us".into(), (det.pts.as_micros() as u64).into());
            obj.insert(
                "coords".into(),
                "normalised 0..1 of the frame; x,y = top-left corner".into(),
            );
        }
        let frame = ws_frame(OP_TEXT, value.to_string().as_bytes());
        self.broadcast(&frame, |v| v.synced, Instant::now() + PACKET_DEADLINE);
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.adopt_pending();
        let close = ws_frame(OP_CLOSE, &[]);
        for v in &mut self.viewers {
            let _ = write_within(&mut v.stream, &close, WRITE_TIMEOUT);
        }
        self.viewers.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edge_core::Detection;
    use std::io::Write;
    use tungstenite::Message;

    fn packet(keyframe: bool, tag: u8) -> EncodedPacket {
        EncodedPacket::new(
            Duration::from_micros(1000 * u64::from(tag)),
            keyframe,
            vec![tag; 5],
        )
    }

    fn connect(ui: &WebUi) -> tungstenite::WebSocket<TcpStream> {
        let stream = TcpStream::connect(ui.local_addr()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .unwrap();
        let url = format!("ws://{}/ws", ui.local_addr());
        tungstenite::client(url, stream).expect("handshake").0
    }

    /// Pushes a delta frame until the sink has adopted a viewer.
    fn push_until_adopted(sink: &mut WebUiSink) {
        for _ in 0..400 {
            sink.push_video(packet(false, 1)).unwrap();
            if !sink.viewers.is_empty() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("viewer never adopted");
    }

    #[test]
    fn serves_the_page_and_404() {
        let ui = WebUi::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let fetch = |path: &str| {
            let mut s = TcpStream::connect(ui.local_addr()).unwrap();
            write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).unwrap();
            out
        };
        let ok = fetch("/");
        assert!(ok.starts_with("HTTP/1.1 200 OK"), "{ok}");
        assert!(ok.contains("text/html") && ok.contains("<canvas"), "{ok}");
        assert!(fetch("/nope").starts_with("HTTP/1.1 404"));
    }

    #[test]
    fn viewer_gets_video_from_the_first_keyframe_and_json() {
        let ui = WebUi::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let mut sink = ui.sink();
        let mut ws = connect(&ui);
        push_until_adopted(&mut sink);
        sink.push_video(packet(false, 2)).unwrap();
        sink.push_video(packet(true, 3)).unwrap();
        let mut det = Detections::with_capacity(1);
        det.frame_seq = 7;
        det.pts = Duration::from_micros(3000);
        det.items.push(Detection {
            x: 0.25,
            y: 0.5,
            width: 0.1,
            height: 0.2,
            score: 0.9,
            class_id: 2,
        });
        sink.push_meta(&det).unwrap();
        ui.publish_stats(r#"{"type":"stats","captured":1}"#);
        sink.push_video(packet(false, 4)).unwrap();

        let Message::Binary(key) = ws.read().unwrap() else {
            panic!("expected binary");
        };
        assert_eq!(key[0], 1, "keyframe flag");
        assert_eq!(u64::from_be_bytes(key[1..9].try_into().unwrap()), 3000);
        assert_eq!(&key[9..], &[3; 5]);
        let Message::Text(text) = ws.read().unwrap() else {
            panic!("expected det text");
        };
        let v: serde_json::Value = serde_json::from_str(text.as_str()).unwrap();
        assert_eq!(v["type"], "det");
        assert_eq!(v["pts_us"], 3000);
        assert_eq!(v["items"][0]["class_id"], 2);
        let Message::Binary(delta) = ws.read().unwrap() else {
            panic!("expected binary");
        };
        assert_eq!((delta[0], &delta[9..]), (0, &[4u8; 5][..]));
        let Message::Text(stats) = ws.read().unwrap() else {
            panic!("expected stats");
        };
        assert!(stats.as_str().contains(r#""type":"stats""#));
    }

    #[test]
    fn a_viewer_that_never_reads_is_dropped_within_the_deadline() {
        let ui = WebUi::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let mut sink = ui.sink();
        let _ws = connect(&ui);
        push_until_adopted(&mut sink);
        let start = Instant::now();
        let big = EncodedPacket::new(Duration::ZERO, true, vec![7; 64 << 20]);
        sink.push_video(big).unwrap();
        assert!(start.elapsed() < Duration::from_millis(1900), "bounded");
        assert!(sink.viewers.is_empty(), "stalled viewer must be dropped");
        assert_eq!(ui.shared.viewers.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn the_fifth_viewer_is_refused_with_a_message() {
        let ui = WebUi::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let mut sink = ui.sink();
        let _held: Vec<_> = (0..MAX_VIEWERS).map(|_| connect(&ui)).collect();
        for _ in 0..400 {
            sink.push_video(packet(false, 1)).unwrap();
            if sink.viewers.len() == MAX_VIEWERS {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(sink.viewers.len(), MAX_VIEWERS);
        let mut extra = connect(&ui);
        let Message::Text(text) = extra.read().unwrap() else {
            panic!("expected error text");
        };
        assert!(text.as_str().contains("too many viewers"), "{text}");
    }

    #[test]
    fn the_page_script_parses() {
        let start = PAGE.find("<script>").expect("script") + "<script>".len();
        let end = PAGE.rfind("</script>").expect("script end");
        let path = std::env::temp_dir().join(format!("web_ui_check_{}.js", std::process::id()));
        std::fs::write(&path, &PAGE[start..end]).unwrap();
        match std::process::Command::new("node")
            .arg("--check")
            .arg(&path)
            .output()
        {
            Ok(out) => assert!(
                out.status.success(),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            ),
            Err(e) => eprintln!("node not available, JS syntax not checked: {e}"),
        }
        let _ = std::fs::remove_file(path);
    }
}
