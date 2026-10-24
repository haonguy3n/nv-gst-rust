#![cfg(feature = "trt")]
//! Checks of the TensorRT, CUDA and VIC pieces that need the Orin. All are
//! `#[ignore]`: run them on the board with
//! `cargo test -p platform-jetson --features trt -- --ignored --test-threads=1`.
//! None needs Argus or a camera. `EDGE_TEST_ONNX` names an ONNX model for the
//! engine test (skipped with a message when unset); `EDGE_TEST_CACHE` is the
//! plan cache folder (default: a folder under the temp dir).

use edge_core::{EngineState, InferenceEngine, Precision};
use gstreamer::prelude::*;
use platform_jetson::cuda::{new_gpu_stream, CudaStream, GpuTimer};
use platform_jetson::trt_logic::ModelSource;
use platform_jetson::vic::VicScaler;
use platform_jetson::warmup::{TrtWarmup, WarmupRequest};
use std::path::PathBuf;
use std::time::Duration;

fn cache_dir() -> PathBuf {
    match std::env::var("EDGE_TEST_CACHE") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => std::env::temp_dir().join("edge-test-cache"),
    }
}

fn load_engine(onnx: &str) -> Box<dyn InferenceEngine> {
    let request = WarmupRequest {
        source: ModelSource::from_path(std::path::Path::new(onnx)).expect("model path"),
        precision: Precision::Fp32,
        cache_dir: cache_dir(),
        soc: "device-test".into(),
        l4t: "device-test".into(),
        use_graph: true,
    };
    let warmup = TrtWarmup::spawn(request).expect("spawn the warm-up");
    EngineState::Warming(Box::new(warmup))
        .wait_ready(Duration::from_secs(30 * 60))
        .expect("the plan builds and loads")
}

#[test]
#[ignore = "needs the Orin GPU"]
fn cuda_stream_creates_and_synchronizes() {
    let stream = CudaStream::new().expect("create a stream");
    stream.synchronize().expect("synchronize an idle stream");
    let wrapped = new_gpu_stream().expect("a GpuStream");
    assert!(wrapped.backend::<CudaStream>().is_some());
}

#[test]
#[ignore = "needs the Orin GPU"]
fn cuda_events_time_a_span_on_a_stream() {
    let stream = CudaStream::new().expect("create a stream");
    let timer = GpuTimer::new().expect("create the events");
    for _ in 0..3 {
        timer.begin(stream.inner()).expect("record start");
        timer.finish(stream.inner()).expect("record end");
        stream.synchronize().expect("synchronize");
        // An empty span: only that the read works and is sane.
        let span = timer.elapsed().expect("elapsed time");
        assert!(span < Duration::from_secs(1), "{span:?}");
    }
}

#[test]
#[ignore = "needs the Orin GPU and EDGE_TEST_ONNX"]
fn engine_infers_twice_with_gpu_time() {
    let Ok(onnx) = std::env::var("EDGE_TEST_ONNX") else {
        eprintln!("skipped: set EDGE_TEST_ONNX to an ONNX model");
        return;
    };
    let mut engine = load_engine(&onnx);
    let stream = new_gpu_stream().expect("stream");
    assert!(
        engine.last_gpu_time().is_none(),
        "no time before an inference"
    );
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    // The first run enqueues and captures the CUDA graph; the second replays it.
    for run in 0..2 {
        engine.infer(&stream).expect("infer on a zero input");
        let gpu = engine.last_gpu_time().expect("GPU time after infer");
        assert!(
            gpu > Duration::ZERO && gpu < Duration::from_secs(5),
            "run {run}: {gpu:?}"
        );
        let now: Vec<Vec<f32>> = engine
            .output_bindings()
            .iter()
            .map(|t| t.as_f32_slice().expect("host view").to_vec())
            .collect();
        assert!(!now.is_empty(), "the model has outputs");
        if run == 1 {
            assert_eq!(now, outputs, "the graph replay gives the same outputs");
        }
        outputs = now;
    }
    for (tensor, data) in engine.output_bindings().iter().zip(&outputs) {
        assert_eq!(
            data.len(),
            tensor.spec.num_elements(),
            "{}",
            tensor.spec.name
        );
        assert!(data.iter().all(|v| v.is_finite()), "{}", tensor.spec.name);
    }
    let input = &engine.input_bindings()[0];
    assert_eq!(
        input.as_f32_slice().expect("input view").len(),
        input.spec.num_elements()
    );
}

/// A solid mid-grey NV12 NVMM frame from a test source, without Argus.
fn nvmm_grey_frame(width: u32, height: u32) -> (gstreamer::Pipeline, gstreamer::Buffer) {
    gstreamer::init().expect("gst init");
    let launch = format!(
        "videotestsrc num-buffers=5 pattern=solid-color foreground-color=0xff808080 \
         ! video/x-raw,width={width},height={height},framerate=30/1 ! nvvidconv \
         ! video/x-raw(memory:NVMM),format=NV12 ! appsink name=sink sync=false max-buffers=1 drop=true"
    );
    let pipeline = gstreamer::parse::launch(&launch)
        .expect("parse")
        .downcast::<gstreamer::Pipeline>()
        .expect("a pipeline");
    let sink = pipeline
        .by_name("sink")
        .expect("sink")
        .downcast::<gstreamer_app::AppSink>()
        .expect("appsink");
    pipeline.set_state(gstreamer::State::Playing).expect("play");
    let sample = sink
        .try_pull_sample(gstreamer::ClockTime::from_seconds(10))
        .expect("an NVMM frame within 10 s");
    (pipeline, sample.buffer_owned().expect("a buffer"))
}

#[test]
#[ignore = "needs the Orin VIC"]
fn vic_letterbox_pads_and_fills_the_input_tensor() {
    use nvbuf_sys::planar::{rgba_to_planar, PlanarSpec};
    use nvbuf_sys::SurfaceRef;
    const PAD: u8 = 114;
    let (dst_w, dst_h) = (416usize, 320usize);
    let (pipeline, buffer) = nvmm_grey_frame(640, 360);
    let map = buffer.map_readable().expect("map the NVMM frame");
    let src = SurfaceRef::from_mapped_bytes(map.as_slice()).expect("surface");
    let mut vic = VicScaler::new(dst_w as u32, dst_h as u32, PAD).expect("VIC scaler");
    let lb = vic.scale(&src).expect("scale");
    assert!(lb.pad_y > 0, "640x360 into 416x320 leaves bars: {lb:?}");

    let mut tensor = trt_sys::MappedBuffer::alloc(3 * dst_w * dst_h).expect("input tensor");
    let plane = vic.dst().map_read().expect("read the scaled surface");
    let spec = PlanarSpec {
        swap_rb: false,
        scale: 1.0,
    };
    rgba_to_planar(
        plane.data,
        plane.pitch,
        dst_w,
        dst_h,
        spec,
        tensor.as_mut_slice(),
    )
    .expect("planar write");

    let data = tensor.as_slice();
    assert!(data.iter().all(|v| v.is_finite()));
    for channel in 0..3 {
        let at = |x: usize, y: usize| data[channel * dst_w * dst_h + y * dst_w + x];
        assert_eq!(at(10, 0), f32::from(PAD), "top bar, channel {channel}");
        assert_eq!(
            at(dst_w - 1, dst_h - 1),
            f32::from(PAD),
            "bottom bar, channel {channel}"
        );
        let centre = at(dst_w / 2, dst_h / 2);
        assert!(
            (centre - 128.0).abs() < 12.0,
            "content, channel {channel}: {centre}"
        );
    }
    // Return the NVMM buffer before the teardown: a held pool buffer
    // deadlocks the Null state change (seen on the Nano).
    drop(map);
    drop(buffer);
    let _ = pipeline.set_state(gstreamer::State::Null);
}
