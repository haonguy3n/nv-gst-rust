#![cfg(feature = "cuda-preprocess")]
//! T9 on the Orin: the CUDA tensor writer equals the CPU writer on the same
//! VIC output. `#[ignore]`: run on the board with
//! `cargo test -p platform-jetson --features cuda-preprocess --test device_cuda_preprocess -- --ignored`.
//! No Argus or camera is needed (videotestsrc through nvvidconv).

use gstreamer::prelude::*;
use nvbuf_sys::planar::{rgba_to_planar, PlanarSpec};
use nvbuf_sys::SurfaceRef;
use platform_jetson::cuda::new_gpu_stream;
use platform_jetson::cuda_preprocess::CudaTensorWriter;
use platform_jetson::vic::VicScaler;

/// An SMPTE-bars NV12 NVMM frame from a test source: many different colours,
/// so a swapped channel or a wrong row would show.
fn nvmm_bars_frame(width: u32, height: u32) -> (gstreamer::Pipeline, gstreamer::Buffer) {
    gstreamer::init().expect("gst init");
    let launch = format!(
        "videotestsrc num-buffers=5 pattern=smpte \
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
#[ignore = "needs the Orin GPU and VIC"]
fn cuda_tensor_equals_the_cpu_tensor() {
    const PAD: u8 = 114;
    let (dst_w, dst_h) = (416usize, 320usize);
    let (pipeline, buffer) = nvmm_bars_frame(640, 360);
    let map = buffer.map_readable().expect("map the NVMM frame");
    let src = SurfaceRef::from_mapped_bytes(map.as_slice()).expect("surface");
    let mut vic = VicScaler::new(dst_w as u32, dst_h as u32, PAD).expect("VIC scaler");
    vic.scale(&src).expect("scale");
    let stream = new_gpu_stream().expect("stream");
    let len = 3 * dst_w * dst_h;
    let mut cpu = trt_sys::MappedBuffer::alloc(len).expect("cpu tensor");
    let mut gpu = trt_sys::MappedBuffer::alloc(len).expect("gpu tensor");

    for (swap_rb, scale) in [
        (false, 1.0),
        (true, 1.0),
        (false, 1.0 / 255.0),
        (true, 1.0 / 255.0),
    ] {
        let spec = PlanarSpec { swap_rb, scale };
        let plane = vic.dst().map_read().expect("read the scaled surface");
        rgba_to_planar(
            plane.data,
            plane.pitch,
            dst_w,
            dst_h,
            spec,
            cpu.as_mut_slice(),
        )
        .expect("planar write");
        let writer = CudaTensorWriter::try_new(&vic, spec).expect("CUDA maps the VIC surface");
        gpu.as_mut_slice().fill(-1.0);
        writer
            .write(gpu.as_mut_slice(), &stream)
            .expect("CUDA write");
        let worst = cpu
            .as_slice()
            .iter()
            .zip(gpu.as_slice())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-6,
            "swap_rb={swap_rb} scale={scale}: max diff {worst}"
        );
        assert!(cpu.as_slice().iter().any(|v| *v != cpu.as_slice()[0]));
    }
    // Return the NVMM buffer before the teardown: a held pool buffer
    // deadlocks the Null state change (seen on the Nano).
    drop(map);
    drop(buffer);
    let _ = pipeline.set_state(gstreamer::State::Null);
}
