# Building with Yocto (kas)

The device image is built from the camera-viber repository
(`orin-nx-camera-viber`, `yocto/kas`). Its layer `meta-vc-camera` has the
recipe `nv-gst-rust` (`recipes-apps/nv-gst-rust/`), which builds this
checkout in place:

- `edge-vision` only (`apps/edge-vision`), with `--features jetson`
  (the GStreamer runtime). A virtual workspace root takes no
  `--features`, so the recipe points cargo at the app's manifest and the
  workspace's `Cargo.lock`.
- Installs `/usr/bin/edge-vision` and `config/orin-nx-yolo.toml`,
  `config/x86-dev.toml` in `/etc/edge-vision/`.
- `onnx` is not built: `ort`'s `download-binaries` fetches ONNX Runtime
  in its build script, and a Yocto `do_compile` has no network.
- Toolchain: OE-core's Rust 1.98.1 and `cargo` class; every crate is
  fetched offline from `Cargo.lock` in `do_fetch`.
- cargo's `target/` goes into the Yocto work directory, never into this
  checkout's `target/`.

The recipe is in no image: the cameras belong to camera-streamer or
lpr-app there.

## Point the build at this checkout

`NV_GST_RUST_SRC` is this checkout's absolute path; unset, the recipe is
skipped. Set it in camera-viber's `local.yml` (gitignored, at its root);
with `yb`, also mount the checkout into the build container (`yb:`
mounts):

```yaml
local_conf_header:
  nv-gst-rust: |
    NV_GST_RUST_SRC = "/home/<you>/nv-gst-rust"
```

## Build

From the camera-viber repository root:

```sh
kas build local.yml --target nv-gst-rust     # or: yb build local.yml --target nv-gst-rust
ls build/tmp/deploy/ipk/aarch64/nv-gst-rust_*.ipk
```

For one run without a `local.yml` section, pass the path through the
environment (bitbake drops variables that `BB_ENV_PASSTHROUGH_ADDITIONS`
does not name):

```sh
kas shell yocto/kas/orin-nx.yml -c 'export BB_ENV_PASSTHROUGH_ADDITIONS=NV_GST_RUST_SRC NV_GST_RUST_SRC=/home/<you>/nv-gst-rust; bitbake nv-gst-rust'
```

Use `yocto/kas/orin-nano.yml` for the Orin Nano machine.

## Put it in a development image

Add to the same `local.yml` section, then build the image as usual
(`kas build local.yml`). Development images only; adding it to a release
image is an image-scope decision for camera-viber.

```yaml
    IMAGE_INSTALL:append = " nv-gst-rust"
```

The encoder plugin comes from the image per machine: `nvv4l2h264enc` /
`nvv4l2h265enc` on the NX, `x264enc` on the Nano (no NVENC). The recipe
itself pulls in the `app`, `nvarguscamerasrc` and `nvvidconv` plugins.

## When `Cargo.lock` changes

The recipe's crate list (`nv-gst-rust-crates.inc` in camera-viber, every
crate with its checksum) must match `Cargo.lock`, or `do_fetch` /
`cargo --frozen` fails. Regenerate it and commit it in camera-viber:

```sh
kas shell local.yml -c 'bitbake -c update_crates nv-gst-rust'
```

The recipe reads only the workspace's `Cargo.lock`; `.cargo-home/` and
`target/` here hold others, which would otherwise end up in the list.

A new crate that compiles C or links a system library needs that
library in the recipe's `DEPENDS`; a new cargo feature needs a
`PACKAGECONFIG` entry there.

## TensorRT engine on the image

The image does not ship `trtexec`, and the service sandbox will not let
the app run a shell. The product path is to build the plan in-process
through `trt-sys` and cache it under `/var/cache/edge-vision` (the
systemd unit needs `CacheDirectory=edge-vision`). `tools/build-engine.sh`
is a bring-up tool for a development image only; see
[jetson-bringup.md](jetson-bringup.md) for its limits (256 MiB
workspace, build lock, memory check, `EDGE_VISION_CACHE`).

The recipe currently installs the binary and the two configs, not the
model. `config/orin-nx-yolo.toml` expects
`/data/models/yolox-tiny.onnx`; install the ONNX model there (or extend
the recipe with a model package) before the profile can run.

## Status

Built with `bitbake nv-gst-rust` for the Orin NX machine on 2026-10-24,
package QA clean. Not run on a device. The recipe is a packaging step
only: the device stages (NVMM, TensorRT, NVENC, CPU encode) are still
M1 work.
