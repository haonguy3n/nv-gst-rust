#!/bin/sh
# Entry point of tools/Dockerfile.jetson-sdk: picks the Yocto SDK (/opt/yocto-sdk)
# or the plain sysroot (/sysroot) and exports the cross environment.
set -eu

if env_file=$(ls /opt/yocto-sdk/environment-setup-* 2>/dev/null | head -1) && [ -n "$env_file" ]; then
    set +u  # the SDK script reads unset variables
    # shellcheck disable=SC1090
    . "$env_file"
    set -u
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="${CC%% *}"
    export CC_aarch64_unknown_linux_gnu="$CC" CXX_aarch64_unknown_linux_gnu="$CXX"
    export SYSROOT="$SDKTARGETSYSROOT"
elif [ -d /sysroot/usr ]; then
    export SYSROOT=/sysroot PKG_CONFIG_SYSROOT_DIR=/sysroot
    export PKG_CONFIG_ALLOW_CROSS=1
    export PKG_CONFIG_LIBDIR=/sysroot/usr/lib/aarch64-linux-gnu/pkgconfig:/sysroot/usr/lib/pkgconfig:/sysroot/usr/share/pkgconfig
    export CC_aarch64_unknown_linux_gnu="aarch64-linux-gnu-gcc --sysroot=/sysroot"
    export CXX_aarch64_unknown_linux_gnu="aarch64-linux-gnu-g++ --sysroot=/sysroot"
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=--sysroot=/sysroot -C link-arg=-Wl,-rpath-link,/sysroot/usr/lib/aarch64-linux-gnu"
fi
exec "$@"
