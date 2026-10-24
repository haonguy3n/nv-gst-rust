#!/bin/sh
# edge-vision's root helper: one request per connection (edge-vision-control.socket), answered "ok" or why not. The
# image has no polkit and systemd lets only root restart units, so this is the one privileged action the unprivileged
# app can ask for (apps/edge-vision/src/recovery.rs).
read -r request || exit 1
case "$request" in
    restart-capture)
        systemctl reset-failed nvargus-daemon.service > /dev/null 2>&1
        systemctl --no-block restart nvargus-daemon.service
        ;;
    *)
        echo "unknown request: $request"
        false
        ;;
esac 2>&1 && echo ok
