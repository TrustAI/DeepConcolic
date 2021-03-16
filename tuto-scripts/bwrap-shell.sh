#!/usr/bin/env bash
#
# Adapted from https://github.com/containers/bubblewrap/blob/master/demos/bubblewrap-shell.sh
#
set -euo pipefail
exec bwrap --ro-bind / / \
     --bind $PWD $PWD \
     --bind /tmp /tmp \
     --proc /proc \
     --dev /dev \
     --share-net \
     --die-with-parent \
     /bin/sh -c "exec $*"
