#!/usr/bin/env bash
# push_face_assets.sh HOST [ASSET_DIR]
# Copy the private Avatar1 face photos to a pod and unpack them where the live server expects them:
#   /workspace/avatar1/sets/1/*.jpg   angle set for key 1 (slicked silver ponytail, many angles)
#   /workspace/avatar1/sets/2/*.jpg   angle set for key 2 (messy silver bun, many angles)
#   /workspace/avatar1/clean/clean_0*.jpg, /workspace/avatar1/necktattoo_front.jpg   keys 3/4/5
# HOST is an ssh alias/target for the pod; extra ssh/scp options via SSH_OPTS="-F ~/.ssh/pod.cfg".
# ASSET_DIR defaults to the box copy /workspace/fs/v2/face_assets (never committed to git).
set -euo pipefail
HOST=${1:?ssh host alias}; SRC=${2:-/workspace/fs/v2/face_assets}
tar -czf /tmp/face_assets.tar.gz -C "$SRC" .
scp ${SSH_OPTS:-} -q /tmp/face_assets.tar.gz "$HOST":/workspace/face_assets.tar.gz
ssh ${SSH_OPTS:-} "$HOST" 'mkdir -p /workspace/avatar1 && tar -xzf /workspace/face_assets.tar.gz -C /workspace/avatar1 && ls /workspace/avatar1/sets/*| head -50'
