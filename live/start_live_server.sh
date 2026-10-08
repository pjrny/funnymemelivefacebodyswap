#!/usr/bin/env bash
# Start (or restart) the realtime Avatar1 server on the pod. Logs: /workspace/logs/live_server.log
# Usage: start_live_server.sh [--source /workspace/avatar1/tattoo_front.jpg] [--region exp|all] [--out_w 960 --out_h 540]
set -euo pipefail
pkill -f pod_live_server.py 2>/dev/null || true
# Runpod's FileBrowser sits on 8080 (the only spare proxied HTTP port). Stop it; restart later with:
#   cd /workspace/runpod-slim && nohup filebrowser >/dev/null 2>&1 &
pkill -x filebrowser 2>/dev/null || true
sleep 1
cd /workspace/LivePortrait
SP=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
mkdir -p /workspace/logs
nohup .venv/bin/python /workspace/live/pod_live_server.py "$@" > /workspace/logs/live_server.log 2>&1 &
echo "started pid $!  ->  https://${RUNPOD_POD_ID:-93x3w1npbz7g4t}-8080.proxy.runpod.net/health"
