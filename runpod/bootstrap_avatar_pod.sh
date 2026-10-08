#!/usr/bin/env bash
# bootstrap_avatar_pod.sh - prepare a Runpod GPU pod for the Avatar1 LivePortrait pipeline.
#
# Target: Runpod Secure Cloud pod from the official "ComfyUI - CUDA 12.8" template (cw3nka7d08,
# image runpod/comfyui:*-cuda12.8). ComfyUI keeps running on :8188; this script only adds:
#   /workspace/funnymemelivefacebodyswap   <- this repo (clone or fast-forward pull)
#   /workspace/avatar1/{source,driving,out} <- Avatar1 reference photos, driving clips, results
#   /workspace/LivePortrait                 <- KlingTeam/KwaiVGI LivePortrait + its own Python 3.10 venv
#   /workspace/LivePortrait/pretrained_weights <- human-mode weights from Hugging Face
#   /workspace/run_lp_still.sh              <- helper for a still-image LivePortrait pass
#
# Usage (on the pod, as root):
#   curl -fsSL https://raw.githubusercontent.com/pjrny/funnymemelivefacebodyswap/main/runpod/bootstrap_avatar_pod.sh | bash
#   # or: bash /workspace/funnymemelivefacebodyswap/runpod/bootstrap_avatar_pod.sh [--smoke]
#
#   --smoke   after install, run LivePortrait once on its bundled example (s9.jpg + d0.mp4)
#
# Safe to re-run: every step is skipped when already done.
set -euo pipefail

WS=/workspace
REPO_URL=${REPO_URL:-https://github.com/pjrny/funnymemelivefacebodyswap}
REPO_DIR=$WS/funnymemelivefacebodyswap
LP_DIR=$WS/LivePortrait
LP_URL=${LP_URL:-https://github.com/KwaiVGI/LivePortrait}
AV_DIR=$WS/avatar1
LOG_DIR=$WS/logs
SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

mkdir -p "$LOG_DIR" "$AV_DIR"/{source,driving,out}
exec > >(tee -a "$LOG_DIR/bootstrap.log") 2>&1
log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none detected')"

# 1. system packages (ffmpeg is needed by LivePortrait for video I/O)
if ! command -v ffmpeg >/dev/null || ! command -v git >/dev/null; then
  log "installing ffmpeg/git"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git >/dev/null
fi

# 2. project repo
if [[ -d $REPO_DIR/.git ]]; then
  log "updating $REPO_DIR"; git -C "$REPO_DIR" pull --ff-only || log "WARN: pull failed, keeping local copy"
else
  log "cloning $REPO_URL"; rm -rf "$REPO_DIR.tmp"; git clone --depth 1 "$REPO_URL" "$REPO_DIR.tmp"
  # keep anything already placed in the folder (e.g. a README written before the clone)
  if [[ -d $REPO_DIR ]]; then cp -rn "$REPO_DIR"/. "$REPO_DIR.tmp"/ 2>/dev/null || true; rm -rf "$REPO_DIR"; fi
  mv "$REPO_DIR.tmp" "$REPO_DIR"
fi

# 3. uv (fast pip + standalone Python 3.10, which LivePortrait's pinned deps expect)
if ! command -v uv >/dev/null; then
  log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"

# 4. LivePortrait source + venv
[[ -d $LP_DIR/.git ]] || { log "cloning LivePortrait"; git clone --depth 1 "$LP_URL" "$LP_DIR"; }
cd "$LP_DIR"
if [[ ! -x .venv/bin/python ]]; then
  log "creating venv (Python 3.10)"; uv venv --python 3.10 .venv
fi
PY=$LP_DIR/.venv/bin/python
if ! "$PY" -c "import torch, cv2, onnxruntime, tyro" 2>/dev/null; then
  log "installing torch 2.4.1 (cu121; bundles cuDNN 9) + LivePortrait requirements"
  uv pip install --python "$PY" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
  # pin albucore to the release that matched albumentations 1.4.10; newer ones pull numkong,
  # which has no wheel here and takes 15+ min to compile from source
  echo "albucore==0.0.13" > /tmp/lp-constraints.txt
  uv pip install --python "$PY" -r requirements.txt -c /tmp/lp-constraints.txt
  # requirements pin onnxruntime-gpu 1.18.0 (CUDA 11 build); swap in a CUDA 12 / cuDNN 9 build
  uv pip install --python "$PY" "onnxruntime-gpu==1.19.2" "huggingface_hub[cli]<1"
fi

# 5. human-mode weights (~0.6 GB). Hugging Face repo moved from KwaiVGI to KlingTeam; try both.
if [[ ! -f pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth ]]; then
  log "downloading LivePortrait weights"
  "$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
last = None
for repo in ("KlingTeam/LivePortrait", "KwaiVGI/LivePortrait"):
    try:
        snapshot_download(repo_id=repo, local_dir="pretrained_weights",
                          ignore_patterns=["*.git*", "README.md", "docs/*", "liveportrait_animals/*"])
        print("weights from", repo); break
    except Exception as e:  # noqa: BLE001
        last = e; print("failed", repo, e)
else:
    raise SystemExit(f"weight download failed: {last}")
PYEOF
fi

# 6. helper for still-image runs (adds the venv's CUDA/cuDNN libs so onnxruntime uses the GPU)
cat > $WS/run_lp_still.sh <<'RUNEOF'
#!/usr/bin/env bash
# run_lp_still.sh SOURCE_IMG DRIVING_IMG_OR_VIDEO [extra inference.py flags]
#   SOURCE  = the Avatar1 photo to animate   (e.g. /workspace/avatar1/source/clean_front.jpg)
#   DRIVING = Oscar's webcam frame or clip    (e.g. /workspace/avatar1/driving/oscar_frame.jpg)
# Results land in /workspace/avatar1/out/
set -euo pipefail
SRC=${1:?source image}; DRV=${2:?driving image or video}; shift 2
cd /workspace/LivePortrait
SP=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"
exec .venv/bin/python inference.py -s "$SRC" -d "$DRV" -o /workspace/avatar1/out "$@"
RUNEOF
chmod +x $WS/run_lp_still.sh

log "versions: $("$PY" -c 'import torch, onnxruntime as o; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| ort", o.__version__, o.get_available_providers())')"

if [[ $SMOKE == 1 ]]; then
  log "smoke test on bundled example"
  time $WS/run_lp_still.sh assets/examples/source/s9.jpg assets/examples/driving/d0.mp4
  ls -la $AV_DIR/out
fi

cat <<'DONE'
================ bootstrap complete ================
Upload Avatar1 photos to /workspace/avatar1/source/ and a webcam frame/clip of Oscar to
/workspace/avatar1/driving/, then run a still-image LivePortrait pass:

  /workspace/run_lp_still.sh /workspace/avatar1/source/<avatar1_photo>.jpg \
                             /workspace/avatar1/driving/<oscar_frame>.jpg

Useful flags (python inference.py -h for all):
  --flag_do_torch_compile        20-30% faster after a ~1 min warm-up
  --no_flag_relative_motion      copy the absolute pose of the driving frame
  --flag_crop_driving_video      auto-crop the face in a wide webcam frame
  --animation_region exp         move only expression (keeps Avatar1's head pose; less tattoo smear)
Output: /workspace/avatar1/out/<source>--<driving>.jpg|mp4 (+ _concat side-by-side)
Logs:   /workspace/logs/bootstrap.log
=====================================================
DONE
