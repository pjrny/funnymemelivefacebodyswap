# Runpod setup for Avatar1 (LivePortrait)

This folder sets up a Runpod GPU pod for the live face/body swap avatar pipeline.

## Pod
- Runpod **Secure Cloud** only, using the official template `cw3nka7d08` ("ComfyUI - CUDA 12.8", image `runpod/comfyui:*-cuda12.8`)
- GPU: 1x RTX 4090 (24 GB). First pod `avatar-comfyui-01` runs in US-IL-1 at about $0.74/hr
- Ports: `8188/http` (ComfyUI), `8080/http`, `8888/http` (Jupyter), `22/tcp` (SSH)
- 100 GB container disk and no network volume. **Everything on the pod is lost when it terminates**
- The pod is created with `terminateAfter` set about 4 h out so billing can't run away. An in-pod watchdog is a backup

## Bootstrap (run on the pod)
```bash
curl -fsSL https://raw.githubusercontent.com/pjrny/funnymemelivefacebodyswap/main/runpod/bootstrap_avatar_pod.sh | bash -s -- --smoke
```
This clones the repo to `/workspace/funnymemelivefacebodyswap`, installs LivePortrait into
`/workspace/LivePortrait` with its own Python 3.10 venv (torch 2.4.1 cu121 and onnxruntime-gpu 1.19.2), downloads the
human-mode weights, creates `/workspace/avatar1/{source,driving,out}`, and writes `/workspace/run_lp_still.sh`.

## Upload Avatar1 from the laptop (Windows PowerShell)
```powershell
scp -P <port> -r "C:\Users\oscar\OneDrive\Documents\AIWorkspace-Shared\kimi\Workspaces\FaceBodySwapStream\facebodyrealtimediffusion\Avatar1\*" root@<pod-ip>:/workspace/avatar1/source/
scp -P <port> .\testdata\webcam_frame.jpg root@<pod-ip>:/workspace/avatar1/driving/oscar_frame.jpg
```
(`<pod-ip>`/`<port>` come from the Runpod console Connect tab, or `runpodctl ssh info <pod-id>`.)

## First LivePortrait still
```bash
/workspace/run_lp_still.sh /workspace/avatar1/source/<avatar1_photo>.jpg /workspace/avatar1/driving/oscar_frame.jpg
# less tattoo smear: keep Avatar1's head pose, transfer expression only
/workspace/run_lp_still.sh <src> <drv> --animation_region exp
```
Results go to `/workspace/avatar1/out/`. Copy them back with `scp -P <port> root@<pod-ip>:/workspace/avatar1/out/* .`

## Stop or terminate
```bash
runpodctl pod stop <pod-id>      # releases the GPU (container disk is wiped)
runpodctl pod delete <pod-id>    # terminate and stop billing completely
```

See also the project docs: [USAGE](../docs/USAGE.md), [TROUBLESHOOTING](../docs/TROUBLESHOOTING.md), [HOW_WE_BUILT_IT](../docs/HOW_WE_BUILT_IT.md).
