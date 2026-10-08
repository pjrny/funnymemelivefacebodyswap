# How we built it

A chronological walkthrough of Avatar1, Sep 24 – Oct 8, 2026.

## The goal

Look like Avatar1 (silver hair, freckles, optional face and neck tattoos) on camera during
Zoom or Google Meet calls, through the OBS Virtual Camera. Funny meme first, useful for
calls second.

## Hardware reality check

Neither of Oscar's machines has an NVIDIA GPU:

| Machine | Specs | Result |
|---|---|---|
| HP ENVY laptop | i7-1255U, Iris Xe, 64 GB | Where the work started |
| Lenovo desktop | i7-9700, UHD 630 | Same limitation |

Cloudflare cannot rent GPUs for this. Runpod Secure Cloud was the pick: SOC 2 data centers,
hourly billing, terminate when done. Community Cloud and Vast.ai were rejected because they
rent other people's machines.

## Phase 1 — Laptop experiments (late Sep)

### Basic face swap failed
Early `inswapper`-style swaps left beard ghosts and hollow eyes. Identity looked wrong and
the result was not Meet-ready.

### Side-by-side: LivePortrait vs improved swap
We rebuilt the swap with better masking, LAB color matching, and an enhancer (`--preset best`,
`--mask parse`, `--color-match lab`, `--enhancer`), then compared it to LivePortrait
(KwaiVGI / KlingTeam, human mode) on the same Avatar1 still and Oscar frame.

| Method | ArcFace cosine vs Avatar1 | Laptop CPU speed |
|---|---|---|
| LivePortrait | **0.84** | About 20 s/frame |
| Improved face swap | 0.37 | About 4 s/frame (DirectML) |

LivePortrait won on identity. Expression-only (`--animation_region exp`) kept face tattoos
crisp. Absolute pose (`--no_flag_relative_motion`) smeared the tattoos. Relative mode with a
still driver behaved like expression-only. LivePortrait became the face path; the swap stayed
as a fallback.

## Phase 2 — Runpod Secure Cloud (Oct 7)

### Auth
Runpod MCP OAuth failed in the app (`redirect_uri is not allowed`). Work went through an API
key and `curl` instead. The key is never printed or committed.

### First pod
- Secure Cloud only, US-IL-1
- Official ComfyUI template `cw3nka7d08` (image `runpod/comfyui:*-cuda12.8`)
- 1× RTX 4090 at about **$0.74/hr**
- Ports: 8188 ComfyUI, 8080 HTTP, 8888 Jupyter, 22 SSH
- 100 GB container disk, **no network volume** (everything is lost on terminate)
- `terminateAfter` set about 4 hours out, plus an in-pod watchdog as backup

### Bootstrap (`runpod/bootstrap_avatar_pod.sh`)
Clones LivePortrait into `/workspace/LivePortrait` with its own Python 3.10 venv (torch 2.4.1
cu121, onnxruntime-gpu 1.19.2), downloads human-mode weights, and creates
`/workspace/avatar1/{source,driving,out}`. `albucore` is pinned to **0.0.13** because newer
versions compile for 15+ minutes on the pod.

### Still-image validation
`/workspace/run_lp_still.sh` confirmed expression-only keeps tattoos sharp and absolute pose
smears them. That locked the live default to `region=exp`.

## Phase 3 — Live webcam → Meet (Oct 8)

The scope doc planned Tailscale + SRT. We shipped something simpler that worked the same night:
an **SSH tunnel** from the laptop to the pod, carrying JPEG frames over WebSocket.

```
Laptop webcam
  → capture_webcam.py (OpenCV, 640×480 JPEG)
  → SSH tunnel localhost:8765 → pod :8080
  → pod_live_server.py (LivePortrait kept warm, CUDA graphs)
  → JPEG back
  → "Avatar1 live" preview + pyvirtualcam OBS Virtual Camera
  → optional Browser source at http://127.0.0.1:8766/ (960×540)
  → Google Meet / Zoom camera = "OBS Virtual Camera"
```

### Performance (RTX 4090, US-IL-1)

| Path | FPS | Glass-to-glass latency |
|---|---|---|
| SSH tunnel (preferred) | About **20 fps** | About **140–160 ms** |
| Runpod HTTPS proxy fallback | About 10–13 fps | About 200 ms |

Pod render itself was about 32 ms GPU plus 10–35 ms CPU. Without pinning OpenMP/MKL/torch
thread pools to **6** (the pod advertises 128 CPUs but its cgroup quota is much smaller) and
without **CUDA graphs**, each frame took about **400 ms**.

Tested working in **Google Meet** with the camera set to OBS Virtual Camera.

### Face picks
Oscar ranked the clean photos. Keys were remapped to:

| Key | Photo | Look |
|---|---|---|
| `1` (default) | `clean_03` | Slicked silver ponytail, favorite |
| `2` | `clean_06` | Smirk bun |
| `3` | `clean_01` | Silver bun |
| `4` | `clean_05` | Parted-lips bun |
| `5` | `necktattoo_front` | Neck-tattoo avatar |

Rejected: `clean_02` (platinum) and `clean_04` (tilted long hair). Lesson: **source photos must
be head-on, front-facing, neutral, and sharp**. Press `c` after every switch.

## Phase 4 — Body LoRA training (Oct 8)

Dataset from `Avatar1\body\` (Main, Hourglass, Testing, Multiple Outfits, plus tattoo look
from the earlier face set). Video frames sampled about every 1–1.5 s and deduped. Final set:
**145 images**.

| Setting | Value |
|---|---|
| GPU | Secure Cloud A40 |
| Time / cost | About **30 min**, about **$0.29** |
| Base | Realistic Vision V6.0 B1 (SD1.5) + `sd-vae-ft-mse` |
| Trainer | kohya sd-scripts **v0.9.1** |
| LoRA | Rank 32, alpha 16 |
| Resolution | 640 px with buckets 320–1024 |
| Optimizer | AdamW8bit, UNet 1e-4, TE 5e-5, cosine, 100 warmup |
| Schedule | Batch 2 × 10 epochs = **2020 steps** |
| Checkpoints | Every 2 epochs; **epoch 10** main, **epoch 6** backup |

Trigger: `avtr1 woman`. Style tokens: `av1main`, `av1hourglass`, `av1testing`, `av1outfits`,
`av1tattoo`. Tattoos are a tag switch (`face tattoos, neck tattoos`). Negative prompt includes
`nude, topless, nipples`. Dataset rule: only clothed Avatar1 images; explicit or off-identity
images are excluded.

Oscar's focus for the next live session: **Main** and **Hourglass**, standing farther back for
full body + face.

### Body realtime status (not live yet)

ControlNet OpenPose + LoRA on the A40:

| Mode | Seconds / frame | Rough FPS |
|---|---|---|
| 25-step DPM | 1.86 | About 0.5 |
| LCM 4 steps | 0.67 | About 1.5 |
| Fused LoRA + LCM, no CFG, 4 steps | 0.38 | About 2.6 |
| Same, 2 steps | 0.29 | About 3.4 |

Target for live is **10+ fps**. Next: RTX 4090 + StreamDiffusion / TensorRT-style pipeline.

## What we spent

Runpod balance went from about **$10 → $7.41**. Face live sessions and body training together
were the bulk of that. Secure Cloud + auto-terminate kept it from running away.

## What we intentionally did not ship

- Tailscale mesh and SRT return stream (SSH tunnel replaced them for the face path)
- Network volume (ephemeral pods for privacy; bootstrap time is the trade-off)
- Body live at call quality (next session)
- Voice changer and mobile (concepted in [VOICE](VOICE.md) and [MOBILE](MOBILE.md))
