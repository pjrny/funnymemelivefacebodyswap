# Optimization roadmap

Ways to cut cost, raise FPS, and improve accuracy for Avatar1. Numbers marked **estimate**
are reasoned guesses; measured numbers come from this project's Oct 2026 runs.

## Cost

### Spot vs on-demand (still Secure Cloud)

Runpod Secure Cloud offers interruptible ("spot") capacity that is cheaper than on-demand,
with a short warning before reclaim. **Estimate:** often 30–60% less than the on-demand rate
for the same GPU; check the console for live prices.

| Use | Recommendation |
|---|---|
| Live calls | Prefer **on-demand**. A mid-call reclaim kills the meeting |
| Training / offline renders | **Spot** is fine. Training can resume from the last checkpoint |
| Always | Stay on **Secure Cloud**. Community / Vast.ai stay off-limits for privacy |

### Cheaper GPUs vs the 4090

Measured: RTX 4090 live face about $0.74/hr; A40 training about $0.29 for 30 min.

| GPU | Good for | Trade-off |
|---|---|---|
| RTX 4090 (24 GB) | Live face today; body live target | Best performance, highest $/hr |
| RTX 3090 (24 GB) | Face live; body LCM at lower fps | **Estimate:** about $0.50/hr Secure Cloud; a bit slower than 4090 |
| A40 (48 GB) | Training, offline grids | Great for LoRA; slower than 4090 for realtime |
| L4 (24 GB) | Cost-sensitive face live | **Estimate:** slower; may need lower resolution / lighter pipeline |

Rule of thumb: train on A40 / 3090 spot; stream on 4090 on-demand while you care about fps.

### Network volume vs bootstrap time

Today: no volume → bootstrap each time (LivePortrait install + weight download, several
minutes) → everything wiped on terminate.

| Option | Cost | Privacy | Speed |
|---|---|---|---|
| Ephemeral (current) | $0 storage | Best: nothing persists | Pay the bootstrap minutes every session |
| Network volume | **About $0.07/GB/month** (first 1 TB Secure Cloud; check current pricing) | Worse: weights and, if you leave them there, LoRAs sit in the cloud | Skip most of bootstrap; start live in about 1–2 min **(estimate)** |

Middle ground that keeps privacy: a volume that only stores **public** assets (LivePortrait
weights, base SD models, ComfyUI). Keep the Avatar1 LoRA and face photos on the laptop and
upload them per session. That cuts startup without parking your likeness in the cloud.

### Prebuilt Docker image

Package LivePortrait + deps + weights into a private container (or a Runpod template that
points at one). First pull is slow; every later start skips `bootstrap_avatar_pod.sh`.
**Estimate:** cold start from about 5–10 min of bootstrap down to about 1–2 min. Combine with
a public-only network volume for the best of both.

## Speed (face)

Already done and measured:

- Thread pools pinned to 6 (was about 400 ms/frame without it)
- CUDA graphs for the LivePortrait networks
- SSH tunnel instead of the HTTPS proxy (about 20 fps vs about 10–13 fps)

Next levers:

| Idea | Expected gain | Effort |
|---|---|---|
| **FasterLivePortrait / TensorRT** | **Estimate:** 1.5–3× GPU speedup after conversion cost | Medium: convert ONNX → TensorRT once per GPU type |
| **GPU-side face tracking** | Cut the CPU landmark cost (10–35 ms today every 2nd frame) | Medium |
| Lower out resolution (e.g. 720×405) | Small fps bump, softer image | Easy |
| Drop client target fps to 15 | Less upload, still Meet-friendly | Easy (`--fps 15`) |
| Pod with a stronger CPU | Prep/encode stages are CPU-bound | Easy: pick a different Secure Cloud host |

Server-side ceiling today is about 18–20 fps. Beating that needs TensorRT or a leaner tracker.

## Speed (body, the big one)

Measured on A40, ControlNet OpenPose + LoRA, 512×512:

| Mode | s/frame | Rough fps |
|---|---|---|
| 25-step DPM | 1.86 | About 0.5 |
| LCM 4 steps | 0.67 | About 1.5 |
| Fused LoRA + LCM, CFG 1.0, 4 steps | 0.38 | About 2.6 |
| Same, 2 steps | 0.29 | About 3.4 |

Target for live: **≥ 10 fps**. Roadmap:

1. **Move to a 4090** (raw speedup over A40). **Estimate:** about 1.5–2× on the same pipeline.
2. **StreamDiffusion-style** or **TensorRT** UNet + ControlNet, with a tiny VAE
   (`AutoencoderTiny`). Fused LCM already helped; TRT usually helps more.
3. **SD-Turbo / LCM at 1–2 steps** with carefully tuned denoising. Quality drop is real; test
   against `render_tests.py` grids before committing.
4. **Lower resolution** for live (448 or 384), upscale lightly on the laptop if needed.
5. **Frame skipping / motion interpolation:** generate every other frame, blend in between.
   Halves GPU load; motion gets softer.
6. **Pose on the laptop, skeleton only to the pod** (original scope). Cuts upload size and
   privacy exposure: your raw face never leaves the machine for the body path.

## Combining face + body

Best of both once body is live enough:

```
webcam → OpenPose (body pose)
       → SD1.5 + ControlNet + body LoRA   → body frame
       → LivePortrait (face region only)  → paste Avatar1 face onto the body frame
       → OBS Virtual Camera
```

LivePortrait already handles expression well. Body diffusion handles outfit and pose. Run face
at full rate and body at a lower rate if needed, and reuse the last body frame between body ticks.

## Accuracy

| Lever | What to do |
|---|---|
| Better source photos | Head-on, neutral, sharp; reject anything tilted (see [BEST_PRACTICES](BEST_PRACTICES.md)) |
| ArcFace checks | Keep comparing rendered faces to the Avatar1 references (we used 0.84 vs 0.37) |
| More full-body data | Stand back, shoot Main + Hourglass in matching outfits, retrain |
| Caption quality | Keep `avtr1 woman, <style>` as the first two tokens; shuffle the rest |
| Epoch choice | Prefer epoch 6 if epoch 10 looks burnt |
| Base model trade-off | SDXL-Turbo LoRA can look sharper but costs more VRAM and may miss the current LCM path; treat as an experiment, not a free upgrade |
| Tattoo consistency | Keep tattoos as a caption tag, not a separate identity; regenerate clean tattooed *Avatar1* photos if you want a stronger tattoo body style |

## Measuring

Track these every change:

| Metric | How |
|---|---|
| Pod render ms | `GET /stats` on the live server |
| End-to-end fps / latency | Printed in the "Avatar1 live" window |
| Body s/frame | `train/bench_lcm.py`, `train/render_tests.py` → `timings*.json` |
| Identity | ArcFace cosine vs Avatar1 refs on a fixed set of frames |
| Cost | Runpod balance before/after; always terminate |

A simple log line per session (GPU, region, fps, latency, $ spent) makes the next optimization
decision much easier.
