
---

# Full body (Avatar1 body LoRA, live) — `*_body*` files

```
laptop webcam ─▶ pose on the LAPTOP CPU (rtmlib RTMPose, ~12 fps) ─▶ 18 skeleton keypoints only ─WebSocket─▶
Runpod RTX 4090: pod_body_server.py  (SD1.5 Realistic Vision V6 + Avatar1 body LoRA + LCM-LoRA fused,
                 ControlNet OpenPose, 2 LCM steps, CFG 1.5 with nudity negatives, NSFW classifier gate)
      ▲                                                     │
      └── window "Avatar1 body" + OBS Virtual Camera ◀──────┘ 960×540 JPEG (512×768 portrait render, auto-framed)
          + http://127.0.0.1:8766/ (OBS Browser Source)
```
No camera pixels leave the laptop: only the skeleton is sent.

| file | where | what |
|---|---|---|
| `bootstrap_body_pod.sh` | pod `/workspace/live/` | installs diffusers 0.31 + FastAPI on `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, downloads public models (RV6 B1, sd-vae-ft-mse, ControlNet OpenPose, LCM-LoRA, TAESD, NSFW classifier). |
| `pod_body_server.py` | pod | FastAPI on :8080. WebSocket `/ws?k=<token>`; `GET /stats`, `/live.jpg`, `/pose.jpg`. Keeps the pipeline warm. |
| `start_body_server.sh` | pod | (re)starts the server; log `/workspace/logs/body_server.log`. Extra args pass through (`--steps 3`, `--w 448 --h 672`, `--guidance 1.0`). |
| `capture_body.py` | laptop `FaceBodySwapStream\live\` | webcam → pose (CPU, capped at 3 threads / 12 fps so the laptop stays usable) → pod; preview, OBS Virtual Camera, MJPEG :8766 |
| `start_body.ps1` / `start_body.bat` | laptop | stops any face/body session, opens the SSH tunnel (8765 → pod 8080, falls back to the HTTPS proxy), runs the client |
| `live_config_body.example.json` | laptop | copy to `live_config_body.json` (url, token, ssh_host, ssh_port). **Never commit the real file.** |

**Pod setup:** scp `bootstrap_body_pod.sh`, `pod_body_server.py`, `start_body_server.sh` to `/workspace/live/`, run the
bootstrap (~4 min), scp the private LoRA to `/workspace/lora/avtr1_body_rv6.safetensors` (never via git/cloud), then
`bash /workspace/live/start_body_server.sh` (ready in ~25 s). Add the laptop key (`runpod_avatar_laptop.pub`) to
`~/.ssh/authorized_keys` on the pod.

**Laptop setup (one time):** `live\.venv\Scripts\pip install rtmlib onnxruntime` (on top of the face-live packages).
Pose models (~40 MB) download to `%USERPROFILE%\.cache\rtmlib` on first run.

**Keys in "Avatar1 body":** `1` Main (default) · `2` Hourglass · `3` Outfits · `4` Testing · `t` tattoos on/off ·
`[` / `]` fewer/more LCM steps (speed vs quality) · `s` new seed (new look) · `f` auto-frame ↔ full camera frame · `q` quit.

**Turning around:** when the face keypoints disappear but both shoulders are visible the prompt switches to
"back view, from behind". Short pose dropouts (<1 s) reuse the last skeleton; with nobody in frame the last image is held.

**Safety:** every frame goes through an NSFW image classifier on the GPU (Falconsai/nsfw_image_detection,
~6 ms); frames scoring > 0.3 are never sent (the last safe frame is held). CFG 1.5 is on so the negative prompt
(`nude, naked, topless, nipples, cleavage, skin-colored clothing, lingerie`) actually applies. The Hourglass style is
forced into an opaque black top and trousers because the LoRA learned it with a skin-tone bodysuit, which read as
nude at 2 steps in the first live test. The stock SD safety checker was dropped: ~80% false positives on tank tops.

**Performance (RTX 4090 Secure Cloud US, Oct 8 2026):** 512×768, 2 LCM steps, CFG 1.5, TAESD decode:
~105–115 ms per frame on the pod; end to end ~7–8 fps, ~220–270 ms latency through the SSH tunnel.
448×672 is barely faster (~100 ms), so the render is launch/overhead-bound, not pixel-bound.
Laptop pose (rtmlib lightweight, 3 ORT threads) ~12 fps. Running ORT with all cores pegged the i7-1255U at 100%
and starved the network (1.5 fps, 1.5 s latency), hence the thread cap.
