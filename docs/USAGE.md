# Usage

How to run Avatar1 live and how to train the body styles. Paths assume the laptop project
lives at `C:\Users\oscar\OneDrive\Documents\kimi\Workspaces\FaceBodySwapStream\` and the pod
uses `/workspace/`.

Contents:
1. [Start a pod](#1-start-a-pod)
2. [Start the live server](#2-start-the-live-server)
3. [Laptop one-time setup](#3-laptop-one-time-setup)
4. [Go live](#4-go-live)
5. [Switch faces](#5-switch-faces)
6. [Switch bodies (planned)](#6-switch-bodies-planned)
7. [Restore the face photos on a new pod](#7-restore-the-face-photos-on-a-new-pod)
8. [Use it in Google Meet, Zoom, and Teams](#8-use-it-in-google-meet-zoom-and-teams)
9. [Shut down](#9-shut-down)
10. [Train the body LoRA](#10-train-the-body-lora)

---

## 1. Start a pod

In the Runpod console (or API):

| Setting | Value |
|---|---|
| Cloud | **Secure Cloud** only |
| GPU | 1× RTX 4090 (24 GB), about $0.74/hr |
| Template | Official **ComfyUI – CUDA 12.8** (`cw3nka7d08`) |
| Ports | `8080/http` (live server), `22/tcp` (SSH). 8188/8888 are optional |
| Disk | 100 GB container disk, no network volume |
| SSH key | Add the laptop's public key (`%USERPROFILE%\.ssh\runpod_avatar_laptop.pub`) under Runpod **Settings → SSH Public Keys** before you deploy |
| Budget | Set auto-terminate (`terminateAfter`) about 2–4 hours out |

Then SSH in (Connect tab shows `ssh root@<pod-ip> -p <port>`) and run:

```bash
curl -fsSL https://raw.githubusercontent.com/pjrny/funnymemelivefacebodyswap/main/runpod/bootstrap_avatar_pod.sh | bash
# optional: add `-s -- --smoke` to run LivePortrait once on its example
```

That takes a few minutes and installs LivePortrait, its weights, and the repo at
`/workspace/funnymemelivefacebodyswap`. Logs: `/workspace/logs/bootstrap.log`.

**Optional watchdog** (backup to `terminateAfter`). Stops billing even if you forget:
```bash
nohup bash -c 'sleep 4h; runpodctl remove pod $RUNPOD_POD_ID' >/dev/null 2>&1 &
```
If `runpodctl` inside the pod is outdated, terminate from the console or the REST API from your own machine instead.

## 2. Start the live server

Upload the face photos first ([§7](#7-restore-the-face-photos-on-a-new-pod)). Then on the pod:

```bash
mkdir -p /workspace/live
cp /workspace/funnymemelivefacebodyswap/live/pod_live_server.py \
   /workspace/funnymemelivefacebodyswap/live/start_live_server.sh /workspace/live/
chmod +x /workspace/live/start_live_server.sh
/workspace/live/start_live_server.sh                   # default: clean_03 ponytail, expression only
tail -f /workspace/logs/live_server.log                # wait for "[live] ready" (about 70 s)
```

Options: `--source /workspace/avatar1/clean/clean_06.jpg`, `--region all`, `--out_w 960 --out_h 540`.

The script stops Runpod's FileBrowser (it normally sits on 8080). On first start the server
creates a random access token in `/workspace/live/token`. **Copy it with `scp`, not chat:**

```powershell
scp -P <port> -i $env:USERPROFILE\.ssh\runpod_avatar_laptop root@<pod-ip>:/workspace/live/token live\token.txt
```

If FastAPI/uvicorn are missing (they normally come with LivePortrait's requirements):
`cd /workspace/LivePortrait && uv pip install --python .venv/bin/python fastapi uvicorn`.

## 3. Laptop one-time setup

1. Install **OBS Studio** so the **OBS Virtual Camera** device exists. OBS doesn't need to run.
2. Create the Python environment in the project's `live\` folder:
   ```powershell
   py -3.13 -m venv live\.venv
   live\.venv\Scripts\pip install opencv-python numpy websocket-client pyvirtualcam
   ```
3. SSH key for the tunnel (once): `ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\runpod_avatar_laptop`.
   Add the `.pub` to Runpod (see §1).
4. Copy `live\live_config.example.json` to `live\live_config.json` and fill it in. **Never commit
   this file** (it's in `.gitignore`).

| Field | Value |
|---|---|
| `url` / `http` | `wss://<POD_ID>-8080.proxy.runpod.net` / `https://<POD_ID>-8080.proxy.runpod.net` (proxy fallback) |
| `token` | Contents of the token file from §2 |
| `cam` | Webcam index, usually `0` |
| `ssh_host` / `ssh_port` | Pod public IP and the SSH port from the Connect tab |

## 4. Go live

Double-click **`live\start_live.bat`**, or:

```powershell
powershell -ExecutionPolicy Bypass -File live\start_live.ps1               # webcam 0 via SSH tunnel
powershell -ExecutionPolicy Bypass -File live\start_live.ps1 -Cam 1        # another webcam
powershell -ExecutionPolicy Bypass -File live\start_live.ps1 -NoTunnel     # Runpod HTTPS proxy (slower)
powershell -ExecutionPolicy Bypass -File live\start_live.ps1 -Video clip.mp4   # test without a webcam
```

The launcher opens an SSH tunnel (`localhost:8765 → pod 127.0.0.1:8080`), falls back to the
proxy if the tunnel fails, and starts `capture_webcam.py`. An **"Avatar1 live"** window opens.

**Calibrate:** look straight at the camera with a neutral face and press **`c`**.

Expected: about 20 fps, about 150 ms delay via the tunnel; about 10–13 fps via the proxy.

## 5. Switch faces

Keys work while the "Avatar1 live" window has focus:

| Key | Look | Photo on pod |
|---|---|---|
| `1` | Slicked silver ponytail (favorite, server default) | `/workspace/avatar1/clean/clean_03.jpg` |
| `2` | Smirk bun | `clean/clean_06.jpg` |
| `3` | Silver bun | `clean/clean_01.jpg` |
| `4` | Parted-lips bun | `clean/clean_05.jpg` |
| `5` | Neck-tattoo avatar | `/workspace/avatar1/necktattoo_front.jpg` |
| `c` | Calibrate. **Press after every switch** | |
| `e` | Expression only (default, sharpest) | |
| `a` | Expression + head pose (more alive, tattoos smear) | |
| `+` / `-` | Motion strength up/down (0.3–2.0) | |
| `q` / Esc | Quit (closes the tunnel too) | |

**Add a new face:** upload a head-on, neutral, sharp photo to `/workspace/avatar1/clean/`, then
change the `sources` map in `live/capture_webcam.py` (search for `ord("1")`).

## 6. Switch bodies (full-body live)

Body live works (first version, Oct 8 2026). Full setup is in [`live/body/README.md`](../live/body/README.md).
Your laptop reads your pose with RTMPose on the CPU and sends only skeleton keypoints to the pod. The pod renders
Avatar1 in that pose with ControlNet OpenPose, the body LoRA, and LCM in 2 steps. That gives about 7–8 fps
and about 230 ms latency on an RTX 4090. Turning around switches to a back view. Every frame passes an NSFW check.

Start it with `live\body\start_body.bat` once the pod is up. Then focus the "Avatar1 body" window:

| Key | Style token | Look |
|---|---|---|
| `1` | `av1main` | White crop tank, long silver hair (default) |
| `2` | `av1hourglass` | Hourglass figure, always in an opaque black top and trousers |
| `3` | `av1outfits` | Tees and shirts, slicked ponytail |
| `4` | `av1testing` | Slim, white linen shirt, bun |
| `t` | add `face tattoos, neck tattoos` | Tattoos on any style |
| `[` / `]` | fewer or more steps | Faster or sharper |
| `s` | new seed | New variation |
| `f` | framing | Auto, full body, or waist-up |
| `q` | quit | |

Stand back far enough that your feet are in the shot for full body. All styles are in one LoRA, so switching is
just a prompt change with no reload.

## 7. Restore the face photos on a new pod

Pods have no volume, so the photos vanish on terminate. Keep the backup locally (the six clean
photos + `necktattoo_front.jpg`) and upload after bootstrap:

```powershell
ssh -p <port> -i $env:USERPROFILE\.ssh\runpod_avatar_laptop root@<pod-ip> "mkdir -p /workspace/avatar1/clean"
scp -P <port> -i $env:USERPROFILE\.ssh\runpod_avatar_laptop "<backup>\clean\clean_0*.jpg" root@<pod-ip>:/workspace/avatar1/clean/
scp -P <port> -i $env:USERPROFILE\.ssh\runpod_avatar_laptop "<backup>\necktattoo_front.jpg" root@<pod-ip>:/workspace/avatar1/
```

Large uploads: Runpod's transfer tools limit single files to about 100 MB, so split big zips.

> **TODO:** extend `runpod/bootstrap_avatar_pod.sh` to create `/workspace/avatar1/clean/` and copy
> `/workspace/live/` automatically, so steps 2 and 7 become one command.

## 8. Use it in Google Meet, Zoom, and Teams

Keep `start_live` running, then pick the camera in the meeting app:

| App | Where |
|---|---|
| **Google Meet** (browser) | ⋮ → Settings → Video → Camera → **OBS Virtual Camera**. Tested ✅ |
| **Zoom** desktop | Settings → Video → Camera → **OBS Virtual Camera** |
| **Microsoft Teams** desktop | Settings → Devices → Camera → **OBS Virtual Camera** |

Important:
- **Don't press "Start Virtual Camera" in OBS** while the script feeds it. Only one app can drive it.
- Turn off the meeting app's own effects (background blur, touch-up) for the cleanest face.
- Mirror view: Meet/Zoom mirror *your* preview only. Others see it unmirrored.

**OBS scene alternative** (overlays, captions, layout): in OBS add a **Browser** source with URL
`http://127.0.0.1:8766/` at **960×540**. Then run the client with `-- --no-vcam` so OBS can own the
virtual camera, and click "Start Virtual Camera" in OBS.

**Watch from any browser:** `https://<POD_ID>-8080.proxy.runpod.net/view?k=<token>`. Don't share
that link: the token is the only lock.

## 9. Shut down

1. Press `q` in the "Avatar1 live" window (closes the tunnel).
2. Copy anything you want to keep off the pod (`scp`).
3. **Terminate the pod** in the Runpod console (Stop still bills for disk; Terminate ends billing).
4. Check the console shows **no running pods**.

## 10. Train the body LoRA

### Dataset folders (on your machine, never in git)

```
Avatar1\body\
  Main\            → av1main       (focus)
  Hourglass\       → av1hourglass  (focus)
  Testing\         → av1testing
  Multiple Oufits\ → av1outfits
  (tattoo look)    → av1tattoo     (from the face/neck-tattoo stills)
```

Rules: same face and silver hair, clothed images only, plain backgrounds help, include
waist-up *and* full-body shots. Short videos are great: frames are pulled every 1–1.5 s and deduped.

### Captions and tokens

Every caption starts with `avtr1 woman, <style token>`, followed by tattoos tag, outfit, hair,
framing, and view. Example:

```
avtr1 woman, av1main, no tattoos, white cropped tank top, light grey leggings, long hair down, waist-up, front view, plain grey studio background
```

`train/build_dataset.py` writes the kohya layout `dataset/<repeats>_<style>/*.jpg + .txt`.
Repeats used: main 2, hourglass 3, testing 6, outfits 2, tattoo 3, face refs 3.

### Run

1. Runpod **Secure Cloud** pod, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
   1× 24–48 GB GPU (A40 worked well), port 22, auto-terminate set.
2. `bash train/bootstrap_train_pod.sh` (kohya sd-scripts v0.9.1, Realistic Vision V6.0 B1, VAE, ControlNet OpenPose, LCM-LoRA).
3. `python train/build_dataset.py` locally, then `scp -r dataset root@<pod-ip>:/workspace/`.
4. `bash train/train_lora.sh` → `/workspace/lora_out/avtr1_body_rv6*.safetensors` (about 30 min on an A40).
5. `python train/render_tests.py` (style grid, poses × clean/tattoo, epoch comparison, LCM timing) and `python train/bench_lcm.py`.
6. Copy the LoRA and test renders back with `scp`. **Terminate the pod.**

### Use the LoRA

| Item | Value |
|---|---|
| Main file | `avtr1_body_rv6.safetensors` (epoch 10) |
| Backup | `avtr1_body_rv6-000006.safetensors` (epoch 6, a bit less overfit) |
| Weight | 0.8–1.0. With LCM-LoRA: 4 steps, guidance 1.0–1.5 |
| Negative | `lowres, blurry, deformed, bad anatomy, nude, topless, nipples` |

Where files go: keep the LoRA in a local, **non-synced** folder (move it out of OneDrive), ideally
inside a 7-Zip AES-256 archive, and upload it to the pod per session.
