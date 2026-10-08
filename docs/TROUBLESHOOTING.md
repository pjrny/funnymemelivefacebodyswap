# Troubleshooting

Symptom → cause → fix for every issue we hit while building Avatar1.

---

## Laptop / webcam

### "No webcam found" and a command prompt that loops forever
- **Symptom:** The PowerShell window prints the same message over and over. Looks like a
  runaway script.
- **Cause:** The laptop's physical camera privacy switch (or Fn key) is **off**. Windows reports
  the device as present but won't open it. Device Manager can also show **Present = False**.
- **Fix:** Flip the privacy switch / Fn camera key on, confirm the LED is on, then restart
  `start_live.bat`. If Device Manager still shows Present=False, disable then re-enable the
  camera device, or reboot.

### PowerShell positional-arg bug: `-Cam` / `-SnapDir` ignored or misparsed
- **Symptom:** Extra args after `start_live.ps1` go to the wrong place, or `Cam` isn't applied.
- **Cause:** Older versions of the launcher used positional remaining arguments that PowerShell
  mangled.
- **Fix:** Use the current `start_live.ps1`, which takes explicit `-Cam`, `-SnapDir`, `-Video`,
  `-NoTunnel`. Example: `start_live.ps1 -Cam 1 -SnapDir C:\Temp\snaps`. Extra capture flags go
  after `--`: `start_live.ps1 -- --fps 15 --no-vcam`.

### OpenCV floods the console with camera-probe warnings
- **Symptom:** Pages of `CAP_MSMF` / DirectShow noise while the script hunts for a camera.
- **Cause:** OpenCV's default logging is chatty on Windows.
- **Fix:** Already handled: `capture_webcam.py` sets `OPENCV_LOG_LEVEL=ERROR` at startup. If you
  still see noise, set it in PowerShell before launching: `$env:OPENCV_LOG_LEVEL="ERROR"`.

### Webcam opens but the preview is black / frozen
- **Cause:** Another app (Meet, Zoom, Teams, Camera app) already holds the device exclusive.
- **Fix:** Close those apps, unplug/replug USB webcams, then restart `start_live`. Prefer the
  built-in camera index `0` first.

---

## SSH tunnel / network

### Tunnel fails; launcher falls back to the Runpod HTTPS proxy
- **Symptom:** Console says `SSH tunnel failed; using Runpod proxy`. FPS drops to about 10–13 and
  latency rises to about 200 ms.
- **Cause:** Wrong `ssh_host`/`ssh_port`, key not on the pod, or the pod is still booting.
- **Fix:**
  1. Confirm the Connect tab IP and port, update `live_config.json`.
  2. Confirm the public key is in Runpod **Settings → SSH Public Keys** (not only on the pod).
  3. Test: `ssh -i $env:USERPROFILE\.ssh\runpod_avatar_laptop -p <port> root@<pod-ip> echo ok`.
  4. Wait until the pod shows Ready before starting the client.

### OBS: "do I need a Tailscale IP?"
- **Symptom:** Confusion about what URL to put in OBS.
- **Cause:** The original scope doc planned Tailscale + SRT. We never set that up.
- **Fix:** Use the **SSH tunnel**. OBS Browser source is `http://127.0.0.1:8766/`. Meet/Zoom use
  the camera **OBS Virtual Camera**. No Tailscale needed for the face path.

### Connection drops mid-call
- **Cause:** Wi-Fi hiccup, or the pod's auto-terminate timer fired.
- **Fix:** Prefer wired Ethernet or 5 GHz Wi-Fi close to the AP. Check the Runpod console: if the
  pod is gone, you hit the timer. Restart a new pod and re-upload the face photos.

---

## Pod / live server

### Port 8080 already in use / `/health` returns FileBrowser
- **Symptom:** Live server won't bind, or the proxy URL shows a file browser UI.
- **Cause:** Runpod's FileBrowser occupies 8080 on the ComfyUI template.
- **Fix:** `start_live_server.sh` already runs `pkill -x filebrowser`. If you started the server
  by hand, kill FileBrowser yourself, then start the server. To bring FileBrowser back later:
  `cd /workspace/runpod-slim && nohup filebrowser >/dev/null 2>&1 &`.

### Frames take about 400 ms each (about 2.5 fps)
- **Cause:** The pod advertises 128 CPUs but the cgroup quota is much smaller. Torch/OpenMP
  spawn hundreds of threads and thrash.
- **Fix:** Already in the code: `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=6`,
  `torch.set_num_threads(6)`, CUDA graphs. Confirm `start_live_server.sh` exports those before
  launch. Check `/stats` for timings.

### `runpodctl` inside the pod is outdated / broken
- **Symptom:** `runpodctl pod stop` fails or behaves oddly.
- **Fix:** Don't rely on the in-pod CLI. Terminate from the Runpod web console, or call the REST
  API from your own machine with your API key. Prefer the console for one-shot terminate.

### Upload fails around 100 MB
- **Symptom:** Transfer tool rejects a big zip (e.g. a 177 MB archive).
- **Cause:** Runpod's file transfer tools cap a single file around 100 MB.
- **Fix:** Split the archive (`7z a -v90m archive.7z …`), upload the parts, reassemble on the pod
  (`7z x archive.7z.001`), or use `scp` which has no such limit.

### Everything disappeared after terminate
- **Cause:** No network volume. Container disk is wiped on stop/terminate.
- **Fix:** Expected. Keep a local backup of `clean/` and `necktattoo_front.jpg`, and re-upload
  after every bootstrap. See [USAGE §7](USAGE.md#7-restore-the-face-photos-on-a-new-pod).

### Server says "No face in source"
- **Cause:** The photo has no detectable face, is too small, or is the wrong path.
- **Fix:** Use a sharp, front-facing, head-and-shoulders photo. Confirm the path under
  `/workspace/avatar1/`. Try `/workspace/run_lp_still.sh` on it first.

### Auth 401 / "bad token"
- **Cause:** `live_config.json` token doesn't match `/workspace/live/token`, or the token file
  was regenerated by a restart after being deleted.
- **Fix:** Re-copy the token from the pod into `live_config.json`. Don't paste it into chat.

---

## OBS / Meet / Zoom

### Meet shows your real face, not Avatar1
- **Cause:** Camera still set to the laptop webcam.
- **Fix:** Meet ⋮ → Settings → Video → Camera → **OBS Virtual Camera**. Leave Meet, rejoin if
  the list doesn't refresh.

### Virtual Camera is blank or stuck on a still
- **Cause:** Both OBS and `capture_webcam.py` tried to own the virtual camera, or the client
  crashed.
- **Fix:** Quit the client (`q`), make sure OBS is **not** running "Start Virtual Camera", then
  restart `start_live.bat`. If you want OBS to own it (Browser source workflow), launch with
  `-- --no-vcam` and start the virtual camera from OBS.

### OBS Browser source is black
- **Cause:** Wrong URL, or the client isn't running.
- **Fix:** Use `http://127.0.0.1:8766/` (an HTML page wrapping the stream). As a Media/other
  source you can also use `http://127.0.0.1:8766/live.mjpg` or `/live.jpg`. Confirm the
  "Avatar1 live" window is open. Width × height in OBS: 960 × 540.

### Tattooed faces look smeared when you turn your head
- **Cause:** Region is set to `all` (expression + pose). LivePortrait warps the tattoos with the
  head.
- **Fix:** Press `e` for expression only. Use `a` only with clean (no-tattoo) faces, or accept
  the smear for more lively motion.

---

## Training

### kohya / pip install blows up on torch or diffusers versions
- **Cause:** Newer `sd-scripts` HEAD pulls torch/diffusers that the pod image can't satisfy.
- **Fix:** Stick to the pinned tag: `git checkout v0.9.1` as `bootstrap_train_pod.sh` does.

### LoRA looks overfit / burnt (artifacts, identical faces)
- **Fix:** Prefer the epoch-6 backup (`avtr1_body_rv6-000006.safetensors`) over epoch 10, or
  drop LoRA weight to 0.7–0.8. Add more varied full-body shots and retrain.

### Style tokens don't separate cleanly
- **Cause:** Too few images in one folder, or captions missing the style token.
- **Fix:** Check each image has a matching `.txt` starting with `avtr1 woman, <style>`. Increase
  repeats for the weak style and retrain.
