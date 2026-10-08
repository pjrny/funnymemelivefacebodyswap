# live/ — Avatar1 realtime webcam driver (LivePortrait on a Runpod GPU)

```
laptop webcam ──JPEG/WebSocket──▶ Runpod RTX 4090: pod_live_server.py (LivePortrait, kept warm, CUDA graphs)
      ▲                                              │
      └── preview window + OBS Virtual Camera ◀──────┘ rendered Avatar1 frame (JPEG)
          + http://127.0.0.1:8766/ (OBS Browser Source)
```

Avatar1 is animated with **your expressions only** by default (`region=exp`): eyes, brows, mouth,
blinks. Her head stays still, which keeps the face tattoos sharp. Press `a` to also follow head
pose (the tattoos smear more).

## Files
| file | where | what |
|---|---|---|
| `pod_live_server.py` | pod `/workspace/live/` | FastAPI server on :8080 (Runpod HTTP proxy port). WebSocket `/ws`, `POST /frame`, `GET /live.jpg`, `/live.mjpg`, `/view`, `/stats`, `POST /control`. Every endpoint needs `?k=<token>`. |
| `start_live_server.sh` | pod | (re)starts the server; log in `/workspace/logs/live_server.log`. Stops Runpod FileBrowser, which normally sits on 8080. |
| `capture_webcam.py` | laptop `FaceBodySwapStream\live\` | webcam capture, upload, preview window, OBS Virtual Camera (pyvirtualcam), local MJPEG on :8766 |
| `start_live.ps1` / `start_live.bat` | laptop | one-click launcher: opens an SSH tunnel to the pod (falls back to the HTTPS proxy), then runs the client |
| `live_config.example.json` | laptop | copy to `live_config.json` and fill in the pod URL, token, and SSH host/port. **Never commit the real file.** |

## Run it
Pod (SSH in, or use the Jupyter terminal):
```bash
/workspace/live/start_live_server.sh                      # default: clean/clean_03.jpg (ponytail), expression-only
/workspace/live/start_live_server.sh --source /workspace/avatar1/clean/clean_06.jpg --region all
tail -f /workspace/logs/live_server.log                   # wait for "[live] ready" (~70 s)
```
Laptop: double-click `live\start_live.bat`, or run
```powershell
powershell -ExecutionPolicy Bypass -File live\start_live.ps1             # webcam 0, SSH tunnel
powershell -ExecutionPolicy Bypass -File live\start_live.ps1 -NoTunnel   # via the Runpod HTTPS proxy
powershell -ExecutionPolicy Bypass -File live\start_live.ps1 -Video clip.mp4   # test without a webcam
```
Laptop setup (one time): `py -3.13 -m venv live\.venv` and
`live\.venv\Scripts\pip install opencv-python numpy websocket-client pyvirtualcam`. OBS Studio must be
installed so the **OBS Virtual Camera** device exists. OBS does not need to be running.

Preview window keys: `c` calibrate (hold a neutral face and press c) · `e` expression only · `a` expression plus head pose ·
`1` ponytail (clean_03) · `2` smirk bun (clean_06) · `3` silver bun (clean_01) · `4` parted-lips bun (clean_05) · `5` neck-tattoo avatar · `+/-` motion strength · `q` quit.

## Using it in a call / OBS
* **Zoom/Meet/Teams:** pick the camera **"OBS Virtual Camera"** while `start_live` is running.
  (Don't press "Start Virtual Camera" inside OBS at the same time. Only one app can feed it.)
* **OBS scene:** add a *Browser* source with URL `http://127.0.0.1:8766/` at 960×540. Don't also feed
  the virtual camera from OBS while the script is doing it.
* Any browser can watch the pod stream directly at `https://<POD_ID>-8080.proxy.runpod.net/view?k=<token>`.

## Performance (RTX 4090 Secure Cloud, US-IL-1, Oct 2026)
* Pod render, with a 640×480 frame in and a 960×540 JPEG out: about 32 ms GPU (warp+decode+paste+nvjpeg) plus 10–35 ms CPU
  (decode, face landmarks every 2nd frame). The pod CPU is slow and quota-limited, so the server pins thread
  pools to 6 threads and replays the LivePortrait networks as **CUDA graphs**. Without that, each frame took ~400 ms.
* End to end (laptop → SSH tunnel → pod → laptop): **~19 fps, ~145 ms glass-to-glass excluding
  camera/display**. Through the Runpod HTTPS proxy it's ~10–13 fps and ~200 ms.
* Server-side ceiling is ~18–20 fps. Next steps to go faster: TensorRT (FasterLivePortrait), GPU-side
  face tracking, and a pod with a faster CPU.

Full walkthrough, key table, Meet/Zoom setup, and troubleshooting live in [docs/USAGE.md](../docs/USAGE.md) and [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md).
