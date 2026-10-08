#!/usr/bin/env python
"""
capture_webcam.py - laptop side of the Avatar1 live pipeline.

Webcam -> (JPEG over a WebSocket) -> Runpod GPU pod running pod_live_server.py (LivePortrait)
       -> rendered Avatar1 frames back -> preview window + OBS Virtual Camera (pyvirtualcam)
       + local MJPEG at http://127.0.0.1:8766/ for an OBS Browser Source.

Usage (Windows, from the FaceBodySwapStream folder):
    live\\.venv\\Scripts\\python.exe live\\capture_webcam.py
Options: --cam 0  --width 640 --height 480  --fps 20  --q 70  --no-vcam  --no-window
         --url wss://<podid>-8080.proxy.runpod.net/ws   (default: read from live\\live_config.json)
Keys in the preview window:
    c = calibrate (hold a neutral face, press c)    e = expression-only   a = expression + head pose
    1/2/3 = tattoo_front / clean_front / necktattoo_front     +/- = motion strength    q/Esc = quit
"""
import argparse, json, os, struct, sys, threading, time
import http.server, socketserver
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide noisy camera-probe warnings

import cv2
import numpy as np
import websocket  # websocket-client

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "live_config.json")


def load_cfg():
    cfg = {}
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH) as f:
            cfg = json.load(f)
    return cfg


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.cam_frame = None
        self.cam_seq = 0
        self.out_jpg = None
        self.out_img = None
        self.out_seq = 0
        self.sent = 0
        self.recv = 0
        self.lat_ms = None
        self.srv_ms = None
        self.fps = 0.0
        self.no_face = False
        self.msg = ""
        self.running = True
        self.ws = None
        self.recv_times = []
        self.cam_msg = ""


S = State()


# ---------------- webcam ----------------
def file_loop(args):
    """--video PATH: drive from a video file (looped) or a still image instead of the webcam."""
    still = cv2.imread(args.video) if args.video.lower().endswith((".jpg", ".jpeg", ".png")) else None
    cap = None if still is not None else cv2.VideoCapture(args.video)
    fps = (cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0) or 25
    print("driving from file", args.video, "fps", fps, flush=True)
    while S.running:
        t0 = time.time()
        if still is not None:
            fr = still
        else:
            ok, fr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        if fr.shape[1] > args.width:
            fr = cv2.resize(fr, (args.width, int(fr.shape[0] * args.width / fr.shape[1])))
        with S.lock:
            S.cam_frame = fr
            S.cam_seq += 1
        time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))


def open_cam(args):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.cam, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def cam_loop(args):
    if args.video:
        return file_loop(args)
    cap, warned = None, False
    while S.running:
        if cap is None:
            cap = open_cam(args)
            if cap is None:
                S.cam_msg = ("No webcam found. HP ENVY: press the camera privacy key (camera icon, top row) "
                             "or plug in the LG monitor camera. Retrying...")
                if not warned:
                    print(S.cam_msg, flush=True)
                    warned = True
                time.sleep(3)
                continue
            S.cam_msg = ""
            print("webcam", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), flush=True)
        ok, fr = cap.read()
        if not ok:
            S.cam_fail = getattr(S, "cam_fail", 0) + 1
            if S.cam_fail > 60:  # camera unplugged / privacy key pressed
                cap.release(); cap = None; S.cam_fail = 0
            time.sleep(0.02)
            continue
        S.cam_fail = 0
        with S.lock:
            S.cam_frame = fr
            S.cam_seq += 1
    if cap is not None:
        cap.release()


# ---------------- network ----------------
def send_loop(args):
    period = 1.0 / args.fps
    last_seq = -1
    fid = 0
    while S.running:
        t0 = time.time()
        ws = S.ws
        with S.lock:
            fr, seq = S.cam_frame, S.cam_seq
        if ws is not None and fr is not None and seq != last_seq and (S.sent - S.recv) < args.inflight:
            if fr.shape[1] != args.width:
                fr = cv2.resize(fr, (args.width, int(fr.shape[0] * args.width / fr.shape[1])))
            ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, args.q])
            fid += 1
            try:
                ws.send(struct.pack("<dI", time.time(), fid) + jpg.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)
                S.sent += 1
                last_seq = seq
            except Exception as e:
                S.msg = f"send error: {e}"
        dt = time.time() - t0
        time.sleep(max(0.0, period - dt))


def recv_loop(args, vcam):
    while S.running:
        try:
            S.msg = "connecting..."
            ws = websocket.create_connection(args.url, timeout=20, enable_multithread=True,
                                             header={"User-Agent": "avatar1-live"})
            ws.settimeout(10)
            S.sent = S.recv = 0
            S.ws = ws
            S.msg = "connected"
            print("connected to pod", flush=True)
            while S.running:
                op, data = ws.recv_data()
                if op == websocket.ABNF.OPCODE_TEXT:
                    print("pod:", data.decode(errors="ignore"), flush=True)
                    S.msg = "settings: " + data.decode(errors="ignore")[:90]
                    continue
                if op != websocket.ABNF.OPCODE_BINARY or len(data) < 16:
                    continue
                ts, fid, srv_ms = struct.unpack("<dIf", data[:16])
                S.recv += 1
                now = time.time()
                if srv_ms != -2.0:
                    lat = (now - ts) * 1000
                    S.lat_ms = lat if S.lat_ms is None else 0.9 * S.lat_ms + 0.1 * lat
                if srv_ms == -2.0:  # pod skipped this frame (a newer one arrived first)
                    continue
                if srv_ms < 0:
                    S.no_face = True
                    continue
                S.no_face = False
                S.srv_ms = srv_ms if S.srv_ms is None else 0.9 * S.srv_ms + 0.1 * srv_ms
                jpg = data[16:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                S.recv_times.append(now)
                S.recv_times = [t for t in S.recv_times if now - t < 2.0]
                S.fps = len(S.recv_times) / 2.0
                with S.lock:
                    S.out_jpg, S.out_img = jpg, img
                    S.out_seq += 1
                if vcam is not None:
                    try:
                        if img.shape[1] != vcam.width or img.shape[0] != vcam.height:
                            img = cv2.resize(img, (vcam.width, vcam.height))
                        vcam.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    except Exception as e:
                        S.msg = f"vcam error: {e}"
        except Exception as e:
            S.ws = None
            S.msg = f"disconnected ({e.__class__.__name__}: {e}); retrying"
            print(S.msg, flush=True)
            time.sleep(2)


def control(d):
    ws = S.ws
    if ws is not None:
        try:
            ws.send(json.dumps(d))
        except Exception as e:
            S.msg = f"control error: {e}"


# ---------------- local MJPEG for OBS Browser Source ----------------
class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/live.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while S.running:
                    with S.lock:
                        seq, jpg = S.out_seq, S.out_jpg
                    if jpg is None or seq == last:
                        time.sleep(0.01)
                        continue
                    last = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except Exception:
                return
        elif self.path.startswith("/live.jpg"):
            with S.lock:
                jpg = S.out_jpg
            if jpg is None:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)
        else:
            body = (b"<!doctype html><html><head><title>Avatar1 live</title><style>html,body{margin:0;"
                    b"background:#000;height:100%;overflow:hidden}img{width:100%;height:100%;object-fit:contain}"
                    b"</style></head><body><img src='/live.mjpg'></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--cam", type=int, default=cfg.get("cam", 0))
    ap.add_argument("--video", default=None, help="drive from a video/image file instead of the webcam (testing)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--q", type=int, default=70)
    ap.add_argument("--inflight", type=int, default=6)
    ap.add_argument("--no-vcam", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--http-port", type=int, default=8766)
    ap.add_argument("--seconds", type=float, default=0, help="auto-exit after N seconds (testing)")
    ap.add_argument("--snap-dir", default=None, help="save a rendered frame every 3 s here (testing)")
    args = ap.parse_args()
    if not args.url:
        if not cfg.get("url") or not cfg.get("token"):
            sys.exit(f"missing url/token: create {CFG_PATH}")
        args.url = f"{cfg['url'].rstrip('/')}/ws?k={cfg['token']}"

    vcam = None
    if not args.no_vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=960, height=540, fps=30, backend="obs")
            print("virtual camera:", vcam.device, flush=True)
        except Exception as e:
            print("virtual camera unavailable (install OBS Studio for 'OBS Virtual Camera'):", e, flush=True)
            vcam = None

    try:
        srv = ThreadingServer(("127.0.0.1", args.http_port), MJPEGHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"local stream: http://127.0.0.1:{args.http_port}/  (OBS Browser Source)", flush=True)
    except Exception as e:
        print("local http server failed:", e, flush=True)

    threading.Thread(target=cam_loop, args=(args,), daemon=True).start()
    threading.Thread(target=recv_loop, args=(args, vcam), daemon=True).start()
    threading.Thread(target=send_loop, args=(args,), daemon=True).start()

    t_start = time.time()
    last_print = 0
    last_snap = 0
    sources = {ord("1"): "tattoo_front.jpg", ord("2"): "clean_front.jpg", ord("3"): "necktattoo_front.jpg"}
    mult = 1.0
    win = "Avatar1 live (c=calibrate, e/a=exp/all, 1-3 source, q=quit)"
    while S.running:
        now = time.time()
        if now - last_print > 5:
            last_print = now
            print(f"fps {S.fps:4.1f} | latency {S.lat_ms or 0:6.0f} ms | pod render {S.srv_ms or 0:5.1f} ms | "
                  f"sent {S.sent} recv {S.recv} | {'NO FACE ' if S.no_face else ''}{S.msg} {S.cam_msg[:40]}", flush=True)
        with S.lock:
            img, cam = S.out_img, S.cam_frame
        if args.snap_dir and img is not None and now - last_snap > 3:
            last_snap = now
            os.makedirs(args.snap_dir, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            cv2.imwrite(os.path.join(args.snap_dir, f"live_{stamp}.jpg"), img)
            if cam is not None:
                cv2.imwrite(os.path.join(args.snap_dir, f"cam_{stamp}.jpg"), cam)
        if not args.no_window:
            if img is not None:
                view = img.copy()
                if cam is not None:  # small self-view in the corner
                    h = view.shape[0] // 4
                    pip = cv2.resize(cv2.flip(cam, 1), (int(cam.shape[1] * h / cam.shape[0]), h))
                    view[-h:, -pip.shape[1]:] = pip
                txt = f"{S.fps:.1f} fps  {S.lat_ms or 0:.0f} ms" + ("  NO FACE" if S.no_face else "")
                cv2.putText(view, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(win, view)
            else:
                wait = cam.copy() if cam is not None else np.zeros((540, 960, 3), np.uint8)
                lines = [S.msg[:70]] + ([S.cam_msg[i:i + 70] for i in range(0, len(S.cam_msg), 70)] if S.cam_msg else [])
                for i, line in enumerate(lines):
                    cv2.putText(wait, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(win, wait)
            k = cv2.waitKey(15) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("c"):
                control({"cmd": "calibrate"})
            elif k == ord("e"):
                control({"region": "exp"})
            elif k == ord("a"):
                control({"region": "all"})
            elif k in sources:
                control({"source": sources[k]})
            elif k in (ord("+"), ord("=")):
                mult = min(2.0, mult + 0.1); control({"multiplier": mult})
            elif k == ord("-"):
                mult = max(0.3, mult - 0.1); control({"multiplier": mult})
        else:
            time.sleep(0.05)
        if args.seconds and now - t_start > args.seconds:
            break
    S.running = False
    print(f"final: fps {S.fps:.1f} latency {S.lat_ms or 0:.0f} ms pod render {S.srv_ms or 0:.1f} ms sent {S.sent} recv {S.recv}", flush=True)
    time.sleep(0.3)
    if vcam is not None:
        vcam.close()


if __name__ == "__main__":
    main()
