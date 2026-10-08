#!/usr/bin/env python
"""
capture_body.py - laptop side of the Avatar1 live FULL-BODY pipeline.

Webcam -> pose detection ON THE LAPTOP (rtmlib RTMPose, CPU, ~15-20 fps) -> only 18 skeleton keypoints go to the
Runpod pod (pod_body_server.py) over a WebSocket -> rendered Avatar1 frames come back -> preview window
'Avatar1 body' + OBS Virtual Camera (pyvirtualcam) + local MJPEG at http://127.0.0.1:8766/ (OBS Browser Source).
No camera pixels leave the laptop.

Usage (from the FaceBodySwapStream folder):  live\\.venv\\Scripts\\python.exe live\\capture_body.py
Keys in the preview window:
  1 Main (default)  2 Hourglass  3 Outfits  4 Testing   t tattoos on/off   [ / ] fewer/more steps (speed/quality)
  s new seed (new look)   f full-frame / auto-frame   q or Esc quit
"""
import argparse, json, os, struct, sys, threading, time
import http.server, socketserver
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2
import numpy as np
cv2.setNumThreads(2)
import websocket  # websocket-client

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "live_config_body.json")
THR = 0.3
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.cam_frame = None; self.cam_seq = 0
        self.kp = None; self.kp_seq = 0; self.kp_wh = (640, 480); self.pose_fps = 0.0
        self.out_jpg = None; self.out_img = None; self.out_seq = 0
        self.sent = 0; self.recv = 0; self.lat_ms = None; self.srv_ms = None; self.fps = 0.0
        self.no_person = False; self.msg = ""; self.cam_msg = ""; self.running = True; self.ws = None
        self.recv_times = []; self.settings = {}


S = State()


def load_cfg():
    return json.load(open(CFG_PATH)) if os.path.exists(CFG_PATH) else {}


# ---------------- webcam ----------------
def open_cam(args):
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def cam_loop(args):
    if args.video:
        still = cv2.imread(args.video) if args.video.lower().endswith((".jpg", ".jpeg", ".png")) else None
        cap = None if still is not None else cv2.VideoCapture(args.video)
        while S.running:
            if still is not None:
                fr = still
            else:
                ok, fr = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            if fr.shape[1] != args.width:
                fr = cv2.resize(fr, (args.width, int(fr.shape[0] * args.width / fr.shape[1])))
            with S.lock:
                S.cam_frame = fr; S.cam_seq += 1
            time.sleep(1 / 25)
        return
    cap, fails = None, 0
    while S.running:
        if cap is None:
            cap = open_cam(args)
            if cap is None:
                S.cam_msg = "No webcam found (camera privacy key?). Retrying..."
                time.sleep(3); continue
            S.cam_msg = ""
            print("webcam", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), flush=True)
        ok, fr = cap.read()
        if not ok:
            fails += 1
            if fails > 60:
                cap.release(); cap = None; fails = 0
            time.sleep(0.02); continue
        fails = 0
        with S.lock:
            S.cam_frame = fr; S.cam_seq += 1
    if cap is not None:
        cap.release()


# ---------------- pose (laptop CPU) ----------------
def _limit_ort_threads(n):
    """rtmlib creates its own onnxruntime sessions; cap their CPU threads so the laptop stays responsive."""
    import onnxruntime as ort
    if getattr(ort.InferenceSession, "_avatar_patched", False):
        return
    orig = ort.InferenceSession

    class Capped(orig):
        _avatar_patched = True

        def __init__(self, path_or_bytes, sess_options=None, providers=None, **k):
            so = sess_options or ort.SessionOptions()
            so.intra_op_num_threads = n
            so.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options=so, providers=providers, **k)
    ort.InferenceSession = Capped


def pose_loop(args):
    _limit_ort_threads(args.pose_threads)
    from rtmlib import Body, PoseTracker
    tracker = PoseTracker(Body, det_frequency=args.det_every, to_openpose=True, mode="lightweight",
                          backend="onnxruntime", device="cpu", tracking=False)
    last_seq, times = -1, []
    print("pose model ready (rtmlib RTMPose lightweight, CPU)", flush=True)
    while S.running:
        with S.lock:
            fr, seq = S.cam_frame, S.cam_seq
        if fr is None or seq == last_seq or time.time() - getattr(S, "_last_pose_t", 0) < 1.0 / args.pose_fps:
            time.sleep(0.005); continue
        last_seq = seq; S._last_pose_t = time.time()
        try:
            kps, scores = tracker(fr)
        except Exception as e:
            print("pose error:", e, flush=True); time.sleep(0.1); continue
        kp = None
        if kps is not None and len(kps):
            # pick the largest / most confident person
            best, bi = -1, 0
            for i in range(len(kps)):
                v = scores[i] > THR
                if v.sum() < 3:
                    continue
                pts = kps[i][v]
                area = (np.ptp(pts[:, 0]) + 1) * (np.ptp(pts[:, 1]) + 1) * v.sum()
                if area > best:
                    best, bi = area, i
            if best > 0:
                kp = np.concatenate([kps[bi].astype(np.float32), scores[bi].astype(np.float32)[:, None]], 1)
        now = time.time(); times.append(now); times[:] = [t for t in times if now - t < 2]
        with S.lock:
            S.kp = kp; S.kp_seq += 1; S.kp_wh = (fr.shape[1], fr.shape[0]); S.pose_fps = len(times) / 2


def draw_skel(kp, W, H, scale_x=1.0, scale_y=1.0):
    c = np.zeros((H, W, 3), np.uint8)
    if kp is None:
        return c
    for i, (a, b) in enumerate(LIMBS):
        if kp[a, 2] >= THR and kp[b, 2] >= THR:
            cv2.line(c, (int(kp[a, 0] * scale_x), int(kp[a, 1] * scale_y)), (int(kp[b, 0] * scale_x), int(kp[b, 1] * scale_y)),
                     COLORS[i][::-1], 2)
    for i in range(18):
        if kp[i, 2] >= THR:
            cv2.circle(c, (int(kp[i, 0] * scale_x), int(kp[i, 1] * scale_y)), 3, COLORS[i][::-1], -1)
    return c


# ---------------- network ----------------
def send_loop(args):
    period = 1.0 / args.fps
    last, fid, last_kp, last_kp_t = -1, 0, None, 0
    while S.running:
        t0 = time.time()
        ws = S.ws
        with S.lock:
            kp, seq, (cw, ch) = S.kp, S.kp_seq, S.kp_wh
        if kp is not None:
            last_kp, last_kp_t = kp, t0
        elif last_kp is not None and t0 - last_kp_t < 1.0:
            kp = last_kp  # brief dropout (e.g. turning): keep the last skeleton for up to 1 s
        if ws is not None and seq != last and kp is not None and (S.sent - S.recv) < args.inflight:
            fid += 1
            try:
                ws.send(struct.pack("<dIHH", time.time(), fid, cw, ch) + kp.astype(np.float32).tobytes(),
                        opcode=websocket.ABNF.OPCODE_BINARY)
                S.sent += 1; last = seq
            except Exception as e:
                S.msg = f"send error: {e}"
        S.no_person = kp is None
        time.sleep(max(0.0, period - (time.time() - t0)))


def recv_loop(args, vcam):
    while S.running:
        try:
            S.msg = "connecting..."
            ws = websocket.create_connection(args.url, timeout=20, enable_multithread=True,
                                             header={"User-Agent": "avatar1-body"})
            ws.settimeout(15)
            S.sent = S.recv = 0; S.ws = ws; S.msg = "connected"
            print("connected to pod", flush=True)
            while S.running:
                try:
                    op, data = ws.recv_data()
                except websocket.WebSocketTimeoutException:
                    ws.ping(); continue  # nothing to render (no person) - keep the connection
                if op == websocket.ABNF.OPCODE_TEXT:
                    txt = data.decode(errors="ignore")
                    print("pod:", txt, flush=True)
                    try:
                        S.settings = json.loads(txt)
                    except Exception:
                        pass
                    continue
                if op != websocket.ABNF.OPCODE_BINARY or len(data) < 16:
                    continue
                ts, fid, srv_ms = struct.unpack("<dIf", data[:16])
                S.recv += 1
                if srv_ms == -2.0 or srv_ms == -1.0:
                    continue  # skipped / no person -> hold last frame
                now = time.time()
                lat = (now - ts) * 1000
                S.lat_ms = lat if S.lat_ms is None else 0.9 * S.lat_ms + 0.1 * lat
                S.srv_ms = srv_ms if S.srv_ms is None else 0.9 * S.srv_ms + 0.1 * srv_ms
                jpg = data[16:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                S.recv_times.append(now); S.recv_times = [t for t in S.recv_times if now - t < 2.0]
                S.fps = len(S.recv_times) / 2.0
                with S.lock:
                    S.out_jpg, S.out_img = jpg, img; S.out_seq += 1
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
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            last = -1
            try:
                while S.running:
                    with S.lock:
                        seq, jpg = S.out_seq, S.out_jpg
                    if jpg is None or seq == last:
                        time.sleep(0.01); continue
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
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(jpg)
        else:
            body = (b"<!doctype html><html><head><title>Avatar1 body</title><style>html,body{margin:0;"
                    b"background:#000;height:100%;overflow:hidden}img{width:100%;height:100%;object-fit:contain}"
                    b"</style></head><body><img src='/live.mjpg'></body></html>")
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(body)


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


STYLE_KEYS = {ord("1"): "av1main", ord("2"): "av1hourglass", ord("3"): "av1outfits", ord("4"): "av1testing"}


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--cam", type=int, default=cfg.get("cam", 0))
    ap.add_argument("--video", default=None, help="drive from a video/image file instead of the webcam (testing)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--det-every", type=int, default=10)
    ap.add_argument("--pose-fps", type=float, default=12, help="cap laptop pose detection rate (CPU)")
    ap.add_argument("--pose-threads", type=int, default=3, help="CPU threads for pose detection")
    ap.add_argument("--no-vcam", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--http-port", type=int, default=8766)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--snap-dir", default=None, help="save rendered output + skeleton every 3 s (no camera frames)")
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
            print("virtual camera unavailable (is another app using OBS Virtual Camera?):", e, flush=True)
    try:
        srv = ThreadingServer(("127.0.0.1", args.http_port), MJPEGHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"local stream: http://127.0.0.1:{args.http_port}/  (OBS Browser Source)", flush=True)
    except Exception as e:
        print("local http server failed:", e, flush=True)

    for fn, a in ((cam_loop, (args,)), (pose_loop, (args,)), (recv_loop, (args, vcam)), (send_loop, (args,))):
        threading.Thread(target=fn, args=a, daemon=True).start()

    t_start = last_print = last_snap = 0
    t_start = time.time()
    steps, tattoo, frame_mode = 2, False, "auto"
    win = "Avatar1 body"
    while S.running:
        now = time.time()
        if now - last_print > 5:
            last_print = now
            print(f"fps {S.fps:4.1f} | latency {S.lat_ms or 0:6.0f} ms | pod render {S.srv_ms or 0:5.0f} ms | "
                  f"pose {S.pose_fps:4.1f} fps | sent {S.sent} recv {S.recv} | {'NO PERSON ' if S.no_person else ''}"
                  f"{S.msg} {S.cam_msg[:40]}", flush=True)
        with S.lock:
            img, cam, kp, (cw, ch) = S.out_img, S.cam_frame, S.kp, S.kp_wh
        if args.snap_dir and img is not None and now - last_snap > 3:
            last_snap = now
            os.makedirs(args.snap_dir, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            cv2.imwrite(os.path.join(args.snap_dir, f"body_{stamp}.jpg"), img)
            cv2.imwrite(os.path.join(args.snap_dir, f"skel_{stamp}.png"), draw_skel(kp, cw, ch))
        if not args.no_window:
            if img is not None:
                view = img.copy()
                h = view.shape[0] // 4; w = int(cw * h / ch)
                inset = draw_skel(kp, w, h, w / cw, h / ch)
                view[-h:, -w:] = cv2.addWeighted(view[-h:, -w:], 0.3, inset, 1.0, 0)
                st = S.settings
                txt = (f"{S.fps:.1f} fps {S.lat_ms or 0:.0f} ms | {st.get('style', '?')} "
                       f"{'tattoo ' if st.get('tattoo') else ''}steps {st.get('steps', '?')} {st.get('frame', '')}"
                       + ("  NO PERSON" if S.no_person else ""))
                cv2.putText(view, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(win, view)
            else:
                wait = np.zeros((540, 960, 3), np.uint8)
                if kp is not None:
                    wait[:ch * 540 // max(ch, 1), :] = 0
                    sk = draw_skel(kp, 720, 540, 720 / cw, 540 / ch); wait[:, 120:840] = sk
                for i, line in enumerate([S.msg[:80], S.cam_msg[:80], f"pose {S.pose_fps:.1f} fps"]):
                    cv2.putText(wait, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(win, wait)
            k = cv2.waitKey(10) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k in STYLE_KEYS:
                control({"style": STYLE_KEYS[k]})
            elif k == ord("t"):
                tattoo = not S.settings.get("tattoo", tattoo); control({"tattoo": tattoo})
            elif k == ord("["):
                steps = max(1, S.settings.get("steps", steps) - 1); control({"steps": steps})
            elif k == ord("]"):
                steps = min(8, S.settings.get("steps", steps) + 1); control({"steps": steps})
            elif k == ord("s"):
                control({"seed": None})
            elif k == ord("f"):
                frame_mode = "full" if S.settings.get("frame", frame_mode) == "auto" else "auto"
                control({"frame": frame_mode})
        else:
            time.sleep(0.05)
        if args.seconds and now - t_start > args.seconds:
            break
    S.running = False
    print(f"final: fps {S.fps:.1f} latency {S.lat_ms or 0:.0f} ms pod render {S.srv_ms or 0:.0f} ms pose {S.pose_fps:.1f} fps "
          f"sent {S.sent} recv {S.recv}", flush=True)
    time.sleep(0.3)
    if vcam is not None:
        vcam.close()


if __name__ == "__main__":
    main()
