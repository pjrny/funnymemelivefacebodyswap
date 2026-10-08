#!/usr/bin/env python
"""
pod_live_server.py - realtime LivePortrait server for the Runpod GPU pod.

Keeps LivePortrait loaded on the GPU and animates an Avatar1 photo with frames
streamed from Oscar's webcam.

Endpoints (all need ?k=<token>, token lives in /workspace/live/token):
  WS   /ws            binary in : <12-byte header: float64 client_ts, uint32 frame_id> + JPEG
                      binary out: <16-byte header: float64 client_ts, uint32 frame_id, float32 server_ms> + JPEG
                      text in   : JSON control, e.g. {"cmd":"calibrate"} / {"region":"all"}
  POST /frame         raw JPEG body -> rendered JPEG (simple, one frame per request)
  GET  /live.jpg      latest rendered frame
  GET  /live.mjpg     MJPEG stream of rendered frames (OBS Browser Source / browser)
  GET  /view          HTML page showing the MJPEG stream (good for OBS Browser Source)
  GET  /stats         JSON timings
  POST /control       JSON body, same keys as WS text control

Run:  /workspace/live/start_live_server.sh   (listens on 0.0.0.0:8080 = Runpod HTTP proxy port)
"""
import os, sys, time, json, struct, asyncio, threading, secrets, argparse
from concurrent.futures import ThreadPoolExecutor

# The pod reports 128 CPUs but its cgroup quota is ~18 cores; letting torch/OpenMP spawn
# 128 threads makes every small CPU op ~1000x slower. Pin thread pools small.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
LP_DIR = os.environ.get("LP_DIR", "/workspace/LivePortrait")
sys.path.insert(0, LP_DIR)
os.chdir(LP_DIR)

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.io import encode_jpeg
torch.backends.cudnn.benchmark = True
torch.set_num_threads(6)
cv2.setNumThreads(4)

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix, headpose_pred_to_degree
from src.utils.crop import crop_image_by_bbox, parse_bbox_from_landmark
from src.utils.io import load_image_rgb, resize_to_limit, contiguous
from src.utils.helper import calc_motion_multiplier
import src.modules.util as lp_util
import src.modules.dense_motion as lp_dm

# --- make LivePortrait's warping network CUDA-graph capturable ---
# (it builds coordinate grids / zeros on the CPU and copies them to the GPU on every call)
_orig_grid = lp_util.make_coordinate_grid
_grid_cache = {}


def _cached_grid(spatial_size, ref, **kw):
    key = (tuple(int(x) for x in spatial_size), ref.dtype, str(ref.device))
    g = _grid_cache.get(key)
    if g is None:
        g = _orig_grid(spatial_size, ref, **kw)
        _grid_cache[key] = g
    return g


def _heatmap_repr(self, feature, kp_driving, kp_source):
    spatial_size = feature.shape[3:]
    gd = lp_util.kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
    gs = lp_util.kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
    heatmap = gd - gs
    zeros = torch.zeros(heatmap.shape[0], 1, *spatial_size, dtype=heatmap.dtype, device=heatmap.device)
    return torch.cat([zeros, heatmap], dim=1).unsqueeze(2)


lp_util.make_coordinate_grid = _cached_grid
lp_dm.make_coordinate_grid = _cached_grid
lp_dm.DenseMotionNetwork.create_heatmap_representations = _heatmap_repr

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

LIVE_DIR = "/workspace/live"
TOKEN_FILE = os.path.join(LIVE_DIR, "token")
OUT_JPG = "/workspace/avatar1/out/live.jpg"


def load_token():
    os.makedirs(LIVE_DIR, exist_ok=True)
    if os.environ.get("LIVE_TOKEN"):
        return os.environ["LIVE_TOKEN"]
    if not os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "w") as f:
            f.write(secrets.token_urlsafe(18))
        os.chmod(TOKEN_FILE, 0o600)
    return open(TOKEN_FILE).read().strip()


class Engine:
    """LivePortrait kept warm on the GPU; drives one source photo with live frames."""

    def __init__(self, source, region="exp", out_w=960, out_h=540, jpeg_q=80,
                 multiplier=1.0, smooth=0.5, driving_option="pose-friendly", redetect_every=30):
        self.inf = InferenceConfig()
        self.inf.flag_use_half_precision = True
        self.crop_cfg = CropConfig()
        self.w = LivePortraitWrapper(inference_cfg=self.inf)
        self.cropper = Cropper(crop_cfg=self.crop_cfg)
        self.device = self.w.device
        self.region = region
        self.out_w, self.out_h, self.jpeg_q = out_w, out_h, jpeg_q
        self.multiplier = multiplier
        self.smooth = smooth  # EMA weight of the previous motion (0 = off)
        self.driving_option = driving_option
        self.redetect_every = redetect_every
        self.lock_prep = threading.Lock()
        self.lock_gpu = threading.Lock()
        self.lmk_every = int(os.environ.get("LIVE_LMK_EVERY", "2"))
        self.stats = {"frames": 0, "no_face": 0}
        self.t_hist = []
        self.set_source(source)

    # ---------- source ----------
    @torch.no_grad()
    def set_source(self, path):
        img = load_image_rgb(path)
        img = resize_to_limit(img, self.inf.source_max_dim, self.inf.source_division)
        crop_info = self.cropper.crop_source_image(img, self.crop_cfg)
        if crop_info is None:
            raise RuntimeError(f"No face in source {path}")
        w = self.w
        I_s = w.prepare_source(crop_info["img_crop_256x256"])
        self.x_s_info = w.get_kp_info(I_s)
        self.x_c_s = self.x_s_info["kp"]
        self.R_s = get_rotation_matrix(self.x_s_info["pitch"], self.x_s_info["yaw"], self.x_s_info["roll"])
        self.f_s = w.extract_feature_3d(I_s)
        self.x_s = w.transform_keypoint(self.x_s_info)
        self.lip_delta0 = None
        if self.inf.flag_normalize_lip and crop_info["lmk_crop"] is not None:
            c = w.calc_combined_lip_ratio([0.0], crop_info["lmk_crop"])
            if c[0][0] >= self.inf.lip_normalize_threshold:
                self.lip_delta0 = w.retarget_lip(self.x_s, c)
        # Output canvas: fit the (portrait) source into out_w x out_h, fill sides with a blurred copy.
        H, W = img.shape[:2]
        s = min(self.out_w / W, self.out_h / H)
        nw, nh = int(round(W * s)), int(round(H * s))
        ox, oy = (self.out_w - nw) // 2, (self.out_h - nh) // 2
        T = np.array([[s, 0, ox], [0, s, oy], [0, 0, 1]], np.float32)
        M_c2o = crop_info["M_c2o"].astype(np.float32)
        if M_c2o.shape[0] == 2:
            M_c2o = np.vstack([M_c2o, [0, 0, 1]]).astype(np.float32)
        self.M_out = (T @ M_c2o)[:2]
        fg = cv2.warpAffine(img, T[:2], (self.out_w, self.out_h), flags=cv2.INTER_AREA)
        sb = max(self.out_w / W, self.out_h / H)
        bg = cv2.resize(img, (int(W * sb) + 1, int(H * sb) + 1), interpolation=cv2.INTER_AREA)
        by, bx = (bg.shape[0] - self.out_h) // 2, (bg.shape[1] - self.out_w) // 2
        bg = bg[by:by + self.out_h, bx:bx + self.out_w]
        bg = cv2.GaussianBlur(bg, (0, 0), 25)
        bg = (bg.astype(np.float32) * 0.6).astype(np.uint8)
        canvas = bg.copy()
        canvas[oy:oy + nh, ox:ox + nw] = fg[oy:oy + nh, ox:ox + nw]
        self.canvas = canvas.astype(np.float32)
        mask_crop = self.inf.mask_crop  # 512x512x3 uint8 (crop space)
        mask_out = cv2.warpAffine(mask_crop, self.M_out, (self.out_w, self.out_h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        # only paste where the source photo actually is (the face crop can extend past the photo edges)
        valid = np.zeros((self.out_h, self.out_w), np.float32)
        valid[oy + 2:oy + nh - 2, ox + 2:ox + nw - 2] = 1.0
        valid = cv2.GaussianBlur(valid, (0, 0), 2)
        mask_out = mask_out * valid[..., None]
        # GPU paste-back: sampling grid from output pixels back into the 512x512 crop
        Minv = np.linalg.inv(np.vstack([self.M_out, [0, 0, 1]]))
        us, vs = np.meshgrid(np.arange(self.out_w, dtype=np.float32), np.arange(self.out_h, dtype=np.float32))
        pts = np.stack([us, vs, np.ones_like(us)], -1) @ Minv.T.astype(np.float32)
        gx = (2 * pts[..., 0] + 1) / 512.0 - 1
        gy = (2 * pts[..., 1] + 1) / 512.0 - 1
        dev = self.device
        grid = torch.from_numpy(np.stack([gx, gy], -1)[None]).to(dev)
        mask_t = torch.from_numpy(mask_out).permute(2, 0, 1)[None].to(dev)
        canvas_t = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        if not hasattr(self, "st"):
            # static buffers: CUDA graphs replay against these fixed addresses
            self.st = {"f_s": self.f_s.clone(), "x_s": self.x_s.clone(), "grid": grid, "mask": mask_t,
                       "canvas": canvas_t, "I_d": torch.zeros(1, 3, 256, 256, device=dev),
                       "x_d": self.x_s.clone()}
        else:
            for k, v in (("f_s", self.f_s), ("x_s", self.x_s), ("grid", grid), ("mask", mask_t), ("canvas", canvas_t)):
                self.st[k].copy_(v)
        self._build_graphs()
        self.source_path = path
        self.source_rgb = img
        self.reset()


    def _ctx(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, cache_enabled=False)

    def _kp_body(self):
        w, st = self.w, self.st
        kp = w.motion_extractor(st["I_d"])
        return {k: v.float() for k, v in kp.items() if torch.is_tensor(v)}

    def _refine(self, out):
        out["pitch"] = headpose_pred_to_degree(out["pitch"])[:, None]
        out["yaw"] = headpose_pred_to_degree(out["yaw"])[:, None]
        out["roll"] = headpose_pred_to_degree(out["roll"])[:, None]
        out["kp"] = out["kp"].reshape(1, -1, 3)
        out["exp"] = out["exp"].reshape(1, -1, 3)
        out["R"] = get_rotation_matrix(out["pitch"], out["yaw"], out["roll"])
        return out

    def _render_body(self):
        w, st = self.w, self.st
        r = w.warping_module(st["f_s"], kp_source=st["x_s"], kp_driving=st["x_d"])
        o = w.spade_generator(feature=r["out"]).float().clamp(0, 1)
        warped = F.grid_sample(o, st["grid"], mode="bilinear", padding_mode="zeros", align_corners=False)
        comp = st["mask"] * warped + (1 - st["mask"]) * st["canvas"]
        return (comp[0] * 255).round().to(torch.uint8)  # 3xHxW uint8 (RGB)

    @torch.no_grad()
    def _build_graphs(self):
        """Capture motion-extractor and warp+decode+paste as CUDA graphs. The pod's CPU is slow
        (~50us per kernel launch), so eager mode spends ~200 ms just launching kernels."""
        self.use_graphs = os.environ.get("LIVE_NO_GRAPHS") != "1"
        if not self.use_graphs:
            return
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), self._ctx():
            for _ in range(3):
                self._kp_body(); self._render_body()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.g_kp = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_kp), self._ctx():
            self.g_kp_out = self._kp_body()
        self.g_r = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g_r), self._ctx():
            self.g_r_out = self._render_body()
        torch.cuda.synchronize()

    def _kp(self, crop_rgb256):
        t = torch.from_numpy(crop_rgb256).to(self.device, non_blocking=True).permute(2, 0, 1)[None].float() / 255.0
        self.st["I_d"].copy_(t)
        if self.use_graphs:
            self.g_kp.replay()
            return self._refine({k: v.clone() for k, v in self.g_kp_out.items()})
        with self._ctx():
            return self._refine(self._kp_body())

    def _render(self, x_new):
        self.st["x_d"].copy_(x_new)
        if self.use_graphs:
            self.g_r.replay()
            img = self.g_r_out
        else:
            with self._ctx():
                img = self._render_body()
        return img  # 3xHxW uint8 RGB on the GPU

    def reset(self):
        """Next frame becomes the neutral reference (calibrate)."""
        self.x_d_0_info = None
        self.R_d_0 = None
        self.motion_mult = None
        self.track_lmk = None
        self.bbox = None
        self.frame_idx = 0
        self.prev_x = None

    # ---------- driving ----------
    def _landmarks(self, frame_rgb):
        need_detect = self.track_lmk is None or (self.redetect_every and self.frame_idx % self.redetect_every == 0)
        if need_detect:
            faces = self.cropper.face_analysis_wrapper.get(
                contiguous(frame_rgb[..., ::-1]), flag_do_landmark_2d_106=True, direction="large-small")
            if len(faces) == 0:
                self.track_lmk = None
                return None
            lmk = self.cropper.human_landmark_runner.run(frame_rgb, faces[0].landmark_2d_106)
        else:
            lmk = self.cropper.human_landmark_runner.run(frame_rgb, self.track_lmk)
        self.track_lmk = lmk
        return lmk

    def _drive_crop(self, frame_rgb):
        if self.bbox is not None and self.lmk_every > 1 and self.frame_idx % self.lmk_every:
            # the crop window moves slowly; reuse it between landmark updates
            ret = crop_image_by_bbox(frame_rgb, self.bbox.tolist(), lmk=self.track_lmk, dsize=256, flag_rot=False, borderValue=(0, 0, 0))
            return ret["img_crop"]
        lmk = self._landmarks(frame_rgb)
        if lmk is None:
            return None
        bb = parse_bbox_from_landmark(lmk, scale=self.crop_cfg.scale_crop_driving_video,
                                      vx_ratio_crop_driving_video=self.crop_cfg.vx_ratio_crop_driving_video,
                                      vy_ratio=self.crop_cfg.vy_ratio_crop_driving_video)["bbox"]
        box = np.array([bb[0, 0], bb[0, 1], bb[2, 0], bb[2, 1]], np.float32)
        if self.bbox is None:
            self.bbox = box
        else:  # light smoothing of the crop window to stop jitter
            self.bbox = 0.7 * self.bbox + 0.3 * box
        ret = crop_image_by_bbox(frame_rgb, self.bbox.tolist(), lmk=lmk, dsize=256, flag_rot=False, borderValue=(0, 0, 0))
        return ret["img_crop"]

    # ---- 3 pipeline stages: prep (CPU+ORT) -> gpu (LivePortrait) -> encode (CPU) ----
    def prep(self, buf):
        """JPEG bytes -> 256x256 RGB face crop (or None if no face)."""
        t0 = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None, 0.0
        with self.lock_prep:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = self._drive_crop(frame_rgb)
            self.frame_idx += 1
        if crop is None:
            self.stats["no_face"] += 1
        elif crop.shape[0] != 256:
            crop = cv2.resize(crop, (256, 256))
        return crop, (time.perf_counter() - t0) * 1e3

    @torch.no_grad()
    def gpu(self, crop):
        """animate + JPEG-encode on the GPU (nvjpeg); returns JPEG bytes"""
        t0 = time.perf_counter()
        with self.lock_gpu:
            img = self._animate(crop)
            try:
                jpg = encode_jpeg(img, quality=self.jpeg_q).cpu().numpy().tobytes()
            except Exception:
                rgb = img.permute(1, 2, 0).contiguous().cpu().numpy()
                jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])[1].tobytes()
        return jpg, (time.perf_counter() - t0) * 1e3

    def record(self, info):
        self.stats["frames"] += 1
        self.t_hist.append((time.time(), info["total_ms"]))
        self.t_hist = self.t_hist[-300:]

    def _animate(self, crop):
        w = self.w
        x_d = self._kp(crop)
        R_d = x_d["R"]
        if self.x_d_0_info is None:
            self.x_d_0_info = {k: v.clone() for k, v in x_d.items() if torch.is_tensor(v)}
            self.R_d_0 = R_d.clone()
        x0, xs, reg = self.x_d_0_info, self.x_s_info, self.region
        if reg in ("all", "exp"):
            delta = xs["exp"] + (x_d["exp"] - x0["exp"])
        else:
            delta = xs["exp"].clone()
        if reg in ("all", "pose"):
            R_new = (R_d @ self.R_d_0.permute(0, 2, 1)) @ self.R_s
            t_new = xs["t"] + (x_d["t"] - x0["t"])
        else:
            R_new = self.R_s
            t_new = xs["t"].clone()
        scale = xs["scale"] * (x_d["scale"] / x0["scale"]) if reg == "all" else xs["scale"]
        t_new[..., 2] = 0
        x_new = scale * (self.x_c_s @ R_new + delta) + t_new
        if self.driving_option == "expression-friendly":
            if self.motion_mult is None:
                self.motion_mult = calc_motion_multiplier(self.x_s, x_new)
                self.x_d_0_new = x_new
            x_new = (x_new - self.x_d_0_new) * self.motion_mult + self.x_s
        x_new = w.stitching(self.x_s, x_new)
        if self.lip_delta0 is not None:
            x_new = x_new + self.lip_delta0
        x_new = self.x_s + (x_new - self.x_s) * self.multiplier
        if self.smooth and self.prev_x is not None:
            x_new = self.smooth * self.prev_x + (1 - self.smooth) * x_new
        self.prev_x = x_new
        return self._render(x_new)

    def process_jpeg(self, buf):
        """Single-shot (non-pipelined) path used by POST /frame."""
        crop, a = self.prep(buf)
        if crop is None:
            return None, {"no_face": True}
        jpg, b = self.gpu(crop)
        info = {"crop_ms": a, "gpu_ms": b, "total_ms": a + b}
        self.record(info)
        return jpg, info

    def control(self, d):
        with self.lock_prep, self.lock_gpu:
            if d.get("cmd") == "calibrate":
                self.reset()
            if "region" in d and d["region"] in ("exp", "all", "pose", "lip", "eyes"):
                self.region = d["region"]; self.reset()
            if "multiplier" in d:
                self.multiplier = float(d["multiplier"])
            if "smooth" in d:
                self.smooth = max(0.0, min(0.95, float(d["smooth"])))
            if "driving_option" in d:
                self.driving_option = d["driving_option"]; self.reset()
            if "jpeg_q" in d:
                self.jpeg_q = int(d["jpeg_q"])
            if "source" in d:
                p = d["source"]
                if not os.path.isabs(p):
                    p = os.path.join("/workspace/avatar1", p)
                self.set_source(p)
        return self.summary()

    def summary(self):
        now = time.time()
        recent = [ms for t, ms in self.t_hist if now - t < 5]
        return {"source": self.source_path, "region": self.region, "multiplier": self.multiplier,
                "smooth": self.smooth, "driving_option": self.driving_option,
                "out": [self.out_w, self.out_h], "frames": self.stats["frames"], "no_face": self.stats["no_face"],
                "render_fps_5s": round(len(recent) / 5.0, 1),
                "avg_render_ms_5s": round(sum(recent) / len(recent), 1) if recent else None}


# ---------------- web app ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--source", default="/workspace/avatar1/tattoo_front.jpg")
ap.add_argument("--region", default="exp")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--out_w", type=int, default=960)
ap.add_argument("--out_h", type=int, default=540)
ap.add_argument("--smooth", type=float, default=0.3)
args, _ = ap.parse_known_args()

TOKEN = load_token()
ENGINE = Engine(args.source, region=args.region, out_w=args.out_w, out_h=args.out_h, smooth=args.smooth)
EXEC = ThreadPoolExecutor(max_workers=1)      # GPU stage + control (serialised)
EXEC_CPU = ThreadPoolExecutor(max_workers=3)  # decode/crop and JPEG encode
app = FastAPI()
latest = {"jpg": None, "seq": 0}
new_frame = None  # asyncio.Condition, made on startup


def check(k):
    if k != TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


@app.on_event("startup")
async def _startup():
    global new_frame
    new_frame = asyncio.Condition()
    # warm up kernels with the source itself as a driving frame
    bgr = cv2.cvtColor(ENGINE.source_rgb, cv2.COLOR_RGB2BGR)
    ok, j = cv2.imencode(".jpg", bgr)
    for _ in range(3):
        await asyncio.get_running_loop().run_in_executor(EXEC, ENGINE.process_jpeg, j.tobytes())
    ENGINE.reset()
    print("[live] ready; warmup done", ENGINE.summary(), flush=True)


async def publish(jpg):
    latest["jpg"] = jpg
    latest["seq"] += 1
    if latest["seq"] % 15 == 0:
        try:
            with open(OUT_JPG + ".tmp", "wb") as f:
                f.write(jpg)
            os.replace(OUT_JPG + ".tmp", OUT_JPG)
        except Exception:
            pass
    async with new_frame:
        new_frame.notify_all()


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/stats")
async def stats(k: str = ""):
    check(k)
    return ENGINE.summary()


@app.post("/control")
async def control(request: Request, k: str = ""):
    check(k)
    d = await request.json()
    res = await asyncio.get_running_loop().run_in_executor(EXEC, ENGINE.control, d)
    return res


@app.post("/frame")
async def frame(request: Request, k: str = ""):
    check(k)
    buf = await request.body()
    jpg, info = await asyncio.get_running_loop().run_in_executor(EXEC, ENGINE.process_jpeg, buf)
    if jpg is None:
        return JSONResponse(info, status_code=422)
    await publish(jpg)
    return Response(jpg, media_type="image/jpeg", headers={"X-Render-Ms": f"{info['total_ms']:.1f}", "X-Timing": json.dumps({k: round(v, 1) for k, v in info.items()})})


@app.get("/live.jpg")
async def live_jpg(k: str = ""):
    check(k)
    if latest["jpg"] is None:
        raise HTTPException(404, "no frame yet")
    return Response(latest["jpg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/live.mjpg")
async def live_mjpg(k: str = ""):
    check(k)

    async def gen():
        last = -1
        while True:
            async with new_frame:
                await new_frame.wait_for(lambda: latest["seq"] != last)
            last = latest["seq"]
            j = latest["jpg"]
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/view")
async def view(k: str = ""):
    check(k)
    return HTMLResponse(f"""<!doctype html><html><head><title>Avatar1 live</title>
<style>html,body{{margin:0;background:#000;height:100%;overflow:hidden}}img{{width:100%;height:100%;object-fit:contain}}</style></head>
<body><img src="/live.mjpg?k={k}"></body></html>""")


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    if ws.query_params.get("k") != TOKEN:
        await ws.close(code=4401)
        return
    await ws.accept()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(EXEC, ENGINE.control, {"cmd": "calibrate"})  # first frame = neutral
    slot = {"buf": None}
    ev_in = asyncio.Event()
    q_gpu = asyncio.Queue(maxsize=1)

    async def reader():
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes"):
                old = slot["buf"]
                slot["buf"] = msg["bytes"]  # newest frame wins; stale frames are dropped
                ev_in.set()
                if old is not None and len(old) >= 12:  # tell the client so its in-flight count stays right
                    ts, fid = struct.unpack("<dI", old[:12])
                    await ws.send_bytes(struct.pack("<dIf", ts, fid, -2.0))
            elif msg.get("text"):
                try:
                    res = await loop.run_in_executor(EXEC, ENGINE.control, json.loads(msg["text"]))
                    await ws.send_text(json.dumps(res))
                except Exception as e:
                    await ws.send_text(json.dumps({"error": str(e)}))

    async def stage_prep():
        while True:
            await ev_in.wait()
            ev_in.clear()
            buf, slot["buf"] = slot["buf"], None
            if not buf or len(buf) < 13:
                continue
            ts, fid = struct.unpack("<dI", buf[:12])
            crop, a = await loop.run_in_executor(EXEC_CPU, ENGINE.prep, buf[12:])
            if crop is None:
                await ws.send_bytes(struct.pack("<dIf", ts, fid, -1.0))
                continue
            await q_gpu.put((ts, fid, crop, a))

    async def stage_gpu():
        while True:
            ts, fid, crop, a = await q_gpu.get()
            jpg, b = await loop.run_in_executor(EXEC, ENGINE.gpu, crop)
            info = {"crop_ms": a, "gpu_ms": b, "total_ms": a + b}
            ENGINE.record(info)
            await ws.send_bytes(struct.pack("<dIf", ts, fid, info["total_ms"]) + jpg)
            await publish(jpg)

    tasks = [asyncio.create_task(f()) for f in (reader, stage_prep, stage_gpu)]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            if t.exception() and not isinstance(t.exception(), WebSocketDisconnect):
                print("[live] ws error:", repr(t.exception()), flush=True)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning", ws_max_size=8 * 1024 * 1024)
