#!/usr/bin/env python
"""
pod_live_server.py - realtime LivePortrait server for the Runpod GPU pod.

Keeps LivePortrait loaded on the GPU and animates Avatar1 with frames streamed from Oscar's webcam.
v2: angle sets (every photo in /workspace/avatar1/sets/<name>/ pre-encoded; each frame the photo
whose head pose best matches Oscar's is used, with hysteresis + crossfade) and a true face-swap
mode (InsightFace inswapper_128 + GFPGAN/GPEN, see face_swap_engine.py) that keeps Oscar's hair.
Control JSON: {"set":"1"} {"source":"clean/clean_01.jpg"} {"mode":"swap"|"lp"|"toggle"}
              {"angle":true|false} {"residual_deg":10} {"enhance":"toggle"|"cycle"|"gfpgan"|"gpen"}

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

SWAP_SETTINGS_PATH = os.environ.get("SWAP_SETTINGS", "/workspace/live/swap_settings.json")
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
AV_DIR = "/workspace/avatar1"
SETS_DIR = os.path.join(AV_DIR, "sets")
FS_MODELS = os.environ.get("FS_MODELS", "/workspace/models/fs")
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
        self.bank = {}
        self.set_name, self.entries, self.cur = None, [], None
        self.angle_on = True          # 'v' key: angle-aware photo switching
        self.residual_deg = float(os.environ.get("LIVE_RESIDUAL_DEG", "10"))  # extra head turn on top of the photo's own pose
        self.hyst_deg, self.hyst_frames, self.fade_frames = 6.0, 3, 4
        self.pose_gain = float(os.environ.get("LIVE_POSE_GAIN", "1.5"))
        self.pose_ema, self.target_pose, self.cand, self.cand_n = None, None, None, 0
        self.fade_from, self.fade_left, self.last_out, self.switches = None, 0, None, 0
        self.mode = "lp"              # 'lp' = LivePortrait avatar, 'swap' = true face swap (InsightFace)
        self.swapper = None
        self.swap_id_key = None
        if os.path.isdir(source) or not os.path.splitext(source)[1]:
            self.set_angle_set(source)
        else:
            self.set_source(source)

    # ---------- sources ----------
    # A "source" is one Avatar1 photo, fully pre-encoded on the GPU (appearance features, keypoints,
    # head pose, output canvas + paste grid). An "angle set" is a list of sources of the same avatar
    # at different head angles; every frame the source whose pose is closest to Oscar's head pose
    # is swapped into the CUDA-graph buffers (a few GPU memcpys, no re-encode).
    @torch.no_grad()
    def encode_source(self, path, aligned=False, flip=False):
        img = load_image_rgb(path)
        if flip:
            img = np.ascontiguousarray(img[:, ::-1])
        img = resize_to_limit(img, self.inf.source_max_dim, self.inf.source_division)
        crop_info = self.cropper.crop_source_image(img, self.crop_cfg)
        if crop_info is None:
            raise RuntimeError(f"No face in source {path}")
        w = self.w
        I_s = w.prepare_source(crop_info["img_crop_256x256"])
        info = w.get_kp_info(I_s)
        e = {"path": path + (" (mirrored)" if flip else ""), "x_s_info": info, "x_c_s": info["kp"],
             "R_s": get_rotation_matrix(info["pitch"], info["yaw"], info["roll"]),
             "f_s": w.extract_feature_3d(I_s), "x_s": w.transform_keypoint(info), "lip_delta0": None,
             "pose": (float(info["pitch"]), float(info["yaw"]), float(info["roll"])), "rgb": img}
        if self.inf.flag_normalize_lip and crop_info["lmk_crop"] is not None:
            c = w.calc_combined_lip_ratio([0.0], crop_info["lmk_crop"])
            if c[0][0] >= self.inf.lip_normalize_threshold:
                e["lip_delta0"] = w.retarget_lip(e["x_s"], c)
        H, W = img.shape[:2]
        M_c2o = crop_info["M_c2o"].astype(np.float32)
        if M_c2o.shape[0] == 2:
            M_c2o = np.vstack([M_c2o, [0, 0, 1]]).astype(np.float32)
        if aligned:
            # angle sets: put every photo's face crop at the same place/size so switching photos
            # does not make the head jump around the frame
            crop_px = 512.0 * float(np.sqrt(abs(np.linalg.det(M_c2o[:2, :2]))))
            cx, cy = (M_c2o @ np.array([256, 256, 1], np.float32))[:2]
            s = (0.95 * self.out_h) / crop_px
            ox, oy = self.out_w / 2 - s * cx, 0.45 * self.out_h - s * cy
        else:
            s = min(self.out_w / W, self.out_h / H)
            ox, oy = (self.out_w - W * s) / 2, (self.out_h - H * s) / 2
        T = np.array([[s, 0, ox], [0, s, oy], [0, 0, 1]], np.float32)
        M_out = (T @ M_c2o)[:2]
        fg = cv2.warpAffine(img, T[:2], (self.out_w, self.out_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        sb = max(self.out_w / W, self.out_h / H)
        bg = cv2.resize(img, (int(W * sb) + 1, int(H * sb) + 1), interpolation=cv2.INTER_AREA)
        by, bx = (bg.shape[0] - self.out_h) // 2, (bg.shape[1] - self.out_w) // 2
        bg = bg[by:by + self.out_h, bx:bx + self.out_w]
        bg = (cv2.GaussianBlur(bg, (0, 0), 25).astype(np.float32) * 0.6)
        inside = cv2.warpAffine(np.ones((H, W), np.float32), T[:2], (self.out_w, self.out_h), flags=cv2.INTER_LINEAR)
        inside = cv2.GaussianBlur(cv2.erode(inside, np.ones((5, 5), np.uint8)), (0, 0), 3)[..., None]
        canvas = inside * fg.astype(np.float32) + (1 - inside) * bg
        mask_crop = self.inf.mask_crop  # 512x512x3 uint8 (crop space)
        mask_out = cv2.warpAffine(mask_crop, M_out, (self.out_w, self.out_h), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask_out = mask_out * inside  # only paste where the source photo actually is
        Minv = np.linalg.inv(np.vstack([M_out, [0, 0, 1]]))
        us, vs = np.meshgrid(np.arange(self.out_w, dtype=np.float32), np.arange(self.out_h, dtype=np.float32))
        pts = np.stack([us, vs, np.ones_like(us)], -1) @ Minv.T.astype(np.float32)
        gx = (2 * pts[..., 0] + 1) / 512.0 - 1
        gy = (2 * pts[..., 1] + 1) / 512.0 - 1
        dev = self.device
        e["grid"] = torch.from_numpy(np.stack([gx, gy], -1)[None]).to(dev)
        e["mask"] = torch.from_numpy(np.ascontiguousarray(mask_out.transpose(2, 0, 1), np.float32))[None].to(dev)
        e["canvas"] = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1) / 255.0, np.float32))[None].to(dev)
        if os.environ.get("LIVE_DEBUG_CANVAS"):
            cv2.imwrite(f"/workspace/avatar1/out/canvas_{os.path.basename(path)}", cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return e

    def _activate(self, e):
        """make encoded source e the one being animated (GPU memcpy into the graph buffers)"""
        self.cur = e
        self.x_s_info, self.x_c_s, self.R_s = e["x_s_info"], e["x_c_s"], e["R_s"]
        self.f_s, self.x_s, self.lip_delta0 = e["f_s"], e["x_s"], e["lip_delta0"]
        self.source_path, self.source_rgb = e["path"], e["rgb"]
        if not hasattr(self, "st"):
            dev = self.device
            self.st = {"f_s": e["f_s"].clone(), "x_s": e["x_s"].clone(), "grid": e["grid"].clone(),
                       "mask": e["mask"].clone(), "canvas": e["canvas"].clone(),
                       "I_d": torch.zeros(1, 3, 256, 256, device=dev), "x_d": e["x_s"].clone()}
            self._build_graphs()
        else:
            for k in ("f_s", "x_s", "grid", "mask", "canvas"):
                self.st[k].copy_(e[k])
        self.motion_mult = None

    @torch.no_grad()
    def set_source(self, path):
        """single photo (keys 3/4/5): classic expression-only LivePortrait"""
        key = ("single", path)
        if key not in self.bank:
            self.bank[key] = [self.encode_source(path)]
        self.set_name, self.entries = None, self.bank[key]
        self._activate(self.entries[0])
        self.reset()

    @torch.no_grad()
    def set_angle_set(self, name):
        """angle set (keys 1/2): every photo in /workspace/avatar1/sets/<name>/ pre-encoded"""
        key = ("set", name)
        if key not in self.bank:
            d = name if os.path.isabs(name) else os.path.join(SETS_DIR, name)
            paths = sorted(p for p in os.listdir(d) if p.lower().endswith((".jpg", ".jpeg", ".png")))
            ents = []
            for p in paths:
                try:
                    ents.append(self.encode_source(os.path.join(d, p), aligned=True))
                except Exception as ex:  # noqa: BLE001
                    print("[live] skip", p, ex, flush=True)
            if not ents:
                raise RuntimeError(f"no usable photos in {d}")
            # fill the other side: mirror strongly turned photos whose mirror angle has no real photo
            if os.environ.get("LIVE_MIRROR", "1") == "1":
                for e in list(ents):
                    y = e["pose"][1]
                    if abs(y) > 20 and not any(abs(o["pose"][1] + y) < 12 for o in ents):
                        try:
                            ents.append(self.encode_source(e["path"], aligned=True, flip=True))
                        except Exception as ex:  # noqa: BLE001
                            print("[live] mirror skip", e["path"], ex, flush=True)
            # the most frontal photo is the "home" pose that a calibrated neutral face maps to
            ents.sort(key=lambda e: abs(e["pose"][1]) + abs(e["pose"][0]))
            self.bank[key] = ents
            print(f"[live] set {name}: " + ", ".join(f"{os.path.basename(e['path'])[:28]} y{e['pose'][1]:+.0f} p{e['pose'][0]:+.0f}" for e in ents), flush=True)
        self.set_name, self.entries = name, self.bank[key]
        self._activate(self.entries[0])
        self.reset()

    def _pick_source(self, x_d):
        """choose the photo whose head pose best matches Oscar's (relative to the calibrated neutral)"""
        if len(self.entries) < 2 or not self.angle_on:
            return
        home = self.entries[0]["pose"]
        rp = float(x_d["pitch"]) - float(self.x_d_0_info["pitch"])
        ry = float(x_d["yaw"]) - float(self.x_d_0_info["yaw"])
        if self.pose_ema is None:
            self.pose_ema = (rp, ry)
        else:
            a = 0.5
            self.pose_ema = (a * self.pose_ema[0] + (1 - a) * rp, a * self.pose_ema[1] + (1 - a) * ry)
        # webcam turns are smaller than the photo angles (+-45..60 deg): scale them up a bit
        tp, ty = home[0] + self.pose_gain * self.pose_ema[0], home[1] + self.pose_gain * self.pose_ema[1]
        self.target_pose = (tp, ty)

        def dist(e):
            return abs(e["pose"][1] - ty) + 0.6 * abs(e["pose"][0] - tp)
        best = min(self.entries, key=dist)
        if best is self.cur:
            self.cand, self.cand_n = None, 0
            return
        if dist(best) > dist(self.cur) - self.hyst_deg:
            self.cand, self.cand_n = None, 0
            return
        if best is self.cand:
            self.cand_n += 1
        else:
            self.cand, self.cand_n = best, 1
        if self.cand_n >= self.hyst_frames:
            self.fade_from = self.last_out.clone() if self.last_out is not None else None
            self.fade_left = self.fade_frames
            self._activate(best)
            self.prev_x = None
            self.switches += 1
            self.cand, self.cand_n = None, 0

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
        self.pose_ema = None
        self.cand, self.cand_n = None, 0
        if getattr(self, "entries", None) and self.cur is not self.entries[0]:
            self._activate(self.entries[0])

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
        if self.mode == "swap":
            return ("swap", frame), (time.perf_counter() - t0) * 1e3
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
        if isinstance(crop, tuple):
            return self._swap_frame(crop[1], t0)
        if self.mode == "swap":  # mode changed while this LivePortrait crop was queued
            return None, 0.0
        with self.lock_gpu:
            img = self._animate(crop)
            try:
                jpg = encode_jpeg(img.contiguous(), quality=self.jpeg_q).cpu().numpy().tobytes()
            except Exception:
                rgb = img.permute(1, 2, 0).contiguous().cpu().numpy()
                jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])[1].tobytes()
        return jpg, (time.perf_counter() - t0) * 1e3

    # ---------- true face-swap mode ----------
    def _swap_identity_paths(self):
        if self.set_name is not None:
            return [e["path"] for e in self.entries if not e["path"].endswith("(mirrored)")]
        return [self.source_path]

    def ensure_swapper(self):
        if self.swapper is None:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from face_swap_engine import FaceSwapEngine
            sc = self.swap_cfg()
            self.swapper = FaceSwapEngine(FS_MODELS, enhancer=sc["enhancer"] if sc["enhancer"] != "off" else "gpen",
                                          beard_on=bool(sc["beard"]), jaw_strength=float(sc["jaw"]))
            if sc["enhancer"] == "off":
                self.swapper.enhance_on = False
            # warm up (cuDNN autotune) on the current source photo
            bgr = cv2.cvtColor(self.source_rgb, cv2.COLOR_RGB2BGR)
        key = tuple(self._swap_identity_paths())
        if key != self.swap_id_key:
            used = self.swapper.set_identity(list(key))
            self.swap_id_key = key
            print(f"[live] swap identity from {len(used)} photo(s)", flush=True)
            bgr = cv2.cvtColor(self.source_rgb, cv2.COLOR_RGB2BGR)
            for _ in range(3):
                self.swapper.kps = None
                self.swapper.process(bgr)
            self.swapper.kps = None

    def _swap_frame(self, frame, t0):
        with self.lock_gpu:
            out = self.swapper.process(frame)
        if out is None:
            self.stats["no_face"] += 1
            return None, (time.perf_counter() - t0) * 1e3
        H, W = out.shape[:2]
        s = min(self.out_w / W, self.out_h / H)
        nw, nh = int(round(W * s)), int(round(H * s))
        if (nw, nh) != (self.out_w, self.out_h):
            canvas = np.zeros((self.out_h, self.out_w, 3), np.uint8)
            ox, oy = (self.out_w - nw) // 2, (self.out_h - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_LINEAR)
            out = canvas
        jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])[1].tobytes()
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
        self._pick_source(x_d)
        x0, xs, reg = self.x_d_0_info, self.x_s_info, self.region
        if reg in ("all", "exp"):
            delta = xs["exp"] + (x_d["exp"] - x0["exp"])
        else:
            delta = xs["exp"].clone()
        if reg in ("all", "pose"):
            R_new = (R_d @ self.R_d_0.permute(0, 2, 1)) @ self.R_s
            t_new = xs["t"] + (x_d["t"] - x0["t"])
        elif self.set_name is not None and self.target_pose is not None and self.residual_deg > 0:
            # angle set: the photo supplies most of the turn; add the small leftover difference
            L = self.residual_deg
            p, y, r = self.cur["pose"]
            dp = max(-L, min(L, self.target_pose[0] - p))
            dy = max(-L, min(L, self.target_pose[1] - y))
            dr = max(-L, min(L, float(x_d["roll"]) - float(x0["roll"])))
            dev = xs["pitch"].device
            R_new = get_rotation_matrix(torch.tensor([[p + dp]], device=dev), torch.tensor([[y + dy]], device=dev),
                                        torch.tensor([[r + dr]], device=dev))
            t_new = xs["t"].clone()
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
        img = self._render(x_new)
        if self.fade_left > 0 and self.fade_from is not None:
            a = self.fade_left / (self.fade_frames + 1.0)
            img = (img.float() * (1 - a) + self.fade_from.float() * a).round().to(torch.uint8)
            self.fade_left -= 1
        self.last_out = img if not self.use_graphs else img.clone()
        return img

    def process_jpeg(self, buf):
        """Single-shot (non-pipelined) path used by POST /frame."""
        crop, a = self.prep(buf)
        if crop is None:
            return None, {"no_face": True}
        jpg, b = self.gpu(crop)
        if jpg is None:
            return None, {"no_face": True}
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
                    p = os.path.join(AV_DIR, p)
                self.set_source(p)
            if "set" in d:
                self.set_angle_set(str(d["set"]))
            if "angle" in d:
                self.angle_on = bool(d["angle"]); self.reset()
            if "pose_gain" in d:
                self.pose_gain = max(0.5, min(3.0, float(d["pose_gain"])))
            if "residual_deg" in d:
                self.residual_deg = max(0.0, min(25.0, float(d["residual_deg"])))
            if "mode" in d:
                m = d["mode"]
                if m == "toggle":
                    m = "lp" if self.mode == "swap" else "swap"
                if m == "swap":
                    self.ensure_swapper()
                self.mode = m if m in ("lp", "swap") else self.mode
                self.reset()
            elif self.mode == "swap" and ("set" in d or "source" in d):
                self.ensure_swapper()  # new identity for the face swap
            if "step" in d:  # '-'/'=' keys: jaw slim strength in swap mode, motion strength in LivePortrait mode
                st = 0.1 * (1 if float(d["step"]) > 0 else -1)
                if self.mode == "swap" and self.swapper is not None:
                    self.swapper.jaw_strength = round(max(0.0, min(1.0, self.swapper.jaw_strength + st)), 2)
                else:
                    self.multiplier = round(max(0.3, min(2.0, self.multiplier + st)), 2)
            if "jaw" in d and self.swapper is not None:
                self.swapper.jaw_strength = max(0.0, min(1.0, float(d["jaw"])))
            if "beard" in d and self.swapper is not None:
                v = d["beard"]
                self.swapper.beard_on = (not self.swapper.beard_on) if v == "toggle" else bool(v)
                if self.swapper.lmk is None:
                    self.swapper.beard_on = False
            if "enhance" in d and self.swapper is not None:
                v = d["enhance"]
                if v == "toggle":
                    self.swapper.enhance_on = not self.swapper.enhance_on
                elif v in self.swapper.enh:
                    self.swapper.enhancer, self.swapper.enhance_on = v, True
                elif v == "cycle":  # gfpgan -> gpen -> off -> gfpgan
                    order = list(self.swapper.enh) + [None]
                    cur = self.swapper.enhancer if self.swapper.enhance_on else None
                    nxt = order[(order.index(cur) + 1) % len(order)]
                    self.swapper.enhance_on = nxt is not None
                    if nxt: self.swapper.enhancer = nxt
                else:
                    self.swapper.enhance_on = bool(v)
        if "swap_settings" in d and isinstance(d["swap_settings"], dict):
            self.apply_swap_settings(d["swap_settings"])
        if self.swapper is None:  # keys pressed before the swap models load still change the defaults
            c = self.swap_cfg()
            if "jaw" in d: c["jaw"] = round(max(0.0, min(1.0, float(d["jaw"]))), 2)
            if "beard" in d and d["beard"] != "toggle": c["beard"] = bool(d["beard"])
        self.save_swap_settings()
        return self.summary()

    # ---------- persistent face-swap settings (jaw / beard / enhancer); face-swap mode only ----------
    def swap_cfg(self):
        if not hasattr(self, "_swap_cfg"):
            c = {"jaw": float(os.environ.get("FS_JAW", "0.4")), "beard": os.environ.get("FS_BEARD", "1") == "1",
                 "enhancer": os.environ.get("FS_ENHANCER", "gpen")}
            try:
                with open(SWAP_SETTINGS_PATH) as f:
                    c.update({k: v for k, v in json.load(f).items() if k in c})
            except Exception:
                pass
            self._swap_cfg = c
        return self._swap_cfg

    def apply_swap_settings(self, d):
        """Set swap defaults (sent by the laptop on connect); applied live if the swapper is loaded."""
        c = self.swap_cfg()
        if "jaw" in d: c["jaw"] = round(max(0.0, min(1.0, float(d["jaw"]))), 2)
        if "beard" in d: c["beard"] = bool(d["beard"])
        if "enhancer" in d and d["enhancer"] in ("gfpgan", "gpen", "off"): c["enhancer"] = d["enhancer"]
        sw = self.swapper
        if sw is not None:
            sw.jaw_strength = c["jaw"]
            sw.beard_on = c["beard"] and sw.lmk is not None
            if c["enhancer"] == "off": sw.enhance_on = False
            elif c["enhancer"] in sw.enh: sw.enhancer, sw.enhance_on = c["enhancer"], True

    def save_swap_settings(self):
        c = dict(self.swap_cfg())
        sw = self.swapper
        if sw is not None:
            c = {"jaw": round(float(sw.jaw_strength), 2), "beard": bool(sw.beard_on),
                 "enhancer": sw.enhancer if (sw.enhance_on and sw.enhancer) else "off"}
            self._swap_cfg.update(c)
        if c != getattr(self, "_swap_saved", None):
            try:
                os.makedirs(os.path.dirname(SWAP_SETTINGS_PATH), exist_ok=True)
                tmp = SWAP_SETTINGS_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(dict(c, saved_at=time.strftime("%Y-%m-%d %H:%M:%S")), f)
                os.replace(tmp, SWAP_SETTINGS_PATH)
                self._swap_saved = c
            except Exception as e:
                print("[live] swap settings save failed:", e, flush=True)

    def summary(self):
        now = time.time()
        recent = [ms for t, ms in self.t_hist if now - t < 5]
        sw = self.swapper
        return {"mode": self.mode, "set": self.set_name, "source": os.path.basename(self.source_path),
                "set_size": len(self.entries), "angle": self.angle_on, "switches": self.switches,
                "residual_deg": self.residual_deg, "pose_gain": self.pose_gain,
                "enhancer": (sw.enhancer if sw.enhance_on else "off") if sw else None,
                "beard": sw.beard_on if sw else None, "jaw": sw.jaw_strength if sw else None,
                "swap_settings": dict(self.swap_cfg()),
                "swap_ms": {k: round(v, 1) for k, v in sw.t.items()} if sw else None,
                "region": self.region, "multiplier": self.multiplier,
                "smooth": self.smooth, "driving_option": self.driving_option,
                "out": [self.out_w, self.out_h], "frames": self.stats["frames"], "no_face": self.stats["no_face"],
                "render_fps_5s": round(len(recent) / 5.0, 1),
                "avg_render_ms_5s": round(sum(recent) / len(recent), 1) if recent else None}


# ---------------- web app ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--source", default=os.environ.get("LIVE_SOURCE", "1"), help="angle-set name under /workspace/avatar1/sets (e.g. 1) or a photo path")
ap.add_argument("--mode", default="lp", choices=["lp", "swap"])
ap.add_argument("--region", default="exp")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--out_w", type=int, default=960)
ap.add_argument("--out_h", type=int, default=540)
ap.add_argument("--smooth", type=float, default=0.3)
args, _ = ap.parse_known_args()

TOKEN = load_token()
ENGINE = Engine(args.source, region=args.region, out_w=args.out_w, out_h=args.out_h, smooth=args.smooth)
if args.mode == "swap" or os.environ.get("LIVE_PRELOAD_SWAP", "1") == "1":
    ENGINE.ensure_swapper()  # load InsightFace models now so the first 'm' press is instant
    ENGINE.control({"mode": args.mode})
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
            if jpg is None:
                await ws.send_bytes(struct.pack("<dIf", ts, fid, -1.0))
                continue
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
