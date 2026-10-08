#!/usr/bin/env python
"""
pod_body_server.py - GPU side of the Avatar1 live FULL-BODY pipeline (Runpod pod).

Laptop (capture_body.py) runs pose detection locally and sends only 18 OpenPose body keypoints
(no camera pixels leave the laptop). This server draws an OpenPose skeleton map, renders Avatar1 with
SD1.5 (Realistic Vision V6 B1) + our body LoRA + LCM-LoRA (fused, no CFG, 2-4 steps) + ControlNet OpenPose,
and streams JPEG frames back over the same WebSocket.

Protocol (same style as pod_live_server.py):
  WS  /ws?k=<token>
  client -> server binary:  <dIHH  (client_ts, frame_id, cam_w, cam_h) + float32[18*3]  (x_px, y_px, score)
  client -> server text:    JSON control {"style":"av1main"} {"tattoo":true} {"steps":3} {"seed":123} {"frame":"auto"|"full"}
  server -> client binary:  <dIf (client_ts, frame_id, server_ms) + JPEG  (server_ms -1 = no person, -2 = skipped)
  server -> client text:    JSON settings summary
Other endpoints: GET /health, GET /stats?k=, GET /live.jpg?k=, GET /pose.jpg?k= (last skeleton map)
Token: /workspace/live/token (created on first start).  Run:  python pod_body_server.py --port 8080
"""
import argparse, asyncio, json, math, os, secrets, struct, threading, time
os.environ.setdefault("OMP_NUM_THREADS", "6")  # Runpod pod CPUs are quota-limited; oversubscribed thread pools add latency
import numpy as np
import cv2
import torch
from PIL import Image
torch.set_num_threads(6)
cv2.setNumThreads(4)

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--models", default="/workspace/models")
ap.add_argument("--lora", default="/workspace/lora/avtr1_body_rv6.safetensors")
ap.add_argument("--lora_scale", type=float, default=0.9)
ap.add_argument("--w", type=int, default=512)
ap.add_argument("--h", type=int, default=768)
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--cn_scale", type=float, default=1.0)
ap.add_argument("--guidance", type=float, default=1.5, help=">1 enables CFG so the negative prompt (nudity terms) is applied")
ap.add_argument("--out_w", type=int, default=960)
ap.add_argument("--out_h", type=int, default=540)
ap.add_argument("--vae", default="taesd", choices=["taesd", "full"])
ap.add_argument("--compile", action="store_true")
ap.add_argument("--no_safety", action="store_true")
ap.add_argument("--safety_margin", type=float, default=0.3, help="hold frames whose NSFW probability exceeds this")
ap.add_argument("--frame", default="auto", choices=["auto", "full"])
args = ap.parse_args()

LIVE_DIR = "/workspace/live"
TOKEN_FILE = os.path.join(LIVE_DIR, "token")


def load_token():
    os.makedirs(LIVE_DIR, exist_ok=True)
    if not os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "w") as f:
            f.write(secrets.token_urlsafe(18))
        os.chmod(TOKEN_FILE, 0o600)
    return open(TOKEN_FILE).read().strip()


TOKEN = load_token()

# ---------------- prompts ----------------
STYLES = {
    "av1main": "av1main, white cropped tank top, light grey leggings, long hair down",
    # the LoRA learned Hourglass with a skin-tone bodysuit, which reads as nude at 2 steps; force an opaque dark outfit
    "av1hourglass": "av1hourglass, hourglass figure, black long-sleeve high-neck top, black high-waisted trousers, fully clothed, hair in bun",
    "av1outfits": "av1outfits, white t-shirt, black leggings, slicked back ponytail",
    "av1testing": "av1testing, slim build, hair in bun, white linen shirt, white linen trousers",
}
NEG = "lowres, blurry, deformed, bad anatomy, nude, naked, topless, nipples, cleavage, skin-colored clothing, lingerie"


def build_prompt(style, tattoo, back, framing):
    t = "face tattoos, neck tattoos" if tattoo else "no tattoos"
    view = "back view, from behind" if back else "front view, looking at viewer"
    return f"avtr1 woman, silver hair, {STYLES[style]}, {t}, {framing}, {view}, clothed, plain grey studio background, photo"


# ---------------- OpenPose drawing (controlnet_aux style) ----------------
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
THR = 0.3


def draw_openpose(kp, W, H):
    """kp: (18,3) in canvas pixels. Returns RGB uint8 HxW."""
    canvas = np.zeros((H, W, 3), np.uint8)
    sw = max(2, int(round(4 * max(W, H) / 512)))
    for i, (a, b) in enumerate(LIMBS):
        if kp[a, 2] < THR or kp[b, 2] < THR:
            continue
        X = (kp[a, 1], kp[b, 1]); Y = (kp[a, 0], kp[b, 0])
        mX, mY = np.mean(X), np.mean(Y)
        length = math.hypot(X[0] - X[1], Y[0] - Y[1])
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        poly = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), sw), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in COLORS[i]])
    for i in range(18):
        if kp[i, 2] >= THR:
            cv2.circle(canvas, (int(kp[i, 0]), int(kp[i, 1])), sw, COLORS[i], thickness=-1)
    return canvas


# ---------------- framing ----------------
class Framer:
    """Maps camera keypoints to the render canvas. 'full' = whole camera frame (letterboxed to canvas aspect);
    'auto' = smoothed crop around the person (portrait canvas), like a center-stage camera."""

    def __init__(self):
        self.box = None

    def compute(self, kp, cw, ch, W, H, mode):
        asp = W / H
        if mode == "full":
            bw, bh = (cw, cw / asp) if cw / ch < asp else (ch * asp, ch)
            bw, bh = max(bw, cw), max(bh, ch)
            if bw / bh > asp: bh = bw / asp
            else: bw = bh * asp
            return np.array([cw / 2 - bw / 2, ch / 2 - bh / 2, bw, bh])
        v = kp[kp[:, 2] >= THR]
        if len(v) < 3:
            return self.box if self.box is not None else self.compute(kp, cw, ch, W, H, "full")
        x0, y0 = v[:, 0].min(), v[:, 1].min(); x1, y1 = v[:, 0].max(), v[:, 1].max()
        # head room above nose/eyes, extend below to cover hands/legs
        h = max(y1 - y0, 1.0)
        y0 -= 0.25 * h + 0.06 * ch
        y1 += 0.12 * h
        hb = max(y1 - y0, 0.55 * ch)
        wb = max(x1 - x0 + 0.3 * hb, hb * asp)
        hb = max(hb, wb / asp); wb = hb * asp
        cx = (x0 + x1) / 2; cy = (y0 + y1) / 2 + 0.0 * hb
        box = np.array([cx - wb / 2, cy - hb / 2, wb, hb])
        if self.box is None:
            self.box = box
        else:
            a = 0.25
            self.box = (1 - a) * self.box + a * box
        return self.box

    @staticmethod
    def apply(kp, box, W, H):
        out = kp.copy()
        out[:, 0] = (kp[:, 0] - box[0]) * W / box[2]
        out[:, 1] = (kp[:, 1] - box[1]) * H / box[3]
        return out


# ---------------- engine ----------------
class Engine:
    def __init__(self):
        from diffusers import (StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, AutoencoderTiny,
                               LCMScheduler)
        M = args.models
        t0 = time.time()
        cn = ControlNetModel.from_pretrained(f"{M}/cn_openpose", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            f"{M}/Realistic_Vision_V6.0_NV_B1_fp16.safetensors", controlnet=cn, torch_dtype=torch.float16,
            load_safety_checker=False)
        if args.vae == "taesd":
            pipe.vae = AutoencoderTiny.from_pretrained(f"{M}/taesd", torch_dtype=torch.float16)
        else:
            pipe.vae = AutoencoderKL.from_single_file(f"{M}/vae-ft-mse-840000-ema-pruned.safetensors",
                                                      torch_dtype=torch.float16)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None; pipe.feature_extractor = None; pipe.requires_safety_checker = False
        pipe.to("cuda")  # fuse LoRAs on the GPU (fp16 fusing on CPU is very slow)
        pipe.load_lora_weights(os.path.dirname(args.lora), weight_name=os.path.basename(args.lora), adapter_name="av")
        pipe.load_lora_weights(f"{M}/lcm_lora", weight_name="pytorch_lora_weights.safetensors", adapter_name="lcm")
        pipe.set_adapters(["av", "lcm"], [args.lora_scale, 1.0])
        pipe.fuse_lora(adapter_names=["av", "lcm"])
        pipe.unload_lora_weights()
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.controlnet.to(memory_format=torch.channels_last)
        if args.compile:
            pipe.unet = torch.compile(pipe.unet, mode="max-autotune-no-cudagraphs", fullgraph=False)
            pipe.controlnet = torch.compile(pipe.controlnet, mode="max-autotune-no-cudagraphs", fullgraph=False)
        self.pipe = pipe
        self.safety = None
        if not args.no_safety:
            # NSFW image classifier (Falconsai/nsfw_image_detection, ViT). On our renders it scores clothed frames
            # 0.00-0.02 and near-nude frames 0.94-1.00, unlike the SD safety checker which was ~80% false positives.
            from transformers import AutoModelForImageClassification
            self.safety = AutoModelForImageClassification.from_pretrained(f"{M}/nsfw_vit", torch_dtype=torch.float16).to("cuda").eval()
        self.W, self.H = args.w, args.h
        self.steps = args.steps
        self.style = "av1main"
        self.tattoo = False
        self.frame_mode = args.frame
        self.seed = 1234
        self.lat = None
        self.emb_cache = {}
        self.framer = Framer()
        self.last_out = None
        self.last_pose = None
        self.flagged = 0
        self.times = []
        self.prof = []
        self.scores = []
        self.n = 0
        self.lock = threading.Lock()
        self._new_latents()
        print(f"[body] models loaded in {time.time() - t0:.1f}s", flush=True)

    @torch.no_grad()
    def nsfw_score(self, t):
        """P(nsfw) from the ViT classifier, computed on the GPU."""
        x = torch.nn.functional.interpolate(t.float(), size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - 0.5) / 0.5).to(torch.float16)
        return float(self.safety(pixel_values=x).logits.float().softmax(-1)[0, 1])

    def _new_latents(self):
        g = torch.Generator("cuda").manual_seed(self.seed)
        self.lat = torch.randn((1, 4, self.H // 8, self.W // 8), generator=g, device="cuda", dtype=torch.float16)

    def emb(self, prompt):
        e = self.emb_cache.get(prompt)
        if e is None:
            with torch.no_grad():
                e, _ = self.pipe.encode_prompt(prompt, "cuda", 1, False)
            self.emb_cache[prompt] = e
        return e

    def control(self, d):
        with self.lock:
            if "style" in d and d["style"] in STYLES:
                self.style = d["style"]
            if "tattoo" in d:
                self.tattoo = bool(d["tattoo"])
            if "steps" in d:
                self.steps = int(min(8, max(1, int(d["steps"]))))
            if "seed" in d:
                self.seed = int(d["seed"]) if d["seed"] is not None else secrets.randbelow(10 ** 6)
                self._new_latents()
            if "frame" in d and d["frame"] in ("auto", "full"):
                self.frame_mode = d["frame"]
                self.framer.box = None
            if "size" in d:
                w, h = d["size"]
                self.W, self.H = int(w) // 8 * 8, int(h) // 8 * 8
                self.framer.box = None
                self._new_latents()

    def summary(self):
        t = self.times[-30:]
        return {"style": self.style, "tattoo": self.tattoo, "steps": self.steps, "seed": self.seed,
                "frame": self.frame_mode, "size": [self.W, self.H],
                "render_ms": round(1000 * sum(t) / len(t), 1) if t else None, "frames": self.n,
                "safety_flags": self.flagged,
                "nsfw_score_max": round(max(self.scores[-100:]), 3) if self.scores else None}

    def compose(self, rgb):
        """Fit the render into out_w x out_h with a grey background matched to the image border."""
        OW, OH = args.out_w, args.out_h
        h, w = rgb.shape[:2]
        s = min(OW / w, OH / h)
        nw, nh = int(w * s), int(h * s)
        im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        border = np.concatenate([rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
        bg = np.median(border, axis=0).astype(np.uint8)
        out = np.empty((OH, OW, 3), np.uint8); out[:] = bg
        x0, y0 = (OW - nw) // 2, (OH - nh) // 2
        out[y0:y0 + nh, x0:x0 + nw] = im
        return out

    @torch.no_grad()
    def render(self, kp, cw, ch):
        with self.lock:
            W, H, steps, style, tattoo, mode = self.W, self.H, self.steps, self.style, self.tattoo, self.frame_mode
            lat = self.lat
        vis = kp[:, 2] >= THR
        if vis[[1, 2, 5]].sum() < 2 and vis.sum() < 4:
            return None  # no person
        box = self.framer.compute(kp, cw, ch, W, H, mode)
        k2 = Framer.apply(kp, box, W, H)
        face = vis[[0, 14, 15]].sum()
        # back view: shoulders visible but no nose/eyes; right shoulder appears on the image right
        back = face == 0 and vis[2] and vis[5]
        if back and kp[2, 0] > kp[5, 0]:
            back = True
        # framing word from what is visible
        legs = vis[[10, 13]].any() or vis[[9, 12]].all()
        framing = "full body, standing" if legs else ("thighs-up, standing" if vis[[8, 11]].any() else "waist-up")
        pose = draw_openpose(k2, W, H)
        self.last_pose = pose
        prompt = build_prompt(style, tattoo, back, framing)
        pe = self.emb(prompt)
        g = args.guidance
        extra = {"negative_prompt_embeds": self.emb(NEG)} if g > 1.0 else {}
        t0 = time.time()
        img = self.pipe(prompt_embeds=pe, image=Image.fromarray(pose), num_inference_steps=steps, guidance_scale=g, **extra,
                        latents=lat.clone(), width=W, height=H, controlnet_conditioning_scale=args.cn_scale,
                        output_type="pt").images  # (1,3,H,W) in [0,1] on GPU
        torch.cuda.synchronize(); t1 = time.time()
        if self.safety is not None:
            sc = self.nsfw_score(img.to(torch.float16))
            self.scores.append(sc); self.scores = self.scores[-300:]
            if sc > args.safety_margin:
                self.flagged += 1
                return self.last_out  # never show a flagged frame; hold the last safe one
        rgb = (img[0].clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize(); t2 = time.time()
        self.prof.append((t1 - t0, t2 - t1))
        if len(self.prof) >= 30:
            a = np.array(self.prof) * 1000; self.prof = []
            print(f"[body] pipe {a[:,0].mean():.0f} ms  safety {a[:,1].mean():.0f} ms  steps {steps} {W}x{H}", flush=True)
        self.times.append(time.time() - t0)
        self.times = self.times[-120:]
        self.n += 1
        out = self.compose(rgb)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        self.last_out = jpg.tobytes()
        return self.last_out


ENGINE = Engine()
# warm up (and trigger torch.compile) with a synthetic standing pose
_kp = np.array([[320, 80, 1], [320, 140, 1], [280, 140, 1], [270, 220, 1], [265, 300, 1], [360, 140, 1], [370, 220, 1],
                [375, 300, 1], [295, 300, 1], [295, 400, 1], [295, 470, 1], [345, 300, 1], [345, 400, 1], [345, 470, 1],
                [310, 70, 1], [330, 70, 1], [300, 75, 1], [340, 75, 1]], np.float32)
for _ in range(3):
    ENGINE.render(_kp, 640, 480)
ENGINE.times.clear(); ENGINE.n = 0
print(f"[body] warm. render {ENGINE.summary()}", flush=True)

# ---------------- web ----------------
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, JSONResponse
import uvicorn

app = FastAPI()


def check(k):
    if not secrets.compare_digest(k or "", TOKEN):
        raise HTTPException(status_code=401, detail="bad token")


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/stats")
async def stats(k: str = ""):
    check(k)
    return JSONResponse(ENGINE.summary())


@app.get("/live.jpg")
async def live_jpg(k: str = ""):
    check(k)
    if ENGINE.last_out is None:
        raise HTTPException(404)
    return Response(ENGINE.last_out, media_type="image/jpeg")


@app.get("/pose.jpg")
async def pose_jpg(k: str = ""):
    check(k)
    if ENGINE.last_pose is None:
        raise HTTPException(404)
    ok, j = cv2.imencode(".jpg", cv2.cvtColor(ENGINE.last_pose, cv2.COLOR_RGB2BGR))
    return Response(j.tobytes(), media_type="image/jpeg")


GPU_LOCK = asyncio.Lock()


@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    if not secrets.compare_digest(ws.query_params.get("k", ""), TOKEN):
        await ws.close(code=4401)
        return
    await ws.accept()
    print("[body] client connected", flush=True)
    latest = {"msg": None}
    ev = asyncio.Event()
    loop = asyncio.get_running_loop()
    alive = True
    await ws.send_text(json.dumps(ENGINE.summary()))

    async def reader():
        nonlocal alive
        try:
            while True:
                m = await ws.receive()
                if m.get("type") == "websocket.disconnect":
                    break
                if m.get("text"):
                    try:
                        ENGINE.control(json.loads(m["text"]))
                        await ws.send_text(json.dumps(ENGINE.summary()))
                    except Exception as e:
                        await ws.send_text(json.dumps({"error": str(e)}))
                elif m.get("bytes"):
                    old = latest["msg"]
                    if old is not None:  # tell the client this frame was skipped (newer one arrived)
                        ts, fid = struct.unpack("<dI", old[:12])
                        await ws.send_bytes(struct.pack("<dIf", ts, fid, -2.0))
                    latest["msg"] = m["bytes"]
                    ev.set()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            alive = False
            ev.set()

    async def worker():
        while alive:
            await ev.wait(); ev.clear()
            msg = latest["msg"]; latest["msg"] = None
            if msg is None:
                continue
            ts, fid, cw, ch = struct.unpack("<dIHH", msg[:16])
            kp = np.frombuffer(msg[16:16 + 18 * 3 * 4], np.float32).reshape(18, 3).copy()
            t0 = time.time()
            async with GPU_LOCK:
                jpg = await loop.run_in_executor(None, ENGINE.render, kp, cw, ch)
            ms = (time.time() - t0) * 1000
            try:
                if jpg is None:
                    await ws.send_bytes(struct.pack("<dIf", ts, fid, -1.0))
                else:
                    await ws.send_bytes(struct.pack("<dIf", ts, fid, ms) + jpg)
            except Exception:
                break

    await asyncio.gather(reader(), worker())
    print("[body] client disconnected", flush=True)


if __name__ == "__main__":
    print(f"[body] ready on :{args.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning", ws_max_size=2 ** 22)
