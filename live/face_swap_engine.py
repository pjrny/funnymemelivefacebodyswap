#!/usr/bin/env python
"""
face_swap_engine.py - "true face swap" mode for the Avatar1 live server.

Keeps Oscar's own head, hair and background and replaces only the inner face with Avatar1's
identity:  InsightFace SCRFD (det_10g) -> inswapper_128 (identity = averaged ArcFace embedding of
an Avatar1 photo set) -> GFPGAN 1.4 or GPEN-BFR-256 face restore -> feathered paste-back.

All models run through onnxruntime (CUDA on the pod, CPU on the box for dev tests).
Models dir (default /workspace/models/fs):
  inswapper_128.onnx, gfpgan_1.4.onnx, gpen_bfr_256.onnx  (+ insightface buffalo_l in ~/.insightface)
"""
import os, glob, time
import numpy as np
import cv2
import onnxruntime as ort

ARCFACE_128 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32) * (128 / 112.0)
ARCFACE_128[:, 0] += 0.0
FFHQ_UNIT = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                      [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32)

# InsightFace 2d106det indices
CONTOUR = [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0,
           24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]   # left ear -> chin -> right ear
LIPS = list(range(52, 72))
NOSE_BOTTOM = [77, 78, 79, 80, 84, 85, 83]
EYE_IDX = [35, 39, 89, 93]
CHEEK_L, CHEEK_R = 12, 28


def _sim(src, dst):
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return M.astype(np.float32)


def _soft_mask(n, pad=0.1, blur=0.12):
    """feathered box mask in crop space (inner face, keeps the hairline/jaw of the target)"""
    m = np.zeros((n, n), np.float32)
    p = int(n * pad)
    m[p:n - p, p:n - p] = 1.0
    k = max(1, int(n * blur)) | 1
    return cv2.GaussianBlur(m, (0, 0), k / 3.0)


class FaceSwapEngine:
    def __init__(self, models_dir="/workspace/models/fs", providers=None, enhancer="gpen",
                 det_size=320, enhance_blend=0.75, beard_on=True, jaw_strength=0.4):
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.providers = providers
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.det = self.app.det_model
        self.det_size = det_size
        self.swapper = get_model(os.path.join(models_dir, "inswapper_128.onnx"), providers=providers)
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enh = {}
        for name, fn, size in (("gfpgan", "gfpgan_1.4.onnx", 512), ("gpen", "gpen_bfr_256.onnx", 256)):
            p = os.path.join(models_dir, fn)
            if os.path.exists(p):
                self.enh[name] = (ort.InferenceSession(p, so, providers=providers), size)
        self.enhancer = enhancer if enhancer in self.enh else (next(iter(self.enh)) if self.enh else None)
        self.enhance_on = self.enhancer is not None
        self.enhance_blend = enhance_blend
        self.mask128 = _soft_mask(128, pad=0.08, blur=0.10)
        self.masks = {}
        self.latent = None
        self.kps = None
        self.miss = 0
        self.smooth = 0.5
        self.t = {}
        # beard cover + jaw slim (need the 106-point landmark model from buffalo_l)
        self.lmk = None
        try:
            from insightface.model_zoo import get_model as _gm
            lp = os.path.join(os.path.expanduser("~/.insightface/models/buffalo_l"), "2d106det.onnx")
            if os.path.exists(lp):
                self.lmk = _gm(lp, providers=providers)
                self.lmk.prepare(ctx_id=0)
        except Exception as e:  # pragma: no cover
            print("[swap] 106-landmark model unavailable:", e)
        self.beard_on = beard_on and self.lmk is not None
        self.jaw_strength = float(jaw_strength)
        self.l106 = None
        rng = np.random.default_rng(7)
        self.grain = cv2.GaussianBlur(rng.normal(0, 1, (256, 256)).astype(np.float32), (0, 0), 0.8) * 2.5

    # ---------- identity ----------
    def set_identity(self, paths, max_yaw=25.0):
        """average the normalised ArcFace embeddings of all (near-frontal) photos in the set"""
        embs, used = [], []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            faces = self.app.get(img, max_num=1)
            if not faces:
                continue
            f = faces[0]
            # frontal check from landmark symmetry: nose x between the eyes
            k = f.kps
            r = (k[2, 0] - k[0, 0]) / max(1e-3, (k[1, 0] - k[0, 0]))
            if not (0.2 < r < 0.8):
                continue
            embs.append(f.normed_embedding); used.append(os.path.basename(p))
        if not embs:
            raise RuntimeError("no usable face in identity set")
        e = np.mean(embs, 0)
        e = e / np.linalg.norm(e)
        latent = e.reshape(1, -1) @ self.swapper.emap
        self.latent = (latent / np.linalg.norm(latent)).astype(np.float32)
        self.identity_used = used
        return used

    # ---------- per frame ----------
    def _detect(self, frame):
        bboxes, kpss = self.det.detect(frame, input_size=(self.det_size, self.det_size), max_num=1, metric="default")
        if bboxes is None or len(bboxes) == 0:
            self.bbox = None
            return None
        self.bbox = bboxes[0][:4].astype(np.float32)
        return kpss[0].astype(np.float32)

    def _landmarks(self, frame):
        if self.lmk is None or self.bbox is None:
            return self.l106
        from insightface.app.common import Face
        f = Face(bbox=self.bbox)
        l = self.lmk.get(frame, f).astype(np.float32)
        if self.l106 is not None and np.abs(l - self.l106).max() < 25:
            l = 0.5 * self.l106 + 0.5 * l
        self.l106 = l
        return l

    # ---------- beard cover ----------
    def _lower_face_poly(self, l, neck=0.03):
        """jaw/chin contour from ear level down + upper neck, closed under the nose"""
        cont = l[CONTOUR]
        nose_y = l[NOSE_BOTTOM, 1].mean()
        lip_top = l[LIPS, 1].min()
        top = nose_y + 0.35 * (lip_top - nose_y)
        fh = l[0, 1] - l[EYE_IDX, 1].mean()          # eye-to-chin height
        low = cont[cont[:, 1] > top - 0.05 * fh]
        # push the chin part downward to cover the upper neck/under-chin stubble
        c = np.array([l[NOSE_BOTTOM, 0].mean(), top], np.float32)
        out = low.copy()
        w = np.clip((low[:, 1] - top) / max(1.0, l[0, 1] - top), 0, 1) ** 2
        out[:, 1] += w * neck * fh
        poly = np.concatenate([[[low[0, 0], top]], out, [[low[-1, 0], top]]])
        return poly.astype(np.int32), fh

    def _beard_clean(self, frame, l):
        """fill the beard area with smooth cheek-coloured skin (+ fine grain), keep the lips"""
        H, W = frame.shape[:2]
        poly, fh = self._lower_face_poly(l)
        x0, y0 = np.maximum(poly.min(0) - 8, 0); x1, y1 = np.minimum(poly.max(0) + 8, (W, H))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame, None
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        m = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillPoly(m, [poly - (x0, y0)], 255)
        lips = cv2.convexHull((l[LIPS] - (x0, y0)).astype(np.int32))
        lm = np.zeros_like(m); cv2.fillConvexPoly(lm, lips, 255)
        lm = cv2.dilate(lm, None, iterations=2)
        m[lm > 0] = 0
        # cheek skin sample: patches between the eye line and the nose bottom, either side of the nose
        ey, ny = l[EYE_IDX, 1].mean(), l[NOSE_BOTTOM, 1].mean()
        cols = []
        for a, b in ((CHEEK_L, 76), (CHEEK_R, 82)):
            cx = int(0.5 * (l[a, 0] + l[b, 0])); cy = int(0.5 * (ey + ny) + 0.15 * (ny - ey))
            r = max(2, int(0.05 * fh))
            p = frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            if p.size:
                cols.append(np.median(p.reshape(-1, 3), 0))
        if not cols:
            return frame, None
        skin = np.mean(cols, 0).astype(np.float32)
        # keep soft lighting: heavily blurred luminance ratio, clamped so dark stubble does not survive
        lum = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = max(3, int(0.12 * fh)) | 1
        shade = cv2.GaussianBlur(lum, (0, 0), k / 2.0) / max(1.0, float(skin.mean()))
        shade = np.clip(shade, 0.93, 1.04)[..., None]
        g = cv2.resize(self.grain, (x1 - x0, y1 - y0))[..., None]
        fill = skin[None, None] * shade + g
        mf = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), max(1.5, 0.03 * fh))[..., None]
        out = frame.copy()
        out[y0:y1, x0:x1] = np.clip(mf * fill + (1 - mf) * roi, 0, 255).astype(np.uint8)
        return out, (poly, fh)

    def _swap_mask(self, frame_shape, M, l, n=128):
        """crop-space mask = inner box mask UNION feathered jaw/chin polygon"""
        poly, fh = self._lower_face_poly(l, neck=0.0)
        full = np.concatenate([l[CONTOUR], poly]).astype(np.float32)
        hull = cv2.convexHull(full)
        pts = cv2.transform(hull.reshape(-1, 1, 2), M).reshape(-1, 2)
        m = np.zeros((n, n), np.float32)
        cv2.fillConvexPoly(m, pts.astype(np.int32), 1.0)
        m = cv2.erode(m, None, iterations=2)
        m = cv2.GaussianBlur(m, (0, 0), n * 0.03)
        return np.maximum(m, self.mask128)

    # ---------- jaw slim ----------
    def _jaw_slim(self, img, l, strength):
        """smooth Gaussian-RBF warp pulling the lower jaw contour toward the face midline.
        displacement is zero at the eyes/brows/cheekbones and decays outside the jaw, so hair and
        background stay put (only a thin band right next to the jaw moves)."""
        if strength <= 0.01:
            return img
        H, W = img.shape[:2]
        fh = l[0, 1] - l[EYE_IDX, 1].mean()
        if fh < 20:
            return img
        cont = l[CONTOUR]
        ny = l[NOSE_BOTTOM, 1].mean()
        midx = l[[72, 73, 74, 86, 0]][:, 0].mean()
        sel = cont[:, 1] > ny - 0.1 * fh
        src = cont[sel]
        # inward pull grows toward the jaw angle/chin, max ~9% of face height at strength 1
        wv = np.clip((src[:, 1] - (ny - 0.1 * fh)) / (0.5 * fh), 0, 1)
        dx = (midx - src[:, 0]); dx = np.sign(dx) * np.minimum(np.abs(dx), fh)
        amt = 0.12 * strength * wv * np.clip(np.abs(dx) / (0.5 * fh), 0, 1)
        dst = src.copy()
        dst[:, 0] += np.sign(dx) * amt * fh
        dst[:, 1] -= 0.02 * strength * fh * (wv ** 3)   # chin a touch shorter
        disp = src - dst   # inverse map: output at dst samples input at src
        sig = 0.16 * fh
        x0 = int(max(0, src[:, 0].min() - 3 * sig)); x1 = int(min(W, src[:, 0].max() + 3 * sig))
        y0 = int(max(0, ny - 0.4 * fh)); y1 = int(min(H, src[:, 1].max() + 3 * sig))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return img
        s = 4  # solve the field at 1/4 resolution, then upsample
        gx, gy = np.meshgrid(np.arange(x0, x1, s, dtype=np.float32), np.arange(y0, y1, s, dtype=np.float32))
        d2 = (gx[..., None] - dst[:, 0]) ** 2 + (gy[..., None] - dst[:, 1]) ** 2
        w = np.exp(-d2 / (2 * sig * sig))
        wn = w / np.maximum(w.sum(-1, keepdims=True), 1e-6)
        fall = np.clip(w.max(-1), 0, 1)                          # decays away from the jaw
        fx = (wn * disp[:, 0]).sum(-1) * fall; fy = (wn * disp[:, 1]).sum(-1) * fall
        sz = (x1 - x0, y1 - y0)
        fx = cv2.resize(fx, sz, interpolation=cv2.INTER_LINEAR); fy = cv2.resize(fy, sz, interpolation=cv2.INTER_LINEAR)
        mx, my = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
        roi = cv2.remap(img, mx + fx, my + fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img[y0:y1, x0:x1] = roi
        return img

    def _enhance(self, img, kps):
        sess, n = self.enh[self.enhancer]
        M = _sim(kps, FFHQ_UNIT * n)
        crop = cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x = crop[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        y = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
        y = ((np.clip(y.transpose(1, 2, 0), -1, 1) + 1) * 127.5)[:, :, ::-1]
        y = self.enhance_blend * y + (1 - self.enhance_blend) * crop.astype(np.float32)
        if n not in self.masks:
            self.masks[n] = _soft_mask(n, pad=0.06, blur=0.08)
        return self._paste(img, y, M, self.masks[n])

    @staticmethod
    def _paste(img, face, M, mask):
        """paste a crop back into img (BGR uint8) using a soft mask, only touching the face ROI"""
        H, W = img.shape[:2]
        n = face.shape[0]
        IM = cv2.invertAffineTransform(M)
        corners = np.array([[0, 0, 1], [n, 0, 1], [0, n, 1], [n, n, 1]], np.float32) @ IM.T
        x0, y0 = np.floor(corners.min(0)).astype(int) - 2
        x1, y1 = np.ceil(corners.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return img
        IMr = IM.copy(); IMr[:, 2] -= (x0, y0)
        sz = (x1 - x0, y1 - y0)
        f = cv2.warpAffine(face.astype(np.float32), IMr, sz, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(mask, IMr, sz, flags=cv2.INTER_LINEAR)[..., None]
        roi = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = np.clip(m * f + (1 - m) * roi, 0, 255).astype(np.uint8)
        return img

    def process(self, frame):
        """frame: BGR uint8 (webcam). Returns swapped BGR frame, or None when no face."""
        t0 = time.perf_counter()
        kps = self._detect(frame)
        t1 = time.perf_counter()
        if kps is None:
            self.miss += 1
            if self.kps is None or self.miss > 5:
                return None
            kps = self.kps
        else:
            self.miss = 0
            if self.kps is not None and np.abs(kps - self.kps).max() < 25:
                kps = self.smooth * self.kps + (1 - self.smooth) * kps
            self.kps = kps
        l = self._landmarks(frame) if (self.beard_on or self.jaw_strength > 0.01) and self.lmk is not None else None
        tb = time.perf_counter()
        base = frame
        if self.beard_on and l is not None:
            base, _ = self._beard_clean(frame, l)
        M = _sim(kps, ARCFACE_128)
        crop = cv2.warpAffine(base, M, (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        pred = self.swapper.session.run(self.swapper.output_names,
                                        {self.swapper.input_names[0]: blob, self.swapper.input_names[1]: self.latent})[0]
        face = (np.clip(pred[0].transpose(1, 2, 0), 0, 1) * 255.0)[:, :, ::-1]
        t2 = time.perf_counter()
        mask = self._swap_mask(frame.shape, M, l) if (self.beard_on and l is not None) else self.mask128
        out = self._paste(base.copy(), face, M, mask)
        if self.enhance_on and self.enhancer:
            out = self._enhance(out, kps)
        t3 = time.perf_counter()
        if l is not None and self.jaw_strength > 0.01:
            out = self._jaw_slim(out, l, self.jaw_strength)
        t4 = time.perf_counter()
        self.t = {"det_ms": (t1 - t0) * 1e3, "lmk_beard_ms": (tb - t1) * 1e3, "swap_ms": (t2 - tb) * 1e3,
                  "enh_paste_ms": (t3 - t2) * 1e3, "jaw_ms": (t4 - t3) * 1e3}
        return out


if __name__ == "__main__":  # quick CPU test: python face_swap_engine.py SET_DIR FRAME... -o OUTDIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir"); ap.add_argument("frames", nargs="+"); ap.add_argument("-o", default=".")
    ap.add_argument("--models", default="/workspace/models/fs"); ap.add_argument("--enh", default="gfpgan")
    ap.add_argument("--beard", type=int, default=1); ap.add_argument("--jaw", type=float, default=0.4)
    a = ap.parse_args()
    E = FaceSwapEngine(a.models, enhancer=a.enh, beard_on=bool(a.beard), jaw_strength=a.jaw)
    print("identity from", E.set_identity(sorted(glob.glob(os.path.join(a.set_dir, "*.jpg")))))
    os.makedirs(a.o, exist_ok=True)
    for f in a.frames:
        E.kps = None; E.l106 = None
        out = E.process(cv2.imread(f))
        print(f, E.t)
        if out is not None:
            cv2.imwrite(os.path.join(a.o, "swap_" + os.path.basename(f)), out)
