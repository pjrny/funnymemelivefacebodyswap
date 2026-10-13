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
                 det_size=320, enhance_blend=0.75):
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
            return None
        return kpss[0].astype(np.float32)

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
        M = _sim(kps, ARCFACE_128)
        crop = cv2.warpAffine(frame, M, (128, 128), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (128, 128), (0, 0, 0), swapRB=True)
        pred = self.swapper.session.run(self.swapper.output_names,
                                        {self.swapper.input_names[0]: blob, self.swapper.input_names[1]: self.latent})[0]
        face = (np.clip(pred[0].transpose(1, 2, 0), 0, 1) * 255.0)[:, :, ::-1]
        t2 = time.perf_counter()
        out = self._paste(frame.copy(), face, M, self.mask128)
        if self.enhance_on and self.enhancer:
            out = self._enhance(out, kps)
        t3 = time.perf_counter()
        self.t = {"det_ms": (t1 - t0) * 1e3, "swap_ms": (t2 - t1) * 1e3, "enh_paste_ms": (t3 - t2) * 1e3}
        return out


if __name__ == "__main__":  # quick CPU test: python face_swap_engine.py SET_DIR FRAME... -o OUTDIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir"); ap.add_argument("frames", nargs="+"); ap.add_argument("-o", default=".")
    ap.add_argument("--models", default="/workspace/models/fs"); ap.add_argument("--enh", default="gfpgan")
    a = ap.parse_args()
    E = FaceSwapEngine(a.models, enhancer=a.enh)
    print("identity from", E.set_identity(sorted(glob.glob(os.path.join(a.set_dir, "*.jpg")))))
    os.makedirs(a.o, exist_ok=True)
    for f in a.frames:
        E.kps = None
        out = E.process(cv2.imread(f))
        print(f, E.t)
        if out is not None:
            cv2.imwrite(os.path.join(a.o, "swap_" + os.path.basename(f)), out)
