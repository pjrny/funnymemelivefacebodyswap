import sys, time, cv2, numpy as np
img = cv2.imread(sys.argv[1]); img = cv2.resize(img, (640, int(img.shape[0]*640/img.shape[1])))
from rtmlib import Body, PoseTracker
import onnxruntime as ort
print("ort providers", ort.get_available_providers())
for mode in ["lightweight", "balanced"]:
    tr = PoseTracker(Body, det_frequency=15, to_openpose=True, mode=mode, backend="onnxruntime", device="cpu", tracking=False)
    for _ in range(3): tr(img)
    t = time.time(); n = 30
    for _ in range(n): k, s = tr(img)
    dt = (time.time()-t)/n; print("tracker", mode, f"{dt*1000:.0f} ms {1/dt:.1f} fps", flush=True)
