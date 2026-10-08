import sys, time, cv2, numpy as np
img = cv2.imread(sys.argv[1]); img = cv2.resize(img, (640, int(img.shape[0]*640/img.shape[1])))
from rtmlib import Body, BodyWithFeet, Wholebody, RTMO, draw_skeleton
for name, ctor in [("body_lightweight", lambda: Body(mode="lightweight", to_openpose=True, backend="onnxruntime", device="cpu")),
                   ("rtmo_s", lambda: RTMO("https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip", to_openpose=True, backend="onnxruntime", device="cpu")),
                   ("wholebody_lightweight", lambda: Wholebody(mode="lightweight", to_openpose=True, backend="onnxruntime", device="cpu"))]:
    try:
        m = ctor()
        for _ in range(3): k, s = m(img)
        t = time.time(); n = 15
        for _ in range(n): k, s = m(img)
        dt = (time.time()-t)/n
        print(name, f"{dt*1000:.0f} ms  {1/dt:.1f} fps", k.shape, flush=True)
        c = draw_skeleton(np.zeros_like(img), k, s, openpose_skeleton=True, kpt_thr=0.3)
        cv2.imwrite(f"pose_{name}.png", c)
    except Exception as e:
        print(name, "FAIL", repr(e)[:300], flush=True)
