# Session log and feedback

## 2026-10-13: face live v2 live test (RTX 4090, about 30 min, about $0.50)

**What we tested:** angle-aware LivePortrait sets (keys `1` and `2`), true face swap (`m`), webcam corner inset (`p`).

**Oscar's feedback:**
- The angle sets and the overall v2 setup look good.
- The corner inset of the real webcam is wanted and works.
- In the true face swap mode (`m`, which keeps his own hair and background):
  - The beard still shows through or blends poorly. It needs much better beard removal or cover.
  - The jawline is too wide and masculine. It should be slimmed slightly toward Avatar1's jaw.

**Observed issues:**
- Set 2 sometimes jumps to a full-profile photo on small head turns. Re-calibrating with `c` helps, and the switching threshold may need tuning.
- `start_live.ps1` prints fps and latency to its window but doesn't write `live_session.log`.

**Next steps:**
- Beard: run a lower-face cleanup before the swap. Options are a skin-texture inpaint of the beard region on the driver frame, a larger swap mask that covers the chin and jaw, or a higher-resolution swapper (for example 256 px or a sharpened variant) so stubble doesn't bleed through.
- Jaw: apply a light liquify or warp of the lower-face landmarks toward Avatar1's jaw proportions before blending, with an adjustable strength key.
- Tune set-2 angle switching and log laptop fps to file.

## 2026-10-13: beard cover and jaw slim for face-swap mode (branch `face-v2-beard-jaw`)

**What changed** (`live/face_swap_engine.py`, server and laptop keys):
- Beard cover (`b`, default on): InsightFace 106-point landmarks find the lower face. The beard and mustache area (lips kept) is filled with Oscar's own cheek skin tone, soft shading and fine grain *before* inswapper. The swap is then blended over a landmark-shaped, feathered chin/jaw mask instead of only the inner-face box. No stubble shows through in the CPU tests.
- Jaw slim (`+`/`-`, 0 to 1, default 0.4): a smooth Gaussian-RBF warp pulls the lower jaw and chin contour toward the midline (up to about 12% of face height at 1.0). The effect fades out within about a sixth of the face height, so hair, headphones and background stay put.
- Env defaults for the pod: `FS_BEARD=1`, `FS_JAW=0.4`.

**Speed** (RTX 4090 bench, GPEN on, 150 frames): 27.7 ms per frame before and 35.3 ms after, so about 7.6 ms added (landmarks plus fill about 2.2 ms, mask about 2 ms, jaw warp about 2.9 ms). That is within the 20 fps live budget.

**Not done:** there is no readily available 256-px inswapper-class model, so the pipeline stays on inswapper_128 + GPEN.

**To check in the next live session:** how the fill looks under different lighting. The skin fill is smooth, so the lower face looks slightly airbrushed. Also check the neck and under-chin when not wearing a hood.

## 2026-10-13: live test of beard cover + jaw slim, revised to v3

**Oscar's feedback (live session):**
- The beard skin-fill was too much: obvious discoloration when looking up or to the side, and with hair down the re-paint covered his hair.
- Go back to how the swap replaced the face before, with only basic smoothing.
- Jaw: keep slimming at max (1.0, smallest jaw) as the default.

**Changes (v3):**
- Removed the cheek-colour fill and the widened chin/neck blend mask; the swap uses the original v2 inner-face mask and blending again.
- `b` now toggles a light bilateral (edge-preserving) smoothing of stubble, limited to the original face-mask footprint intersected with the landmark lower-face polygon, lips excluded. Hair outside the face mask is never touched.
- Default jaw 1.0 (`FS_JAW=1.0`); saved settings (`swap_settings.json`) still win.
- CPU comparison on front/turned/looking-up frames: webcam / old v2 / heavy fill / v3 (`beard_v3_compare.jpg`, not in git). Cost about 1 ms extra vs. old v2 on top of the jaw warp.
