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
