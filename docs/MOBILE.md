# Mobile: using Avatar1 on phone calls

Honest map of what works today, what doesn't, and the practical workarounds. Mobile Zoom,
Google Meet, WhatsApp, and FaceTime **do not** expose a virtual-camera picker the way desktop
apps do. Nothing here requires jailbreak or root.

## Easiest path (recommended)

**Run the meeting on the laptop.** Keep the face live pipeline as it is
(webcam → Runpod → OBS Virtual Camera → Meet/Zoom desktop). Use the phone only for chat,
notes, or a second screen. Lowest latency, fewest moving parts, already tested in Google Meet.

## Phone camera as the laptop webcam

If you want to sit farther from the laptop (or use a better camera), pipe the phone camera
into the existing Windows pipeline:

| Tool | Platform | Notes |
|---|---|---|
| **Windows 11 Phone Link** ("Use as connected camera") | Android → Windows 11 (built-in) | Settings → Bluetooth & devices → Mobile devices → Manage devices → turn on *Use as connected camera*; then point `start_live.ps1 -Cam <n>` at it |
| **iVCam** | iOS / Android → Windows | App on phone + desktop driver; shows up as a webcam |
| **DroidCam** | Android (and iOS companion) → Windows | Free tier available; USB is more reliable than Wi-Fi |
| **Camo** | iOS / Android → Windows / Mac | Higher quality, paid; excellent for Meet/Zoom |

Once Windows sees the phone as a camera, Avatar1 works exactly as on the built-in webcam.
Still: the **meeting itself** runs on the laptop.

## Joining the meeting from the phone

Stock **Zoom** and **Google Meet mobile apps cannot select a virtual camera**. That is a
platform limit, not something this project can fix. Options:

### (a) Laptop hosts the call; phone is audio-only companion
Join Meet/Zoom from the laptop with Avatar1 as the camera. On the phone, mute video and use
the phone mic only if you prefer (or stay fully on the laptop). Simplest and most reliable.

### (b) Cloud viewer + phone screen share
1. Pod (or laptop) serves a small password-protected page showing the live avatar
   (WebRTC or the existing MJPEG `/view`). Prefer a private path (SSH tunnel, Tailscale, or
   a short-lived token URL: never leave it public).
2. On the phone, open that page in the browser.
3. Join Zoom/Meet from the phone → **Share screen** → share the browser tab showing Avatar1.
4. Use the phone's microphone for audio.

Works on both Zoom and Google Meet mobile today. Trade-offs: screen-share frame rate is lower
than a native camera (often about 15–25 fps), and battery/heat go up. Fine for casual calls;
use (a) for anything important. This is the path the original scope doc sketched.

### (c) WhatsApp and FaceTime

| App | Can it take Avatar1? | How |
|---|---|---|
| **WhatsApp mobile** (iOS/Android) | **No** virtual camera on stock OS | No supported path without jailbreak/root. Don't. |
| **WhatsApp Desktop (Windows)** | **Not** via OBS Virtual Camera directly | WhatsApp Desktop usually ignores OBS Virtual Camera. Common workaround: OBS **DroidCam Virtual Output** plugin + its Windows kernel driver, then pick **DroidCam Video** in WhatsApp Desktop → Settings → Video. Test before relying on it; WhatsApp updates can break detection. WhatsApp Web in the browser is even more limited. |
| **FaceTime (iPhone/iPad)** | **No** virtual camera | Continuity Camera only feeds the iPhone *into* a Mac, not the other way. |
| **FaceTime on a Mac** | **Mostly no** for OBS Virtual Camera | Apple's system apps (FaceTime, Photo Booth) block third-party virtual cameras under library validation. Continuity Camera (iPhone as Mac webcam) works, but that is the phone camera, not Avatar1. Unsupported SIP-weakening tricks exist; we don't recommend them. |

Practical WhatsApp path: run Avatar1 on the Windows laptop into OBS, enable DroidCam Virtual
Output, call from **WhatsApp Desktop**. Keep the phone nearby for notifications only.

## Suggested setups by situation

| Situation | Do this |
|---|---|
| Important Meet/Zoom | Laptop hosts, Avatar1 via OBS Virtual Camera |
| Casual Meet/Zoom from the couch | Phone camera → laptop (Camo/iVCam/DroidCam) → Avatar1 → laptop hosts the call |
| Truly phone-only Meet/Zoom | Option (b): pod viewer + phone screen share |
| WhatsApp video | WhatsApp Desktop + DroidCam Virtual Output (test first) |
| FaceTime | Not supported for Avatar1 without unsupported Mac hacks; use Meet/Zoom instead |

## Privacy notes for mobile

- Prefer Tailscale or an SSH tunnel over a public `/view?k=…` link if you reuse the viewer often.
- Short-lived tokens, HTTPS, and terminate-the-pod after the call still apply.
- Screen share sends whatever is on that phone screen. Close notifications and other tabs first.
