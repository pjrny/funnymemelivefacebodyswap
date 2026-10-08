# Best practices

What worked best for Avatar1, and how to get the cleanest, cheapest, safest result.

## Best versions (as of Oct 8, 2026)

| Thing | Best pick | Backup / notes |
|---|---|---|
| Face engine | **LivePortrait**, human mode, expression only (`e`) | `a` (with head pose) only on clean faces |
| Face photo | **`clean_03`** slicked silver ponytail (key `1`) | `clean_06` smirk bun (`2`), `clean_01` (`3`), `clean_05` (`4`) |
| Tattoo look | **`necktattoo_front`** (key `5`) | Face-tattoo stills smear with pose; keep `e` |
| Rejected faces | `clean_02` (platinum), `clean_04` (tilted long hair) | Not head-on enough |
| Body LoRA | **Epoch 10** `avtr1_body_rv6.safetensors` | **Epoch 6** if the look feels overfit |
| Body styles | **`av1main`**, **`av1hourglass`** | `av1testing`, `av1outfits`, tattoo tag |
| Live path | **SSH tunnel** (about 20 fps, about 150 ms) | Runpod HTTPS proxy (about 10–13 fps, about 200 ms) |
| GPU | **RTX 4090**, Secure Cloud | A40 is fine for training |

## Source photos (for new faces)

- **Head-on**, eyes to the lens, face level. Tilted or 3/4 views animate badly.
- **Neutral or soft expression.** Big smiles become the baseline and look odd when you talk.
- **Sharp, well lit, face fills a good part of the frame.** No heavy filters or motion blur.
- **Hair away from the eyes** where possible; fringes flicker.
- Test new photos with `/workspace/run_lp_still.sh` before adding them to the keys.

## At the computer

| Do | Why |
|---|---|
| **Sit centered**, face in the middle of the frame | Face tracking and crop stay stable |
| **Webcam at eye height**, level | Matches the head-on source photos; less warp |
| **Soft, even front light** (window or lamp behind the screen) | Cleaner landmarks; less jitter on eyes and mouth |
| Avoid strong backlight | The face goes dark and tracking drops |
| **Calibrate with a neutral face** (`c`) | Sets the zero point; press after **every** face switch |
| Small, natural head movement | Big turns break the illusion, especially with tattoos |
| Glasses off if you can | Reflections confuse eye tracking |

## Framing

- **Now (face live):** head and shoulders, close to the laptop. Waist-up is fine.
- **Body live (next):** start **waist-up**; that's what the current camera framing gives.
- **Full body later:** stand back so head to knees (or feet) is visible, plain background,
  even light. Shoot matching full-body training images of Main and Hourglass first.

## Network

- **Wired Ethernet** best; otherwise **5 GHz Wi-Fi** close to the router.
- Use the **SSH tunnel**; the Runpod proxy is the fallback, not the default.
- Pick a pod region near you (US-IL-1 worked well from Central time).
- Close big uploads/downloads (OneDrive sync, game updates) during calls.

## Privacy

- **Secure Cloud only.** Never Community Cloud or Vast.ai for anything touching your likeness.
- **Ephemeral pods.** No network volume; terminate after every session.
- **Keep personal files local and out of git:** avatar photos, videos, datasets, LoRA weights,
  `live_config.json`, the live token.
- **Move the LoRA out of OneDrive** and keep it in an encrypted archive (7-Zip AES-256, or a
  Cryptomator vault). Upload per session; it's wiped when the pod terminates.
- **Never paste API keys or tokens into chat**, screenshots, or commits. Copy them with `scp`.
- Don't share the pod `/view?k=…` URL; the token is the only lock.
- Turn on **2FA** on Runpod, GitHub, and ElevenLabs.

## Cost hygiene

- **Always terminate** when done (Stop still bills for disk). Check the console says no pods.
- **Set auto-terminate** (`terminateAfter`) every time, plus the in-pod watchdog as backup.
- Upload photos and start the server **before** you need the camera; idle setup time costs money too.
- One training run with all styles is cheaper than many small runs. The full 5-style LoRA cost about $0.29.
- Keep an eye on the balance. This whole project went from about $10 to $7.41.

## Responsible use

- Use Avatar1 **for fun with people who know** it's an avatar.
- **Don't impersonate real people**, and don't use it to deceive anyone about who you are.
- Check the meeting host's and platform's rules. Some contexts require disclosure of synthetic media.
- Training data: only clothed images of Avatar1 that you have the rights to use.
