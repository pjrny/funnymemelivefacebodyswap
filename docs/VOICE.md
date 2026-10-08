# Voice changer concepts

Three ways to give Avatar1 a matching voice on calls, from lowest latency to most realistic.
None of these are built yet. This is a design doc.

All of them end the same way: processed audio goes into a **virtual audio cable**, and Meet or
Zoom uses that cable as its **microphone**.

```
mic → [voice processing] → virtual cable (VB-Audio CABLE / VoiceMeeter) → Meet/Zoom mic
```

| Option | Latency | Realism | Cost | Best for |
|---|---|---|---|---|
| **(b) Ableton Live 11** pitch/formant | Near zero (a few ms to tens of ms) | Stylized; clearly "effected" | Already owned | Live conversation, meme vibe |
| **(c) Local RVC** realtime voice conversion | About 0.3–1 s **(estimate)** | Good: a different voice | Free software; needs GPU | Middle ground |
| **(a) ElevenLabs Voice Changer** | Seconds per phrase **(estimate)** | Highest | Per-minute credits | Pre-recorded clips, soundboard lines |

## Virtual audio routing (Windows)

| Tool | What it does |
|---|---|
| **VB-Audio Virtual Cable** | One virtual cable. Output to "CABLE Input", pick "CABLE Output" as the mic in Meet/Zoom |
| **VoiceMeeter** (Banana/Potato) | Mixer + virtual inputs/outputs, and a virtual ASIO driver. Lets you monitor yourself and mix sources |

Wear headphones so call audio doesn't feed back into the voice processor.

## (a) ElevenLabs

**What it is:** ElevenLabs **Voice Changer** (formerly Speech-to-Speech) transforms recorded
audio into a target voice (e.g. a voice you designed in ElevenLabs), keeping your timing and
emotion. API: `POST /v1/speech-to-speech/{voice_id}` and a `/stream` variant that streams the
**output** back. Recommended model: `eleven_multilingual_sts_v2`.

**Realtime limits (checked Oct 2026):**
- Input is a **recorded clip** (max 5 minutes per request). There is no documented live
  bidirectional speech-to-speech WebSocket for microphone input. The `/stream` endpoint streams
  the result, but you still send a finished audio chunk.
- Lowest-latency input format: raw 16-bit PCM, 16 kHz, mono (`file_format=pcm_s16le_16`).
- Billing is per minute of processed audio. See ElevenLabs pricing for your plan.

**Concept pipeline** (push-to-talk style):
```
mic → chunker (VAD: cut at pauses, ~1–3 s phrases) → ElevenLabs STS /stream (PCM 16 kHz)
    → playback to VB-Audio CABLE Input → Meet/Zoom mic = CABLE Output
```
- **Estimate:** 1–3 s from end of phrase to voice out (phrase length + network + generation).
  OK for one-liners and bits; awkward for back-and-forth conversation.
- Good use: a **soundboard** of pre-generated Avatar1 lines (zero live latency), or short
  push-to-talk phrases.
- Keep the API key in an environment variable or OS secret store. Never paste it in chat or commit it.

## (b) Ableton Live 11

Live sits between your mic and the virtual cable and processes audio in real time.

**Built-in devices:**
- **Shifter** (Live **11.1+**, Standard and Suite): realtime pitch shifting, frequency
  shifting, ring mod. Its pitch mode doesn't have independent **formant** control, so large
  shifts can sound chipmunk-y.
- Chain with EQ Eight, Compressor, Saturator, and a touch of Reverb for character.

**Third-party plugins** (better for convincing voice changes):
- **Soundtoys Little AlterBoy**: pitch + independent formant, robot mode. Classic for vocal character.
- **Auburn Sounds Graillon**: live pitch shifting/correction with formant preservation (free edition available).

**Routing on Windows:**
1. Live → Preferences → Audio: input = your mic/interface.
2. Output: either your interface (for monitoring) plus a send to the virtual cable, or set the
   output device to **CABLE Input** (MME/DirectX driver type) or VoiceMeeter's virtual ASIO.
3. Create an audio track: input = mic, monitor = **In**, add Shifter / Little AlterBoy.
4. Meet/Zoom mic = **CABLE Output** (or the VoiceMeeter output). Turn off the app's noise
   suppression and auto-gain if they fight the effect.
5. Buffer size 128–256 samples. With an ASIO interface, latency is typically single-digit to
   low tens of ms.

Pros: near zero delay, perfect lip sync, no cloud. Cons: it sounds *processed*, not like a new person.

## (c) Local realtime voice conversion (RVC)

A middle ground: open-source **RVC** models convert your voice to a target voice locally.
**w-okada/voice-changer** is a popular realtime client (CUDA edition for NVIDIA; DirectML/ONNX
edition for AMD/Intel GPUs).

- Latency is mostly the chunk size plus conversion time. **Estimate:** about 0.3–1 s on a
  decent NVIDIA GPU.
- **This laptop has no NVIDIA GPU** (Iris Xe). Expect higher latency on DirectML/CPU. Options:
  run RVC on a desktop with an NVIDIA card, or on the Runpod pod and stream audio both ways
  (adds network delay and a second stream to manage).
- Output → VB-Audio CABLE → Meet/Zoom mic, same as above.

## Lip sync with the avatar

The face path adds about 150 ms of video delay (SSH tunnel). Voice delay depends on the option:

| Option | Video vs voice | Fix |
|---|---|---|
| Ableton | Video lags voice by about 150 ms | Optional: OBS audio **Sync Offset** on the mic (+150 ms) when routing audio through OBS |
| RVC | Voice may lag video by a few hundred ms | Add an OBS **Render Delay** filter to the Avatar1 Browser source (OBS caps it at 500 ms) |
| ElevenLabs | Voice lags by seconds | Lip sync isn't realistic live; use it for clips or turn video expressions down during lines |

To use OBS for sync, run the client with `-- --no-vcam`, add the Browser source
`http://127.0.0.1:8766/`, apply the delay there, and start OBS's own Virtual Camera.

## Consent and rights

- Only use voices you have rights to: **your own voice**, or a voice you **designed** in
  ElevenLabs. Don't clone real people without permission.
- Same rule as the face: fun with people who know it's an avatar, never deception.
