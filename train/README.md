# Avatar1 body LoRA (SD1.5) – training + test renders

One character LoRA for the live **webcam pose → OpenPose ControlNet → SD1.5 + LoRA** path.
Images and weights are private and are **never committed** (scripts/configs only).

## Tokens
Always start the prompt with the identity trigger `avtr1 woman`, then one body-style token:

| token | source folder (`Avatar1\body\...`) | look |
|---|---|---|
| `av1main` | `Main` | white cropped tank, light grey leggings, long silver hair down |
| `av1hourglass` | `Hourglass` | hourglass figure, nude-colored seamless bodysuit / black lace, bun |
| `av1testing` | `Testing` | slim build, white linen shirt + trousers, bun |
| `av1outfits` | `Multiple Oufits` | white tee / grey tank / blue shirt / black bodysuit, slicked ponytail |
| `av1tattoo` | earlier face set (face + neck tattoo stills) | tattooed look |

Tattoos are a separate switch on any style: add `face tattoos, neck tattoos` or `no tattoos`.
Suggested negative: `lowres, blurry, deformed, bad anatomy, nude, topless, nipples`.

## Run
1. Create a Runpod **Secure Cloud** pod (image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, 1x 24–48 GB GPU, 22/tcp) with `terminateAfter` set.
2. `bash train/bootstrap_train_pod.sh` (kohya sd-scripts **v0.9.1** pinned + public models, no HF login).
3. Build the dataset on your own machine: `python train/build_dataset.py` → `dataset/<repeats>_<style>/*.jpg|.txt`, then `scp -r dataset pod:/workspace/`.
4. `bash train/train_lora.sh` → `/workspace/lora_out/avtr1_body_rv6*.safetensors` (checkpoint every 2 epochs).
5. `python train/render_tests.py` (style grid, pose × clean/tattoo, epoch comparison, LCM 4-step grid + timing) and `python train/bench_lcm.py` (fused-LoRA speed).
6. Copy the LoRA off the pod, then terminate the pod.

## Settings used (run 1, Oct 8 2026)
Base Realistic Vision V6.0 B1 (fp16, noVAE) + sd-vae-ft-mse · LoRA rank 32 / alpha 16 · 640px with buckets 320–1024, no upscaling ·
AdamW8bit, unet 1e-4 / TE 5e-5, cosine, 100 warmup · batch 2 × 10 epochs = 2020 steps · bf16, SDPA, noise offset 0.05, min-SNR 5 ·
captions with `--shuffle_caption --keep_tokens 2`. Repeats: main 2, hourglass 3, testing 6, outfits 2, tattoo 3, clean face refs 3.

Step-by-step training + privacy notes: [docs/USAGE.md §10](../docs/USAGE.md#10-train-the-body-lora). Body realtime roadmap: [docs/OPTIMIZATION.md](../docs/OPTIMIZATION.md).
