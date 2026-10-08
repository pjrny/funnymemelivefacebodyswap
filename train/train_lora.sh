#!/usr/bin/env bash
# Train the Avatar1 multi-style character LoRA (SD1.5 / Realistic Vision 6.0 B1) with kohya sd-scripts v0.9.1.
# Expects: /workspace/dataset/<repeats>_<style>/*.jpg+.txt, models in /workspace/models (see bootstrap_train_pod.sh)
set -euo pipefail
. /workspace/venv/bin/activate
cd /workspace/sd-scripts
OUT=/workspace/lora_out; mkdir -p $OUT
cp /workspace/train/sample_prompts.txt $OUT/
accelerate launch --num_cpu_threads_per_process 8 --mixed_precision bf16 train_network.py \
  --pretrained_model_name_or_path /workspace/models/Realistic_Vision_V6.0_NV_B1_fp16.safetensors \
  --vae /workspace/models/vae-ft-mse-840000-ema-pruned.safetensors \
  --train_data_dir /workspace/dataset --output_dir $OUT --output_name avtr1_body_rv6 --logging_dir $OUT/logs \
  --caption_extension .txt --shuffle_caption --keep_tokens 2 \
  --resolution 640,640 --enable_bucket --min_bucket_reso 320 --max_bucket_reso 1024 --bucket_reso_steps 64 --bucket_no_upscale \
  --network_module networks.lora --network_dim 32 --network_alpha 16 \
  --optimizer_type AdamW8bit --learning_rate 1e-4 --unet_lr 1e-4 --text_encoder_lr 5e-5 \
  --lr_scheduler cosine --lr_warmup_steps 100 \
  --train_batch_size 2 --max_train_epochs 10 --save_every_n_epochs 2 --save_model_as safetensors --save_precision fp16 \
  --mixed_precision bf16 --sdpa --cache_latents --persistent_data_loader_workers --max_data_loader_n_workers 4 \
  --noise_offset 0.05 --min_snr_gamma 5 --clip_skip 1 --seed 1234 \
  --sample_every_n_epochs 2 --sample_prompts $OUT/sample_prompts.txt --sample_sampler euler_a
