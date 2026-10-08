#!/usr/bin/env bash
# bootstrap_body_pod.sh - prepare a Runpod pod (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04) for the
# Avatar1 live full-body server. Public models only (no HF login). The body LoRA is private: scp it to /workspace/lora/.
set -euo pipefail
mkdir -p /workspace/models /workspace/lora /workspace/live /workspace/logs
cd /workspace
pip -q install "diffusers==0.31.0" "transformers==4.44.2" "peft==0.12.0" "accelerate==0.34.2" "huggingface_hub<0.26" \
    fastapi "uvicorn[standard]" opencv-python-headless "numpy<2" safetensors 2>&1 | grep -v "notice" || true
python - <<'P'
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, snapshot_download
M='/workspace/models'
jobs=[lambda: hf_hub_download("SG161222/Realistic_Vision_V6.0_B1_noVAE","Realistic_Vision_V6.0_NV_B1_fp16.safetensors",local_dir=M),
      lambda: hf_hub_download("stabilityai/sd-vae-ft-mse-original","vae-ft-mse-840000-ema-pruned.safetensors",local_dir=M),
      lambda: snapshot_download("lllyasviel/control_v11p_sd15_openpose",local_dir=f"{M}/cn_openpose",allow_patterns=["*.json","diffusion_pytorch_model.fp16.safetensors"]),
      lambda: snapshot_download("latent-consistency/lcm-lora-sdv1-5",local_dir=f"{M}/lcm_lora"),
      lambda: snapshot_download("madebyollin/taesd",local_dir=f"{M}/taesd",allow_patterns=["*.json","diffusion_pytorch_model.safetensors"]),
      lambda: snapshot_download("CompVis/stable-diffusion-safety-checker",local_dir=f"{M}/safety_checker",allow_patterns=["*.json","model.safetensors"]),
      lambda: snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",allow_patterns=["*.json","*.txt"])]
with ThreadPoolExecutor(7) as ex:
    for r in ex.map(lambda f: f(), jobs): print(r, flush=True)
P
# cn_openpose ships fp16 weights under the variant name; diffusers loads variant='fp16' or we symlink
cd /workspace/models/cn_openpose && [ -f diffusion_pytorch_model.safetensors ] || ln -s diffusion_pytorch_model.fp16.safetensors diffusion_pytorch_model.safetensors
python -c "import torch,diffusers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'diffusers',diffusers.__version__)"
echo BOOTSTRAP_DONE
