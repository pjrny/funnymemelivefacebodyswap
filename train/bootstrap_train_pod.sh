#!/usr/bin/env bash
# bootstrap_train_pod.sh - prepare a Runpod Secure Cloud GPU pod for Avatar1 body-LoRA training + test renders.
# Image used: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 (torch 2.4.1 cu124 preinstalled).
# Installs kohya sd-scripts v0.9.1 (pinned: newer HEAD pulls torch 2.14/diffusers 0.40 which break on this image)
# into /workspace/venv (system site packages), plus diffusers/controlnet_aux/peft for the test renders,
# and downloads the public models (no Hugging Face login needed):
#   Realistic Vision V6.0 B1 (fp16, noVAE) + sd-vae-ft-mse, ControlNet v1.1 OpenPose, LCM-LoRA SD1.5, OpenPose annotators.
# Dataset (private, never in git): copy to /workspace/dataset/<repeats>_<style>/*.jpg + *.txt with scp.
set -euo pipefail
cd /workspace
[[ -d sd-scripts ]] || git clone -q https://github.com/kohya-ss/sd-scripts
git -C sd-scripts fetch -q --depth 1 origin tag v0.9.1 && git -C sd-scripts checkout -q v0.9.1
[[ -d venv ]] || python -m venv --system-site-packages venv
. venv/bin/activate
pip -q install -r sd-scripts/requirements.txt
pip -q install "peft==0.12.0" "controlnet_aux==0.0.9" opencv-python-headless "numpy<2"
mkdir -p models && python - <<'P'
from huggingface_hub import hf_hub_download, snapshot_download
M='/workspace/models'
hf_hub_download("SG161222/Realistic_Vision_V6.0_B1_noVAE","Realistic_Vision_V6.0_NV_B1_fp16.safetensors",local_dir=M)
hf_hub_download("stabilityai/sd-vae-ft-mse-original","vae-ft-mse-840000-ema-pruned.safetensors",local_dir=M)
snapshot_download("lllyasviel/control_v11p_sd15_openpose",local_dir=f"{M}/cn_openpose",allow_patterns=["*.json","diffusion_pytorch_model.fp16.safetensors","diffusion_pytorch_model.safetensors"])
snapshot_download("latent-consistency/lcm-lora-sdv1-5",local_dir=f"{M}/lcm_lora")
for f in ["body_pose_model.pth","hand_pose_model.pth","facenet.pth"]:
    hf_hub_download("lllyasviel/Annotators",f,local_dir=f"{M}/annotators")
P
python -c "import torch,diffusers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'diffusers',diffusers.__version__)"
echo BOOTSTRAP_DONE
