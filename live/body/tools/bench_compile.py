import os; os.environ.setdefault("OMP_NUM_THREADS","6")
import sys, time, torch, numpy as np
torch.set_num_threads(6)
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderTiny, LCMScheduler
mode=sys.argv[1]
M='/workspace/models'
cn=ControlNetModel.from_pretrained(f"{M}/cn_openpose",torch_dtype=torch.float16)
pipe=StableDiffusionControlNetPipeline.from_single_file(f"{M}/Realistic_Vision_V6.0_NV_B1_fp16.safetensors",controlnet=cn,torch_dtype=torch.float16,load_safety_checker=False)
pipe.safety_checker=None; pipe.feature_extractor=None
pipe.vae=AutoencoderTiny.from_pretrained(f"{M}/taesd",torch_dtype=torch.float16)
pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
pipe.load_lora_weights("/workspace/lora",weight_name="avtr1_body_rv6.safetensors",adapter_name="av")
pipe.load_lora_weights(f"{M}/lcm_lora",weight_name="pytorch_lora_weights.safetensors",adapter_name="lcm")
pipe.set_adapters(["av","lcm"],[0.9,1.0]); pipe.fuse_lora(adapter_names=["av","lcm"]); pipe.unload_lora_weights()
pipe.unet.to(memory_format=torch.channels_last); pipe.controlnet.to(memory_format=torch.channels_last)
if mode!="none":
    import torch._inductor.config as ic
    ic.conv_1x1_as_mm=True
    pipe.unet=torch.compile(pipe.unet,mode=mode,fullgraph=True)
    pipe.controlnet=torch.compile(pipe.controlnet,mode=mode,fullgraph=True)
    pipe.vae.decoder=torch.compile(pipe.vae.decoder,mode=mode)
pe,_=pipe.encode_prompt("avtr1 woman, av1main, photo","cuda",1,False); ne,_=pipe.encode_prompt("nude","cuda",1,False)
pose=Image.open('/workspace/live/s_pose.png') if False else Image.new("RGB",(512,768))
lat=torch.randn(1,4,96,64,device="cuda",dtype=torch.float16)
f=lambda: pipe(prompt_embeds=pe,negative_prompt_embeds=ne,image=pose,num_inference_steps=2,guidance_scale=1.5,latents=lat.clone(),width=512,height=768,output_type="pt").images
t=time.time(); f(); torch.cuda.synchronize(); print(mode,"first call",round(time.time()-t,1),"s",flush=True)
for _ in range(3): f()
torch.cuda.synchronize(); t=time.time()
for _ in range(20): f()
torch.cuda.synchronize(); print(mode,"2 steps cfg:",round((time.time()-t)/20*1000,1),"ms",flush=True)
