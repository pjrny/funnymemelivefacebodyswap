import time, torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderTiny, LCMScheduler
M='/workspace/models'
cn=ControlNetModel.from_pretrained(f"{M}/cn_openpose",torch_dtype=torch.float16)
pipe=StableDiffusionControlNetPipeline.from_single_file(f"{M}/Realistic_Vision_V6.0_NV_B1_fp16.safetensors",controlnet=cn,torch_dtype=torch.float16,load_safety_checker=False)
pipe.vae=AutoencoderTiny.from_pretrained(f"{M}/taesd",torch_dtype=torch.float16)
pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
for n,m in [("unet",pipe.unet),("cn",pipe.controlnet),("vae",pipe.vae),("te",pipe.text_encoder)]: print(n, next(m.parameters()).device, next(m.parameters()).dtype)
pe,_=pipe.encode_prompt("avtr1 woman, photo","cuda",1,False)
pose=Image.new("RGB",(512,768))
def T(f,n=10):
    f(); torch.cuda.synchronize(); t=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-t)/n*1000
lat=torch.randn(1,4,96,64,device="cuda",dtype=torch.float16)
print("pipe 1step latent", T(lambda: pipe(prompt_embeds=pe,image=pose,num_inference_steps=1,guidance_scale=1.0,latents=lat.clone(),width=512,height=768,output_type="latent")))
print("pipe 1step np", T(lambda: pipe(prompt_embeds=pe,image=pose,num_inference_steps=1,guidance_scale=1.0,latents=lat.clone(),width=512,height=768,output_type="np")))
print("pipe 2step np", T(lambda: pipe(prompt_embeds=pe,image=pose,num_inference_steps=2,guidance_scale=1.0,latents=lat.clone(),width=512,height=768,output_type="np")))
t=torch.tensor([999],device="cuda"); cimg=torch.zeros(1,3,768,512,device="cuda",dtype=torch.float16)
with torch.no_grad():
    def raw():
        d,m=pipe.controlnet(lat,t,encoder_hidden_states=pe,controlnet_cond=cimg,return_dict=False)
        pipe.unet(lat,t,encoder_hidden_states=pe,down_block_additional_residuals=d,mid_block_additional_residual=m,return_dict=False)
    print("raw unet+cn", T(raw))
    print("vae decode", T(lambda: pipe.vae.decode(lat)))
