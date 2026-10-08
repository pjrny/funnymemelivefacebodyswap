"""Realtime-ish speed check: LoRA + LCM-LoRA fused, 4 steps, 512x512, OpenPose ControlNet, guidance 1.0 (no CFG)."""
import time, torch, json
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, LCMScheduler, AutoencoderTiny
M='/workspace/models'
cn=ControlNetModel.from_pretrained(f'{M}/cn_openpose',torch_dtype=torch.float16)
pipe=StableDiffusionControlNetPipeline.from_single_file(f'{M}/Realistic_Vision_V6.0_NV_B1_fp16.safetensors',controlnet=cn,torch_dtype=torch.float16,load_safety_checker=False)
pipe.vae=AutoencoderKL.from_single_file(f'{M}/vae-ft-mse-840000-ema-pruned.safetensors',torch_dtype=torch.float16)
pipe.scheduler=LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights('/workspace/lora_out',weight_name='avtr1_body_rv6.safetensors',adapter_name='av')
pipe.load_lora_weights(f'{M}/lcm_lora',weight_name='pytorch_lora_weights.safetensors',adapter_name='lcm')
pipe.set_adapters(['av','lcm'],[1.0,1.0]); pipe.fuse_lora(); pipe.unload_lora_weights() if False else None
pipe.to('cuda'); pipe.set_progress_bar_config(disable=True); pipe.unet.to(memory_format=torch.channels_last)
pose=Image.open('/workspace/tests/pose_front_arms_down.png')
P='avtr1 woman, av1main, no tattoos, white cropped tank top, long hair down, waist-up, looking at viewer, plain grey studio background, photo'
res={}
def bench(tag,gs,steps=4,n=10):
    for _ in range(2): pipe(P,image=pose,num_inference_steps=steps,guidance_scale=gs,width=512,height=512)
    torch.cuda.synchronize(); t=time.time()
    for i in range(n): im=pipe(P,image=pose,num_inference_steps=steps,guidance_scale=gs,width=512,height=512,generator=torch.Generator('cuda').manual_seed(i)).images[0]
    torch.cuda.synchronize(); res[tag]=round((time.time()-t)/n,3); im.save(f'/workspace/tests/F_bench_{tag}.jpg'); print(tag,res[tag],flush=True)
bench('fused_cfg1.0_4step_vaefull',1.0)
bench('fused_cfg1.0_2step_vaefull',1.0,steps=2)
try:
    pipe.vae=AutoencoderTiny.from_pretrained('madebyollin/taesd',torch_dtype=torch.float16).to('cuda')
    bench('fused_cfg1.0_4step_taesd',1.0)
except Exception as e: print('taesd fail',e)
res['gpu']=torch.cuda.get_device_name(0); json.dump(res,open('/workspace/tests/timings_fused.json','w'),indent=1); print(res); print('BENCH_DONE')
