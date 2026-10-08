"""Test renders for the Avatar1 body LoRA: OpenPose ControlNet + LoRA (full steps) and LCM-LoRA (4 steps) with timing.
Run on the pod inside /workspace/venv. Outputs to /workspace/tests."""
import os, sys, time, json, glob, torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler, LCMScheduler
from controlnet_aux import OpenposeDetector
M='/workspace/models'; OUT='/workspace/tests'; os.makedirs(OUT,exist_ok=True)
LORA_DIR='/workspace/lora_out'; FINAL=os.environ.get('LORA','avtr1_body_rv6.safetensors')
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18) if os.path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf') else ImageFont.load_default()
def sq(im, box=None, size=512):
    if box: W,H=im.size; im=im.crop((int(box[0]*W),int(box[1]*H),int(box[2]*W),int(box[3]*H)))
    w,h=im.size; s=min(w,h); im=im.crop(((w-s)//2,0,(w-s)//2+s,s)) if w>h else im.crop((0,0,w,s))
    return im.resize((size,size),Image.LANCZOS)
POSES={'oscar_webcam':('/workspace/poses_src/oscar_frame.jpg',None),
       'front_arms_down':('/workspace/poses_src/main2.jpg',(0,0,1,0.55)),
       'hands_on_hips':('/workspace/poses_src/outfit07.jpg',(0.25,0,0.75,1)),
       'hands_clasped':('/workspace/poses_src/outfit16.jpg',(0.25,0,0.75,1))}
det=OpenposeDetector.from_pretrained(f'{M}/annotators')
pose_imgs={}
for k,(p,box) in POSES.items():
    src=sq(Image.open(p).convert('RGB'),box)
    pm=det(src, hand_and_face=True, detect_resolution=512, image_resolution=512)
    pm=pm.resize((512,512)); pm.save(f'{OUT}/pose_{k}.png'); src.save(f'{OUT}/posesrc_{k}.jpg'); pose_imgs[k]=pm
cn=ControlNetModel.from_pretrained(f'{M}/cn_openpose',torch_dtype=torch.float16)
pipe=StableDiffusionControlNetPipeline.from_single_file(f'{M}/Realistic_Vision_V6.0_NV_B1_fp16.safetensors',controlnet=cn,torch_dtype=torch.float16,load_safety_checker=False)
pipe.vae=AutoencoderKL.from_single_file(f'{M}/vae-ft-mse-840000-ema-pruned.safetensors',torch_dtype=torch.float16)
pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
DPM=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True,algorithm_type='dpmsolver++')
LCM=LCMScheduler.from_config(pipe.scheduler.config)
NEG='lowres, blurry, deformed, bad anatomy, extra fingers, nude, topless, nipples, cartoon, painting'
STY={'main':'avtr1 woman, av1main, no tattoos, white cropped tank top, long hair down',
     'hourglass':'avtr1 woman, av1hourglass, no tattoos, hourglass figure, nude-colored seamless bodysuit, hair in bun',
     'testing':'avtr1 woman, av1testing, no tattoos, slim build, hair in bun, white linen shirt',
     'outfits':'avtr1 woman, av1outfits, no tattoos, white t-shirt, slicked back ponytail',
     'tattoo':'avtr1 woman, av1tattoo, face tattoos, neck tattoos, long hair down, black t-shirt',
     'main_tattoo':'avtr1 woman, av1main, face tattoos, neck tattoos, white cropped tank top, long hair down',
     'hourglass_tattoo':'avtr1 woman, av1hourglass, face tattoos, neck tattoos, hourglass figure, nude-colored seamless bodysuit, hair in bun',
     'no_token':'avtr1 woman, no tattoos, white t-shirt, hair in bun',
     'base_no_lora':'a woman with silver hair, freckles, blue eyes, white t-shirt'}
SUF=', waist-up, looking at viewer, plain grey studio background, photo, high detail'
def load_lora(name, lcm=False):
    pipe.unload_lora_weights()
    pipe.load_lora_weights(LORA_DIR,weight_name=name,adapter_name='av')
    if lcm:
        pipe.load_lora_weights(f'{M}/lcm_lora',weight_name='pytorch_lora_weights.safetensors',adapter_name='lcm')
        pipe.set_adapters(['av','lcm'],[1.0,1.0])
def gen(prompt,pose,lcm=False,seed=7,size=512,h=None):
    g=torch.Generator('cuda').manual_seed(seed)
    if lcm:
        pipe.scheduler=LCM
        return pipe(prompt,image=pose,num_inference_steps=4,guidance_scale=1.5,negative_prompt=NEG,generator=g,width=size,height=h or size,controlnet_conditioning_scale=0.9).images[0]
    pipe.scheduler=DPM
    return pipe(prompt,image=pose,num_inference_steps=25,guidance_scale=6.0,negative_prompt=NEG,generator=g,width=size,height=h or size,controlnet_conditioning_scale=0.9).images[0]
def grid(rows,rlabels,clabels,path,T=320):
    R=len(rows); C=len(rows[0]); L=150; top=30
    sh=Image.new('RGB',(L+C*T,top+R*T),'white'); d=ImageDraw.Draw(sh)
    for j,c in enumerate(clabels): d.text((L+j*T+5,5),c,fill='black',font=font)
    for i,row in enumerate(rows):
        d.text((5,top+i*T+T//2),rlabels[i],fill='black',font=font)
        for j,im in enumerate(row): sh.paste(im.resize((T,T)),(L+j*T,top+i*T))
    sh.save(path,quality=90); print('saved',path,flush=True)
P=list(pose_imgs); timings={}
# A) style separation: same pose (front_arms_down + oscar_webcam), every style token, final LoRA, 25 steps
load_lora(FINAL)
styles=['main','hourglass','testing','outfits','tattoo','main_tattoo','hourglass_tattoo','no_token']
rows=[]; 
for pk in ['front_arms_down','oscar_webcam','hands_on_hips']:
    rows.append([pose_imgs[pk]]+[gen(STY[s]+SUF,pose_imgs[pk]) for s in styles])
grid(rows,['front','oscar cam','hands hips'],['pose']+styles,f'{OUT}/A_styles_final_25steps.jpg')
# B) 4 poses x clean/tattoo (main + hourglass + testing)
sel=['main','main_tattoo','hourglass','testing','tattoo']
rows=[[pose_imgs[pk]]+[gen(STY[s]+SUF,pose_imgs[pk],seed=11) for s in sel] for pk in P]
grid(rows,P,['pose']+sel,f'{OUT}/B_poses_clean_vs_tattoo_25steps.jpg')
# C) epoch comparison at one pose
eps=sorted(glob.glob(f'{LORA_DIR}/avtr1_body_rv6-*.safetensors'))+[f'{LORA_DIR}/{FINAL}']
rows=[];labels=[]
for e in eps:
    load_lora(os.path.basename(e)); labels.append(os.path.basename(e).replace('avtr1_body_rv6','').replace('.safetensors','') or 'final')
    rows.append([gen(STY[s]+SUF,pose_imgs['front_arms_down'],seed=21) for s in ['main','hourglass','testing','outfits','tattoo']])
grid(rows,labels,['main','hourglass','testing','outfits','tattoo'],f'{OUT}/C_epochs.jpg')
# D) base model without LoRA (reference)
pipe.unload_lora_weights()
grid([[pose_imgs['front_arms_down'],gen(STY['base_no_lora']+SUF,pose_imgs['front_arms_down'])]],['no LoRA'],['pose','RV6 base'],f'{OUT}/D_base_no_lora.jpg')
# E) LCM-LoRA 4 steps @512x512 + timing
load_lora(FINAL,lcm=True)
for _ in range(3): gen(STY['main']+SUF,pose_imgs['front_arms_down'],lcm=True)
torch.cuda.synchronize(); t=time.time(); N=10
for i in range(N): gen(STY['main']+SUF,pose_imgs[P[i%4]],lcm=True,seed=i)
torch.cuda.synchronize(); timings['lcm_4step_512_s_per_frame']=(time.time()-t)/N
rows=[[pose_imgs[pk]]+[gen(STY[s]+SUF,pose_imgs[pk],lcm=True,seed=5) for s in ['main','hourglass','testing','outfits','tattoo','main_tattoo']] for pk in P]
grid(rows,P,['pose','main','hourglass','testing','outfits','tattoo','main_tattoo'],f'{OUT}/E_lcm_4steps_512.jpg')
# full-step timing for reference
load_lora(FINAL)
torch.cuda.synchronize(); t=time.time()
for i in range(3): gen(STY['main']+SUF,pose_imgs['front_arms_down'],seed=i)
torch.cuda.synchronize(); timings['dpm_25step_512_s_per_frame']=(time.time()-t)/3
timings['gpu']=torch.cuda.get_device_name(0)
json.dump(timings,open(f'{OUT}/timings.json','w'),indent=1); print(timings)
print('RENDER_DONE')
