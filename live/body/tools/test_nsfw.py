import glob, torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from huggingface_hub import snapshot_download
p=snapshot_download("Falconsai/nsfw_image_detection", local_dir="/workspace/models/nsfw_vit", allow_patterns=["*.json","*.safetensors"])
m=AutoModelForImageClassification.from_pretrained(p, torch_dtype=torch.float16).cuda().eval(); pr=AutoImageProcessor.from_pretrained(p)
print(m.config.id2label, pr.image_mean, pr.image_std, pr.size)
for f in sorted(glob.glob('/workspace/live/body_*.jpg'))+sorted(glob.glob('/workspace/live/cyc_*_0.jpg'))+sorted(glob.glob('/workspace/live/bench_*.jpg')):
    im=Image.open(f).convert('RGB'); im=im.crop((240,0,720,540)) if im.size==(960,540) else im
    x=pr(images=im,return_tensors='pt').pixel_values.cuda().half()
    with torch.no_grad(): pnsfw=m(x).logits.float().softmax(-1)[0, 1].item()
    print(f.split('/')[-1], round(pnsfw,3))
