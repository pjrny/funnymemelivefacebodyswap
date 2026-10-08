"""Build the Avatar1 body LoRA dataset (kohya folder layout) from the curated sources.
One shared identity trigger 'avtr1 woman' + one style token per body folder."""
import os, shutil, json
from PIL import Image, ImageOps
RAW='/workspace/fs/body/raw'; VF='/workspace/fs/body/work/vframes'; PN='/workspace/fs/body/work/panels'
AV1='/workspace/av1'; CLEAN='/workspace/fs/pod_backup/clean'
OUT='/workspace/fs/body/dataset'
shutil.rmtree(OUT, ignore_errors=True)
TRIG='avtr1 woman'
REPEATS={'main':2,'hourglass':3,'testing':6,'outfits':2,'tattoo':3,'face':3}
TOK={'main':'av1main','hourglass':'av1hourglass','testing':'av1testing','outfits':'av1outfits','tattoo':'av1tattoo','face':None}
manifest=[]
def put(style, img, caption, name, crop=None, flip=False):
    d=f"{OUT}/{REPEATS[style]}_{style}"; os.makedirs(d,exist_ok=True)
    im=Image.open(img).convert('RGB') if isinstance(img,str) else img
    W,H=im.size
    if crop: im=im.crop(tuple(int(v*(W if i%2==0 else H)) for i,v in enumerate(crop)))
    if flip: im=ImageOps.mirror(im)
    # cap long side at 1536 to keep upload small
    im.thumbnail((1536,1536), Image.LANCZOS)
    tok=TOK[style]; parts=[TRIG]+([tok] if tok else [])+[caption]
    cap=', '.join(parts)
    im.save(f"{d}/{name}.jpg", quality=95); open(f"{d}/{name}.txt",'w').write(cap)
    manifest.append({'style':style,'file':f"{REPEATS[style]}_{style}/{name}.jpg",'src':img if isinstance(img,str) else '', 'crop':crop,'flip':flip,'caption':cap, 'size':im.size})
WU=(0,0,1,0.55)   # waist-up crop of a full-body frame
BG='plain grey studio background'
# ---------------- MAIN ----------------
m=f'{RAW}/Main'
O1='no tattoos, white cropped tank top, light grey leggings'
put('main',f'{m}/image (1).jpg',f'{O1}, long hair down, full body, standing, front view, {BG}','m_img1')
put('main',f'{m}/image (1).jpg',f'{O1}, long hair down, waist-up, front view, {BG}','m_img1_wu',crop=WU)
put('main',f'{m}/image (2).jpg',f'{O1}, long hair down, full body, standing, front view, {BG}','m_img2')
put('main',f'{m}/image (2).jpg',f'{O1}, long hair down, waist-up, front view, {BG}','m_img2_wu',crop=WU)
put('main',f'{m}/image (3).jpg',f'{O1}, hair in bun, full body, standing, back view, looking over shoulder, {BG}','m_img3')
put('main',f'{m}/image (4).jpg',f'{O1}, long hair down, full body, standing, front view, {BG}','m_img4')
put('main',f'{m}/image (4).jpg',f'{O1}, long hair down, waist-up, front view, {BG}','m_img4_wu',crop=WU)
g13=f'{VF}/Main_grok_video_cb5a7ab3_002d_429c_9ef3_d955777bd998__13_'
O2='no tattoos, white t-shirt, white shorts, bright white room'
put('main',f'{g13}/f_001.jpg',f'{O2}, long hair down, full body, standing, front view','m_v13_01')
put('main',f'{g13}/f_003.jpg',f'{O2}, long hair down, full body, arms raised','m_v13_03')
put('main',f'{g13}/f_004.jpg',f'{O2}, long hair down, waist-up, holding phone, taking selfie','m_v13_04')
put('main',f'{g13}/f_006.jpg',f'{O2}, long hair down, close-up, upper body, holding phone, smiling','m_v13_06')
put('main',f'{g13}/f_008.jpg',f'{O2}, long hair down, upper body, holding phone, laughing','m_v13_08')
put('main',f'{g13}/f_010.jpg',f'{O2}, full body, standing, front view','m_v13_10')
for vid,picks in [('14',[1,4,7,10,13,16]),('15',[1,4,7,10,13,16]),('9',[1,4,7,10])]:
    d=f'{VF}/Main_grok_video_cb5a7ab3_002d_429c_9ef3_d955777bd998__{vid}_'
    for j,p in enumerate(picks):
        put('main',f'{d}/f_{p:03d}.jpg',f'{O1}, long hair down, full body, standing, posing, {BG}',f'm_v{vid}_{p:02d}')
        if j%2==0: put('main',f'{d}/f_{p:03d}.jpg',f'{O1}, long hair down, waist-up, posing, {BG}',f'm_v{vid}_{p:02d}_wu',crop=(0,0,1,0.5))
hg=f'{VF}/Main_higgsfield_bf8a1184_6d45_4f25_9009_70fc6ead3adc__1_'
for p in [1,5,9,13,17,20]:
    put('main',f'{hg}/f_{p:03d}.jpg','no tattoos, sheer white top, messy bun, close-up portrait, head and shoulders, soft window light',f'm_hf_{p:02d}')
# ---------------- HOURGLASS ----------------
h=f'{RAW}/Hourglass'
NB='no tattoos, hourglass figure, nude-colored seamless bodysuit, hair in bun'
put('hourglass',f'{h}/image (5).jpg',f'{NB}, full body, standing, side view, {BG}','h5')
put('hourglass',f'{h}/image (5).jpg',f'{NB}, waist-up, side view, {BG}','h5_wu',crop=(0,0,1,0.5))
put('hourglass',f'{h}/image (6).jpg',f'{NB}, full body, standing, front view, arms at sides, {BG}','h6')
put('hourglass',f'{h}/image (6).jpg',f'{NB}, waist-up, front view, {BG}','h6_wu',crop=(0,0,1,0.5))
put('hourglass',f'{h}/image (8).jpg',f'{NB}, waist-up, front view, looking at viewer, {BG}','h8')
put('hourglass',f'{h}/image (9).jpg',f'{NB}, waist-up, side profile view, {BG}','h9')
put('hourglass',f'{h}/image (10).jpg',f'{NB}, full body, standing, back view, {BG}','h10')
put('hourglass',f'{h}/image (11).jpg',f'{NB}, waist-up, front view, looking at viewer, {BG}','h11')
put('hourglass',f'{h}/image (16).jpg','no tattoos, hourglass figure, black lace bodysuit, hair in bun, thighs-up, back view, looking over shoulder, '+BG,'h16r',crop=(0.5,0,1,1))
PANELS={
 'hg01_r0c0':'nude-colored seamless bodysuit, standing, three-quarter view',
 'hg01_r0c3':'nude-colored seamless bodysuit, standing, hands behind head',
 'hg01_r2c0':'nude-colored seamless bodysuit, standing, arms crossed',
 'hg02_r0c0':'nude-colored seamless bodysuit, standing, front view',
 'hg02_r0c1':'nude-colored seamless bodysuit, standing, three-quarter view, hand on hip',
 'hg02_r0c3':'nude-colored seamless bodysuit, standing, one hand on head',
 'hg02_r1c0':'nude-colored seamless bodysuit, standing, three-quarter view',
 'hg02_r1c2':'nude-colored seamless bodysuit, standing, arms crossed',
 'hg03_r0c0':'nude-colored seamless bodysuit, standing, front view',
 'hg03_r1c2':'nude-colored seamless bodysuit, standing, one arm extended',
 'hg03_r1c3':'nude-colored seamless bodysuit, standing, hand on chin',
 'hg04_r0c0':'nude-colored seamless bodysuit, standing, front view',
 'hg04_r1c1':'nude-colored seamless bodysuit, standing, one arm raised',
 'hg04_r1c3':'nude-colored seamless bodysuit, standing, hands on hips',
 'hg15_r0c0':'black lace lingerie set, standing, front view',
 'hg15_r0c3':'black lace lingerie set, standing, hands behind head',
 'hg15_r1c2':'black lace lingerie set, standing, arms crossed',
 'hg17_r0c0':'black lace bodysuit, standing, front view',
 'hg17_r1c0':'black lace bodysuit, standing, three-quarter view',
 'hg17_r1c1':'black lace bodysuit, standing, one arm raised',
 'hg18_r0c0':'black lace bodysuit, standing, three-quarter view',
 'hg18_r1c0':'black lace bodysuit, standing, hands behind head',
}
for k,v in PANELS.items():
    put('hourglass',f'{PN}/{k}.jpg',f'no tattoos, hourglass figure, {v}, hair in bun, thighs-up, {BG}',f'hp_{k}')
# ---------------- TESTING (crops avoid the bare chest) ----------------
t=f'{RAW}/Testing'
TS='no tattoos, slim build, hair in bun'
for flip in (False,True):
    s='_f' if flip else ''
    put('testing',f'{t}/image (2).jpg',f'{TS}, open white linen shirt, head and shoulders, front view, looking at viewer, {BG}','t2_head'+s,crop=(0.22,0,0.78,0.255),flip=flip)
    put('testing',f'{t}/image (2).jpg',f'{TS}, white linen shirt, white linen trousers, barefoot, lower body, standing, front view, {BG}','t2_low'+s,crop=(0.1,0.43,0.9,1),flip=flip)
    put('testing',f'{t}/image (3).jpg',f'{TS}, open white linen shirt, head and shoulders, side view, looking at viewer, {BG}','t3_head'+s,crop=(0.25,0,0.75,0.24),flip=flip)
    put('testing',f'{t}/image (3).jpg',f'{TS}, white linen shirt, white linen trousers, barefoot, lower body, standing, side view, {BG}','t3_low'+s,crop=(0.2,0.45,0.8,1),flip=flip)
    put('testing',f'{t}/image (4).jpg',f'{TS}, white linen shirt, white linen trousers, barefoot, full body, standing, back view, looking over shoulder, {BG}','t4'+s,flip=flip)
    put('testing',f'{t}/image (4).jpg',f'{TS}, white linen shirt, waist-up, back view, looking over shoulder, {BG}','t4_wu'+s,crop=(0,0,1,0.5),flip=flip)
    put('testing',f'{t}/image (1).jpg',f'{TS}, head and shoulders, front view, looking at viewer, {BG}','t1_head'+s,crop=(0.25,0,0.75,0.2),flip=flip)
# ---------------- OUTFITS ----------------
mo=sorted(os.listdir(f'{RAW}/Multiple Oufits'))
OUTF={0:'black long-sleeve patterned bodysuit, t-pose, arms out',1:'black long-sleeve patterned bodysuit, standing, front view',
2:'light blue button-up shirt, black trousers, sitting on floor, leaning back',3:'white t-shirt, upper body, side profile view',
7:'white t-shirt, black leggings, waist-up, hands on hips, three-quarter view',8:'white t-shirt, black leggings, sitting on chair, side view',
9:'white t-shirt, waist-up, typing on keyboard, front view',11:'white t-shirt, upper body, sitting, hands clasped, side view',
12:'white t-shirt, waist-up, arms raised overhead, stretching',13:'white t-shirt, waist-up, sitting, hands on knees, front view',
16:'white t-shirt, waist-up, hands clasped at chest, front view',18:'white t-shirt, black leggings, full body, standing, arms crossed',
19:'grey tank top, black leggings, sitting cross-legged',20:'grey tank top, black leggings, full body, t-pose, arms out',
21:'light blue button-up shirt, black trousers, full body, back view',23:'white t-shirt, black leggings, full body, t-pose, arms out',
26:'white t-shirt, black leggings, full body, pointing at viewer',27:'white t-shirt, black leggings, full body, palms up, shrugging',
28:'white t-shirt, black leggings, full body, walking',29:'grey tank top, black leggings, full body, hand on hip, three-quarter view',
30:'black long-sleeve patterned bodysuit, sitting on chair, front view',32:'black long-sleeve patterned bodysuit, kneeling lunge, pointing',
34:'grey tank top, black leggings, full body, arms raised overhead',35:'grey tank top, black leggings, full body, side stretch',
36:'grey tank top, black leggings, waist-up, hands on hips, front view',38:'grey tank top, black leggings, sitting on stool, front view',
40:'light blue button-up shirt, black trousers, kneeling lunge, arm extended',44:'black long-sleeve patterned bodysuit, thighs-up, hands on hips, front view',
45:'grey tank top, black leggings, standing, side view',47:'light blue button-up shirt, black trousers, side stretch, arm raised',
50:'light blue button-up shirt, black trousers, full body, arms crossed',52:'grey tank top, black leggings, standing, back view, looking over shoulder',
53:'light blue button-up shirt, black trousers, full body, walking, pointing'}
for i,c in OUTF.items():
    put('outfits',f'{RAW}/Multiple Oufits/{mo[i]}',f'no tattoos, {c}, slicked back ponytail, plain white studio background',f'o_{i:02d}')
# ---------------- TATTOO look (from the earlier face set; 'tattoo body' folder excluded) ----------------
tat=[l.strip() for l in open('/tmp/tat.txt')]
NECK={0,1,2,3,4,5,6,7,8,9,10}; FACE={11,12,13,14,17,18,19,20,21}
for i in sorted(NECK|FACE|{15}):
    if i in (16,17): continue
    f=f'{AV1}/{tat[i]}'
    if i in NECK: c='neck tattoos, no face tattoos, long hair down, close-up portrait, head and shoulders'
    elif i==15: c='face tattoos, neck tattoos, chest tattoos, sequined top, long hair down, upper body'
    else: c='face tattoos, neck tattoos, long hair down, close-up portrait, head and shoulders'
    put('tattoo',f,c,f'tt_{i:02d}')
# ---------------- clean face refs (identity only, no style token) ----------------
for n,c in [('03','slicked back ponytail'),('06','hair in bun, smirk'),('01','hair in bun'),('05','hair in bun, lips parted')]:
    put('face',f'{CLEAN}/clean_{n}.jpg',f'no tattoos, white t-shirt, {c}, close-up portrait, head and shoulders, {BG}',f'face_{n}')
json.dump(manifest,open('/workspace/fs/body/dataset_manifest.json','w'),indent=1)
from collections import Counter
c=Counter(x['style'] for x in manifest); print(c, sum(c.values()))
print('steps/epoch (bs1):', sum(REPEATS[k]*v for k,v in c.items()))
