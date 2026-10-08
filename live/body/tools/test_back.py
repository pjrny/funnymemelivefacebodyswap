import asyncio, json, struct, time, numpy as np, websockets
tok=open('/workspace/live/token').read().strip()
# standing, seen from behind: no nose/eyes, ears low-confidence, person's right shoulder on the IMAGE RIGHT
kp=np.array([[0,0,0],[320,140,1],[360,140,1],[370,220,1],[375,300,1],[280,140,1],[270,220,1],[265,300,1],[345,300,1],[345,400,1],[345,470,1],[295,300,1],[295,400,1],[295,470,1],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],np.float32)
front=kp.copy(); front[[2,3,4,8,9,10]],front[[5,6,7,11,12,13]]=kp[[5,6,7,11,12,13]],kp[[2,3,4,8,9,10]]
front[0]=[320,85,1]; front[14]=[310,75,1]; front[15]=[330,75,1]; front[16]=[300,80,1]; front[17]=[340,80,1]
async def main():
    async with websockets.connect(f"ws://127.0.0.1:8080/ws?k={tok}", max_size=2**22) as ws:
        await ws.recv()
        for name,k in [("back",kp),("front_fullbody",front)]:
            for i in range(3):
                await ws.send(struct.pack("<dIHH",time.time(),i,640,480)+k.tobytes())
                while True:
                    m=await ws.recv()
                    if isinstance(m,bytes) and len(m)>16: break
            open(f"/workspace/live/test_{name}.jpg","wb").write(m[16:])
            print(name,"ok",flush=True)
asyncio.run(main())
