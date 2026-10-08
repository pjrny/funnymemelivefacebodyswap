import asyncio, json, struct, time, sys, numpy as np, websockets
tok=open('/workspace/live/token').read().strip()
kp=np.array([[320,80,1],[320,140,1],[280,140,1],[270,220,1],[265,300,1],[360,140,1],[370,220,1],[375,300,1],[295,300,1],[295,400,1],[295,470,1],[345,300,1],[345,400,1],[345,470,1],[310,70,1],[330,70,1],[300,75,1],[340,75,1]],np.float32)
async def main():
    async with websockets.connect(f"ws://127.0.0.1:8080/ws?k={tok}", max_size=2**22) as ws:
        print(await ws.recv())
        for steps in [int(s) for s in sys.argv[1:]] or [2,3,4]:
            await ws.send(json.dumps({"steps":steps})); print(await ws.recv())
            n=0; t=time.time(); srv=[]
            for i in range(30):
                k=kp.copy(); k[4,1]-=i*3; await ws.send(struct.pack("<dIHH",time.time(),i,640,480)+k.tobytes())
                m=await ws.recv(); ts,fid,ms=struct.unpack("<dIf",m[:16]); srv.append(ms)
                if i==29: open(f'/workspace/live/bench_{steps}.jpg','wb').write(m[16:])
            dt=(time.time()-t)/30; print(f"steps {steps}: {dt*1000:.0f} ms/frame ({1/dt:.1f} fps) server {np.mean(srv[5:]):.0f} ms", flush=True)
asyncio.run(main())
