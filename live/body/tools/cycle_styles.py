import asyncio, json, time, urllib.request, websockets
tok=open('/workspace/live/token').read().strip()
async def main():
    async with websockets.connect(f"ws://127.0.0.1:8080/ws?k={tok}") as ws:
        await ws.recv()
        for st,tat in [("av1main",False),("av1hourglass",False),("av1outfits",False),("av1testing",False),("av1main",True),("av1hourglass",True)]:
            await ws.send(json.dumps({"style":st,"tattoo":tat})); await ws.recv()
            await asyncio.sleep(4)
            for i in range(2):
                d=urllib.request.urlopen(f"http://127.0.0.1:8080/live.jpg?k={tok}").read()
                open(f"/workspace/live/cyc_{st}_{'tat' if tat else 'clean'}_{i}.jpg","wb").write(d)
                await asyncio.sleep(1.5)
            print(st,tat,urllib.request.urlopen(f"http://127.0.0.1:8080/stats?k={tok}").read().decode(),flush=True)
        await ws.send(json.dumps({"style":"av1main","tattoo":False})); await ws.recv()
asyncio.run(main())
