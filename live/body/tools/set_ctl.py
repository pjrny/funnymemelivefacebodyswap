import asyncio, json, sys, websockets
tok=open('/workspace/live/token').read().strip()
async def main():
    async with websockets.connect(f"ws://127.0.0.1:8080/ws?k={tok}") as ws:
        await ws.recv(); await ws.send(sys.argv[1]); print(await ws.recv())
asyncio.run(main())
