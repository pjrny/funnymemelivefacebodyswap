#!/usr/bin/env python
"""ws_bench_v2.py - pod-local throughput test for every v2 mode (no network in the loop).
Usage on the pod: /workspace/LivePortrait/.venv/bin/python ws_bench_v2.py FRAME_DIR [seconds]"""
import asyncio, glob, json, struct, sys, time
import websockets
TOK = open("/workspace/live/token").read().strip()
FR = [open(f, "rb").read() for f in sorted(glob.glob(sys.argv[1] + "/*.jpg"))]
SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 10


async def run(ctl):
    async with websockets.connect(f"ws://127.0.0.1:8080/ws?k={TOK}", max_size=None) as ws:
        await ws.send(json.dumps(ctl)); r = await ws.recv()
        while isinstance(r, bytes):
            r = await ws.recv()
        await asyncio.sleep(1.0)
        sent = got = 0; lat = []; srv = []; t0 = time.time(); inflight = 0

        async def sender():
            nonlocal sent, inflight
            i = 0
            while time.time() - t0 < SEC:
                if inflight < 3:
                    await ws.send(struct.pack("<dI", time.time(), i) + FR[i % len(FR)]); sent += 1; inflight += 1; i += 1
                await asyncio.sleep(0.005)

        async def receiver():
            nonlocal got, inflight
            while time.time() - t0 < SEC + 1:
                try:
                    m = await asyncio.wait_for(ws.recv(), 1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(m, bytes):
                    ts, fid, ms = struct.unpack("<dIf", m[:16]); inflight -= 1
                    if ms >= 0:
                        got += 1; lat.append((time.time() - ts) * 1e3); srv.append(ms)
        await asyncio.gather(sender(), receiver())
        lat.sort()
        print(json.dumps({"ctl": ctl, "fps": round(got / SEC, 1), "server_ms_avg": round(sum(srv) / max(1, len(srv)), 1),
                          "lat_ms_p50": round(lat[len(lat) // 2], 1) if lat else None}), flush=True)


async def main():
    for ctl in ({"mode": "lp", "set": "1"}, {"mode": "lp", "set": "2"}, {"mode": "lp", "source": "clean/clean_01.jpg"},
                {"mode": "swap", "set": "1", "enhance": "gpen"}, {"mode": "swap", "set": "1", "enhance": "gfpgan"},
                {"mode": "swap", "set": "1", "enhance": False}):
        await run(ctl)
asyncio.run(main())
