"""An online editor's round trip while K offline batches arrive at once (as after a backend restart).

On the 521-rule set, an online client renames one rule every 100 ms on node 1 while K clients, spread over the
nodes, each send one offline batch of 50 renames of other rules at the same moment. Printed: how many batches
merged, their completion times, and the online round trips before and during the herd.

    PORTS=8750,8751 K=10 uv run --no-project --with-requirements ui/tree-editor/backend/requirements.txt \
        --with-editable . --with polars python benchmarks/tree_editor_live_herd.py
"""

import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui/tree-editor/e2e"))
from real_trees import BY_ID  # noqa: E402

PORTS = [int(p) for p in os.environ.get("PORTS", "8750,8751").split(",")]
K = int(os.environ.get("K", "10"))


def pct(values, p):
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


async def until(ws, check):
    while True:
        message = json.loads(await ws.recv())
        if check(message):
            return message


def rename(key, value):
    return {"op": "replace", "path": f"/rules/{key}/meta/name", "value": value}


async def main():
    base_url = f"http://localhost:{PORTS[0]}"
    async with httpx.AsyncClient(base_url=base_url, timeout=60) as http:
        email = f"{uuid.uuid4().hex[:10]}@example.com"
        await http.post("/api/auth/basic/register", data={"email": email, "password": "Passw0rd!x", "name": "Bench"})
        if "session_id" not in http.cookies:
            await http.post("/api/auth/basic/login", data={"email": email, "password": "Passw0rd!x"})
        project = (await http.post("/api/v1/projects", json={"name": "herd", "project_type": "flat_rule",
                                                              "project_data": BY_ID["generated-521"].document})).json()
        cookie = http.cookies["session_id"]

    async def connect(i, client):
        port = PORTS[i % len(PORTS)]
        ws = await websockets.connect(f"ws://localhost:{port}/api/v1/projects/{project['id']}/live",
                                      additional_headers={"Cookie": f"session_id={cookie}",
                                                          "Origin": f"http://localhost:{port}"}, max_size=2 ** 26)
        await ws.send(json.dumps({"type": "hello", "since": 0, "client_id": client}))
        return ws, (await until(ws, lambda m: m["type"] == "ready"))["version"]

    online, _ = await connect(0, "online")
    async with httpx.AsyncClient(base_url=base_url, timeout=60, cookies={"session_id": cookie}) as http:
        data = (await http.get(f"/api/v1/projects/{project['id']}")).json()["project_data"]
    keys = [r["rule"]["id"] for r in data["rules"]]
    offline = [await connect(i + 1, f"offline {i}") for i in range(K)]
    base = offline[-1][1]
    trips, stop = [], False

    async def online_loop():
        n = 0
        while not stop:
            msg_id = str(uuid.uuid4())
            started = time.perf_counter()
            await online.send(json.dumps({"type": "ops", "id": msg_id, "base": base,
                                          "ops": [rename(keys[0], f"online {n}")]}))
            await until(online, lambda m: m.get("msg_id") == msg_id and m["type"] in ("applied", "rejected"))
            trips.append((time.perf_counter() - started) * 1e3)
            n += 1
            await asyncio.sleep(0.1)

    async def batch(i, ws):
        msg_id = str(uuid.uuid4())
        ops = [rename(keys[1 + i * 50 + j], f"offline {i}.{j}") for j in range(50)]
        started = time.perf_counter()
        await ws.send(json.dumps({"type": "ops", "id": msg_id, "base": base, "ops": ops, "offline": True}))
        answer = await until(ws, lambda m: m.get("msg_id") == msg_id and m["type"] in ("applied", "rejected"))
        return answer["type"], (time.perf_counter() - started) * 1e3

    task = asyncio.create_task(online_loop())
    await asyncio.sleep(1)
    before = len(trips)
    results = await asyncio.gather(*(batch(i, ws) for i, (ws, _) in enumerate(offline)))
    during = trips[before:]
    await asyncio.sleep(1)
    stop = True
    await task
    done = [ms for _, ms in results]
    print(f"K={K} offline batches of 50, generated-521, {len(PORTS)} nodes: "
          f"{[kind for kind, _ in results].count('applied')} merged, done after p50 {statistics.median(done):.0f} "
          f"max {max(done):.0f} ms; online round trip before p50 {statistics.median(trips[:before]):.0f} ms, during "
          f"n={len(during)} p50 {statistics.median(during):.0f} p95 {pct(during, 0.95):.0f} max {max(during):.0f} ms")


asyncio.run(main())
