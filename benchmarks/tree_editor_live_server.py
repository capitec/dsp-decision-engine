"""Server cost of live editing on the largest real flat-rule sets (521 rules), which the browser editor cannot hold.

A websocket client renames one rule per message and waits for the ack, N times per tree, then saves the whole
document over REST a few times. Round trips are printed here; the per-message breakdown (lock, apply, check,
persist, validation) is in the backend log (`live apply`, `live message`, `live validation` lines).

    BASE=http://localhost:8710 N=50 uv run --no-project --with-requirements ui/tree-editor/backend/requirements.txt \
        --with-editable . --with polars python benchmarks/tree_editor_live_server.py
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

BASE = os.environ.get("BASE", "http://localhost:8710")
N = int(os.environ.get("N", "50"))


def pct(values, p):
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


async def until(ws, check):
    while True:
        message = json.loads(await ws.recv())
        if check(message):
            return message


async def one_tree(http: httpx.AsyncClient, tree_id: str) -> None:
    doc = BY_ID[tree_id].document
    project = (await http.post("/api/v1/projects", json={"name": tree_id, "project_type": "flat_rule",
                                                          "project_data": doc})).json()
    url = BASE.replace("http", "ws") + f"/api/v1/projects/{project['id']}/live"
    headers = {"Cookie": f"session_id={http.cookies['session_id']}", "Origin": BASE}
    async with websockets.connect(url, additional_headers=headers, max_size=2 ** 26) as ws:
        started = time.perf_counter()
        await ws.send(json.dumps({"type": "hello", "since": project["version"], "client_id": "bench"}))
        version = (await until(ws, lambda m: m["type"] == "ready"))["version"]
        hello_ms = (time.perf_counter() - started) * 1e3
        keys = [r["rule"]["id"] for r in (await http.get(f"/api/v1/projects/{project['id']}")).json()["project_data"]["rules"]]
        rtts = []
        for i in range(N):
            msg_id = str(uuid.uuid4())
            ops = [{"op": "replace", "path": f"/rules/{keys[i * 7 % len(keys)]}/meta/name", "value": f"renamed {i}"}]
            started = time.perf_counter()
            await ws.send(json.dumps({"type": "ops", "id": msg_id, "base": version, "ops": ops}))
            version = (await until(ws, lambda m: m.get("msg_id") == msg_id and m["type"] == "applied"))["version"]
            rtts.append((time.perf_counter() - started) * 1e3)
    current = (await http.get(f"/api/v1/projects/{project['id']}")).json()
    puts = []
    for _ in range(5):
        started = time.perf_counter()
        r = await http.put(f"/api/v1/projects/{project['id']}", json={"project_data": current["project_data"]})
        assert r.status_code == 200, r.text
        puts.append((time.perf_counter() - started) * 1e3)
    print(f"{tree_id}: project {project['id']} bytes {len(json.dumps(doc))} hello+stamp+catch-up {hello_ms:.0f} ms; "
          f"op round trip n={N} p50 {statistics.median(rtts):.1f} p95 {pct(rtts, 0.95):.1f} max {max(rtts):.1f} ms; "
          f"REST PUT (incl. validation_errors) p50 {statistics.median(puts):.0f} ms")


async def main():
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as http:
        email = f"{uuid.uuid4().hex[:10]}@example.com"
        await http.post("/api/auth/basic/register", data={"email": email, "password": "Passw0rd!x", "name": "Bench"})
        if "session_id" not in http.cookies:
            await http.post("/api/auth/basic/login", data={"email": email, "password": "Passw0rd!x"})
        for tree_id in ("generated-521", "fraud-live"):
            await one_tree(http, tree_id)


asyncio.run(main())
