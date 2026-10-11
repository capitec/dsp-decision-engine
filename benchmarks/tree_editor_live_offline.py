"""Server cost of merging an offline batch of edits on the largest real flat-rule sets (521 rules).

Per run: another client renames 10 rules live, then a client that went offline at the version before sends one
offline batch of 50 renames of other rules (merged), and one renaming a rule the other client changed (rejected).
Round trips are printed here; the breakdown (replay of the base document, conflict check, validation) is in the
backend log (`live apply ... offline=True` lines).

    BASE=http://localhost:8720 RUNS=10 uv run --no-project --with-requirements ui/tree-editor/backend/requirements.txt \
        --with-editable . --with polars python benchmarks/tree_editor_live_offline.py
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

BASE = os.environ.get("BASE", "http://localhost:8720")
RUNS = int(os.environ.get("RUNS", "10"))


def pct(values, p):
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


async def until(ws, check):
    while True:
        message = json.loads(await ws.recv())
        if check(message):
            return message


def rename(key, value):
    return {"op": "replace", "path": f"/rules/{key}/meta/name", "value": value}


async def send(ws, base, ops, offline=False):
    msg_id = str(uuid.uuid4())
    started = time.perf_counter()
    await ws.send(json.dumps({"type": "ops", "id": msg_id, "base": base, "ops": ops, "offline": offline}))
    answer = await until(ws, lambda m: m.get("msg_id") == msg_id and m["type"] in ("applied", "rejected"))
    return answer, (time.perf_counter() - started) * 1e3


async def one_tree(http: httpx.AsyncClient, tree_id: str) -> None:
    doc = BY_ID[tree_id].document
    project = (await http.post("/api/v1/projects", json={"name": tree_id, "project_type": "flat_rule",
                                                          "project_data": doc})).json()
    url = BASE.replace("http", "ws") + f"/api/v1/projects/{project['id']}/live"
    headers = {"Cookie": f"session_id={http.cookies['session_id']}", "Origin": BASE}
    async with websockets.connect(url, additional_headers=headers, max_size=2 ** 26) as online, \
            websockets.connect(url, additional_headers=headers, max_size=2 ** 26) as offline:
        for ws, client in ((online, "online"), (offline, "offline")):
            await ws.send(json.dumps({"type": "hello", "since": project["version"], "client_id": client}))
            version = (await until(ws, lambda m: m["type"] == "ready"))["version"]
        keys = [r["rule"]["id"] for r in (await http.get(f"/api/v1/projects/{project['id']}")).json()["project_data"]["rules"]]
        merged, rejected = [], []
        for run in range(RUNS):
            base = version
            for i in range(10):
                answer, _ = await send(online, version, [rename(keys[i], f"online {run}.{i}")])
                version = answer["version"]
            batch = [rename(keys[20 + i * 7 % (len(keys) - 20)], f"offline {run}.{i}") for i in range(50)]
            answer, ms = await send(offline, base, batch, offline=True)
            assert answer["type"] == "applied", answer
            version = answer["version"]
            merged.append(ms)
            answer, ms = await send(offline, base, [rename(keys[0], "clash")] + batch, offline=True)
            assert answer["type"] == "rejected" and answer["conflicts"], answer
            rejected.append(ms)
    print(f"{tree_id}: bytes {len(json.dumps(doc))} versions {version}; offline batch of 50 against 10 other edits, "
          f"n={RUNS}: merged p50 {statistics.median(merged):.0f} p95 {pct(merged, 0.95):.0f} ms, "
          f"rejected p50 {statistics.median(rejected):.0f} p95 {pct(rejected, 0.95):.0f} ms")


async def main():
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as http:
        email = f"{uuid.uuid4().hex[:10]}@example.com"
        await http.post("/api/auth/basic/register", data={"email": email, "password": "Passw0rd!x", "name": "Bench"})
        if "session_id" not in http.cookies:
            await http.post("/api/auth/basic/login", data={"email": email, "password": "Passw0rd!x"})
        for tree_id in ("generated-521", "fraud-live"):
            await one_tree(http, tree_id)


asyncio.run(main())
