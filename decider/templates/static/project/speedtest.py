"""
Speed test for $project_title.

Usage (server must already be running):
    python $project_dir/speedtest.py [--url URL] [--n N] [--concurrency C]
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

# TODO: replace with your module's input columns
PAYLOAD = json.dumps({"id": ["row_1"]}).encode()
HEADERS = {"Content-Type": "application/json"}


async def run(url: str, n: int, concurrency: int):
    timings = []
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient):
        nonlocal errors
        async with semaphore:
            t0 = time.perf_counter()
            try:
                r = await client.post(url, content=PAYLOAD, headers=HEADERS)
                if r.status_code != 200:
                    errors += 1
            except Exception:
                errors += 1
            timings.append((time.perf_counter() - t0) * 1000)

    print(f"Sending {n} requests  concurrency={concurrency}  →  {url}")
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(url, content=PAYLOAD, headers=HEADERS)  # warm-up
        wall_start = time.perf_counter()
        await asyncio.gather(*[one(client) for _ in range(n)])
        wall_ms = (time.perf_counter() - wall_start) * 1000

    ok = n - errors
    p = sorted(timings)

    def pct(q):
        return p[min(int(len(p) * q / 100), len(p) - 1)]

    print(f"\n{'─'*40}")
    print(f"  Requests    {n}  ({ok} OK, {errors} errors)")
    print(f"  Wall time   {wall_ms:.0f} ms")
    print(f"  Throughput  {ok / (wall_ms / 1000):.0f} req/s")
    print(f"{'─'*40}")
    print(f"  Latency (ms)  min={min(timings):.1f}  p50={pct(50):.1f}  p90={pct(90):.1f}  p99={pct(99):.1f}  max={max(timings):.1f}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",         default="http://localhost:8080/predict")
    parser.add_argument("--n",           type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.n, args.concurrency))
