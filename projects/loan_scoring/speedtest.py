"""
Speed test for the loan scoring server.

Usage (server must already be running):
    python projects/loan_scoring/speedtest.py [--url URL] [--n N] [--concurrency C]

Defaults: url=http://localhost:8080, n=1000 requests, concurrency=50
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

PAYLOAD = json.dumps({
    "debt":          [25000],
    "income":        [50000],
    "credit_used":   [4000],
    "credit_limit":  [10000],
}).encode()

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
        # warm-up
        await client.post(url, content=PAYLOAD, headers=HEADERS)

        wall_start = time.perf_counter()
        await asyncio.gather(*[one(client) for _ in range(n)])
        wall_ms = (time.perf_counter() - wall_start) * 1000

    ok = n - errors
    rps = ok / (wall_ms / 1000)
    p = sorted(timings)

    def pct(q):
        return p[min(int(len(p) * q / 100), len(p) - 1)]

    print(f"\n{'─'*40}")
    print(f"  Requests    {n}  ({ok} OK, {errors} errors)")
    print(f"  Wall time   {wall_ms:.0f} ms")
    print(f"  Throughput  {rps:.0f} req/s")
    print(f"{'─'*40}")
    print(f"  Latency (ms)")
    print(f"    min   {min(timings):.1f}")
    print(f"    p50   {pct(50):.1f}")
    print(f"    p90   {pct(90):.1f}")
    print(f"    p99   {pct(99):.1f}")
    print(f"    max   {max(timings):.1f}")
    print(f"    mean  {statistics.mean(timings):.1f}")
    print(f"    stdev {statistics.stdev(timings):.1f}")
    print(f"{'─'*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",         default="http://localhost:8080/predict")
    parser.add_argument("--n",           type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.n, args.concurrency))
