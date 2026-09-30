"""`each` across its two execution modes, record counts, item counts and child complexity.

Answers: where per-row vs batch wins, and how items-per-record and a heavier
child (intermediate steps + a loop) move the crossover. Writes
`benchmarks/results_each_modes.csv`.

    uv run python benchmarks/each_modes.py [items...] [records...]
"""
import csv
import gc
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from decider import EachMode, Engine, each, flow, loop, missing_as, param, step

ARGS = [int(a) for a in sys.argv[1:]]
ITEMS = ARGS[:4] if ARGS else [1, 5, 20, 100]
RECORDS = ARGS[4:] if len(ARGS) > 4 else [1, 10, 50, 100, 500, 1000, 10000]
OUT = Path(__file__).with_name("results_each_modes.csv")


# --- lightweight child: one scalar step ----------------------------------------

def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
    return weight > heavy_kg


def light(each_mode):
    return each("items", flow(heavy, name="item"), name="items", execution_mode=each_mode)


# --- complex child: several intermediate steps then a small loop -----------------

@step(output="a1")
def a1(weight: float) -> float:
    return weight * 1.1 + 2.0


@step(output="a2")
def a2(a1: float) -> float:
    return a1 * 0.9 - 1.0


@step(output="a3")
def a3(a2: float) -> float:
    return a2 + 3.0


@step(output="a4")
def a4(a3: float) -> float:
    return a3 / 1.2


@step(output="a5")
def a5(a4: float) -> float:
    return a4 - 4.0


def again(i: int) -> bool:
    return i < 3


@step(outputs=("i", "total"))
def fold(total: float, a5: float, i: int) -> tuple[int, float]:
    return i + 1, total + a5


@step(output="i")
def tick(i: int) -> int:
    return i + 1


@step(outputs=("i", "total"))
def seed(a5: float) -> tuple[int, float]:
    return 0, 0.0


def complex_loop(each_mode):
    body = flow(fold, tick, name="body")
    child = flow(a1, a2, a3, a4, a5, seed, loop(again, body, carries=["i", "total"],
                                                max_iterations=4, name="acc"), name="item")
    return each("items", child, name="items", execution_mode=each_mode)


PIPELINES = {"light": light, "complex": complex_loop}


def make_frame(records, items):
    rng = np.random.default_rng(0)
    weights = rng.uniform(1, 50, (records, items)).round(2)
    rows = [[{"weight": float(w)} for w in ws] for ws in weights]
    return pl.DataFrame({"items": rows},
                        schema={"items": pl.List(pl.Struct({"weight": pl.Float64}))})


def timed(fn, repeat, warm=2):
    for _ in range(warm):
        fn()
    gc.disable()
    times = []
    for _ in range(repeat):
        t = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t) / 1e3)  # microseconds
    gc.enable()
    return times


def med(times):
    return statistics.median(times)


def reps(records, items):
    total = records * items
    if records >= 5000 or total >= 20000:
        return 1
    if records >= 500 or total >= 5000:
        return 3
    return 5


if __name__ == "__main__":
    print(f"items={ITEMS} records={RECORDS} -> {OUT}")
    rows = []
    header = f"{'child':<9}{'each_mode':<10}{'items':>6}{'records':>8}   {'run us':>10} {'us/rec':>9}"
    print(header)
    print("-" * len(header))
    for child_name, build in PIPELINES.items():
        for each_mode in (EachMode.PER_ROW, EachMode.BATCH):
            mode = "per_row" if each_mode is EachMode.PER_ROW else "batch"
            for items in ITEMS:
                for r in RECORDS:
                    pipe = build(each_mode)
                    exe = Engine().bind(pipe, mode="fused")
                    frame = make_frame(r, items)
                    us = med(timed(lambda: exe.run(frame), reps(r, items)))
                    per = us / max(r, 1)
                    rows.append({"child": child_name, "each_mode": mode, "items": items,
                                 "records": r, "run_us": round(us, 2), "us_per_record": round(per, 3)})
                    print(f"{child_name:<9}{mode:<10}{items:>6}{r:>8}   {us:>10.1f} {per:>9.2f}")
            print()

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["child", "each_mode", "items", "records", "run_us", "us_per_record"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")
