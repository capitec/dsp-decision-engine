"""What a semantic `str` step costs: Python per row, njit per row, interpreted.

Every shape is measured in one process, with a numeric kernel as the reference
row, so machine load can't favour one variant.

    uv run python benchmarks/strings_default.py
"""
import gc
import time
import warnings

import numpy as np
import polars as pl

from decider import Engine, FallbackWarning, branch, flow, loop, param, step
from decider.engine.compile import Fallback
from decider.engine.compile.njit import jit
from decider.engine.ir.decls import base_annotation
from decider.engine.run.runners.stepped import SteppedRunner
from decider.types import is_raw

N = 200_000
MODE = "fused"
warnings.simplefilter("ignore", FallbackWarning)

# The njit-per-row variant wants the strings themselves, which the shipped `_typed` no longer builds.
_typed = SteppedRunner._typed
SteppedRunner._typed = lambda self, x, mask, decl, alive, source: (
    x if base_annotation(decl.annotation) is str and not is_raw(decl.annotation)
    else _typed(self, x, mask, decl, alive, source))


def rate(x: float) -> float:
    return 0.9 if x > 0.5 else 1.0


def sector_rate(sector: str, private: str = param("private")) -> float:
    return 0.9 if sector == private else 1.0


def is_private(sector: str) -> float:
    return 1.0 if sector == "private" else 0.0


def same(sector: str, owner: str, agent: str) -> float:
    return 1.0 if sector == owner or sector == agent else 0.0


def label(x: float) -> str:
    return "big" if x > 0.5 else "small"


def relabel(sector: str) -> str:
    return "private" if sector == "private" else "other"


def prefixed(sector: str) -> float:
    return 1.0 if sector.startswith("pri") or "bli" in sector or sector[:3] == "gov" else 0.0


def private(sector: str) -> bool:
    return sector == "private"


@step(output="rate")
def keep(x: float) -> float:
    return x


@step(output="rate")
def halve(x: float) -> float:
    return x / 2


def unpaid(months: float) -> bool:
    return months < 3.0


@step(output="months")
def tick(months: float, sector: str) -> float:
    return months + (1.0 if sector == "private" else 2.0)


rng = np.random.default_rng(0)
SECTORS = ["private", "public", "government", "blind_trust"]
frame = pl.DataFrame({
    "sector": rng.choice(SECTORS, N), "owner": rng.choice(SECTORS, N), "agent": rng.choice(SECTORS, N),
    "x": rng.uniform(0, 1, N), "months": np.zeros(N),
})
ROW = {"sector": "private", "owner": "public", "agent": "private", "x": 0.7, "months": 0.0}

SHAPES = {
    "numeric kernel (reference)": flow(rate),
    "str input vs str param": flow(sector_rate),
    "str input vs body literal": flow(is_private),
    "three str inputs": flow(same),
    "str output only": flow(label),
    "str input and str output": flow(relabel),
    "str in a branch condition": flow(branch(private, keep, halve, modifies=["rate"], name="by_sector")),
    "str in a loop body": flow(loop(unpaid, tick, carries=["months"], max_iterations=4, name="pay")),
    "str methods (startswith/in/slice)": flow(prefixed),
}


def _time(f, *args):
    t = time.perf_counter()
    f(*args)
    return time.perf_counter() - t


def batch(run, reps=3):
    run(frame)
    return N / min(_time(run, frame) for _ in range(reps))


def latency(score, calls):
    for _ in range(500):
        score(ROW)
    gc.collect()
    samples = sorted(_time(score, ROW) for _ in range(calls))
    return samples[len(samples) // 2] * 1e6, samples[int(len(samples) * 0.99)] * 1e6


def force_njit(exe):
    """Put every semantic-`str` fallback back on its njit dispatcher: the behaviour before this change."""
    exe.fallbacks()
    found = False
    for unit in exe.runner.units.values():
        if isinstance(unit, Fallback) and "as str" in unit.reason:
            unit.fn = jit(unit.calls[0].node.fn)[1]
            found = True
    return found


def measure(pipeline, variant):
    exe = Engine().bind(pipeline, mode="interpreted" if variant == "interpreted" else MODE)
    label = variant
    if variant != "interpreted":
        fell_back = force_njit(exe) if variant == "njit per row" else any("as str" in r for r in exe.fallbacks().values())
        if not fell_back:
            if variant == "njit per row":
                return None
            label = "kernel"
    calls = 2000 if variant == "interpreted" else 5000
    return (label, batch(exe.run), *latency(exe.score, calls))


def main() -> None:
    header = f"{'shape':<36}{'variant':<16}{'batch rows/s':>16}{'p50 us':>9}{'p99 us':>9}"
    print(header)
    print("-" * len(header))
    for name, pipeline in SHAPES.items():
        for variant in ("python per row", "njit per row", "interpreted"):
            got = measure(pipeline, variant)
            if got is None:
                continue
            label, rows, p50, p99 = got
            print(f"{name:<36}{label:<16}{rows:>16,.0f}{p50:>9.1f}{p99:>9.1f}")


if __name__ == "__main__":
    main()
