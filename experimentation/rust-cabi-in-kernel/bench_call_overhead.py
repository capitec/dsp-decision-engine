"""Bare njit -> extern "C" (real Rust) call overhead, against njit -> njit
(fully inlinable) — the same shape as the original C-stand-in probe
(`/tmp/claude-1000/cabi/bench.py`), redone with the real `rust_cabi_tree`
cdylib rather than C standing in for it.
"""
from __future__ import annotations

import time

from numba import njit

from tree_walk_cabi import trivial_c

trivial = trivial_c


@njit(cache=False)
def via_rust_c_abi(n):
    t = 0
    for i in range(n):
        t += trivial(i, 1)
    return t


@njit(cache=False)
def _inline(a, b):
    return a + b


@njit(cache=False)
def via_njit(n):
    t = 0
    for i in range(n):
        t += _inline(i, 1)
    return t


N = 5_000_000


def main() -> None:
    for name, f in (("njit -> Rust extern \"C\"", via_rust_c_abi), ("njit -> njit (inlinable)", via_njit)):
        f(10)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            f(N)
            times.append(time.perf_counter() - t0)
        best = min(times)
        print(f"{name:26s} {best/N*1e9:6.2f} ns/call  (range {min(times)/N*1e9:.2f}-{max(times)/N*1e9:.2f})")


if __name__ == "__main__":
    main()
