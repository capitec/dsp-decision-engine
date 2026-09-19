"""Distribution of the collision effect across independent trials.

Each trial uses a fresh njit dispatcher and a fresh, globally unique class name,
so trials cannot contaminate one another. Answers: is the 1.03x from runs 1-2 or
the 2.53x from run 3 the real number?

Run: <repo>/.venv/bin/python stability.py [n_trials]
"""
import collections, sys, time
from statistics import median
from numba import njit

N_CALLS, REPEATS, WARMUP = 2000, 11, 300

def percall_us(fn, arg):
    for _ in range(WARMUP): fn(arg)
    s = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(N_CALLS): fn(arg)
        s.append((time.perf_counter() - t0) / N_CALLS * 1e6)
    return median(s), min(s), max(s)

def trial(i):
    A = collections.namedtuple(f"St{i}", ["x", "y"])
    @njit(cache=False)
    def k(t): return t.x + t.y
    a = A(1.0, 2.0)
    before = percall_us(k, a)
    B = collections.namedtuple(f"St{i}", ["x", "y"])   # the twin
    k(B(3.0, 4.0))
    after = percall_us(k, a)
    return before, after, len(k.signatures)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
print(f"{'trial':>5} {'before med':>11} {'before min':>11} {'after med':>10} "
      f"{'after min':>10} {'med ratio':>10} {'min ratio':>10} {'sigs':>5}")
ratios_med, ratios_min, befores, afters = [], [], [], []
for i in range(n):
    (bm, bmin, bmax), (am, amin, amax), sigs = trial(i)
    ratios_med.append(am / bm); ratios_min.append(amin / bmin)
    befores.append(bm); afters.append(am)
    print(f"{i:>5} {bm:>11.3f} {bmin:>11.3f} {am:>10.3f} {amin:>10.3f} "
          f"{am/bm:>10.2f} {amin/bmin:>10.2f} {sigs:>5}")
print()
print(f"  median-of-trials  before = {median(befores):.3f} us   after = {median(afters):.3f} us")
print(f"  ratio (median of per-trial medians) = {median(ratios_med):.2f}x  "
      f"range {min(ratios_med):.2f}-{max(ratios_med):.2f}")
print(f"  ratio (median of per-trial minima)  = {median(ratios_min):.2f}x  "
      f"range {min(ratios_min):.2f}-{max(ratios_min):.2f}")
print(f"  max 'after' observed across all trials = {max(afters):.3f} us")
print(f"  min/max 'before' across all trials     = {min(befores):.3f}/{max(befores):.3f} us")
