"""Turn results.json (one or more runs) into the tables in README.md.

    python analyse.py results.json results_run2.json > results_tables.txt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BODY_ORDER = ["trivial", "medium10", "inner64", "inner256"]


def load(paths):
    return [(Path(p).name, json.loads(Path(p).read_text())) for p in paths]


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def cells(run):
    return {(c["body"], c["rows"]): c for c in run["sweep"] if "skipped" not in c}


def t_speedup(runs):
    hdr("1. serial/prange speedup (>1 = prange wins).  median of trial medians.")
    all_rows = sorted({r for _, run in runs for c in run["sweep"] for r in [c["rows"]]})
    for body in BODY_ORDER:
        print(f"\n  body = {body}")
        print(
            "  {:>10}  {:>12}  {:>12}  {:>8}  {:>8}  {:>9}".format(
                "rows", "serial ms", "prange ms", "speedup", "run2", "ser ns/row"
            )
        )
        for n in all_rows:
            c = cells(runs[0][1]).get((body, n))
            if not c:
                continue
            c2 = cells(runs[1][1]).get((body, n)) if len(runs) > 1 else None
            print(
                "  {:>10,}  {:>12.4f}  {:>12.4f}  {:>7.3f}x  {:>7}  {:>9.2f}".format(
                    n,
                    c["serial_med_s"] * 1e3,
                    c["prange_med_s"] * 1e3,
                    c["speedup_med"],
                    f"{c2['speedup_med']:.3f}x" if c2 else "-",
                    c["serial_ns_per_row"],
                )
            )


def t_crossover(runs):
    hdr("2. crossover: lowest row count whose measured speedup exceeds 1.0")
    print(
        "  {:>10}  {:>12}  {:>12}  {:>16}  {:>16}".format(
            "body", "run1 rows", "run2 rows", "run1 serial@xover", "run1 ns/row@xover"
        )
    )
    for body in BODY_ORDER:
        out = []
        for _, run in runs:
            cs = sorted(
                [c for c in run["sweep"] if "skipped" not in c and c["body"] == body],
                key=lambda c: c["rows"],
            )
            hit = next((c for c in cs if c["speedup_med"] > 1.0), None)
            out.append(hit)
        a = out[0]
        b = out[1] if len(out) > 1 else None
        print(
            "  {:>10}  {:>12}  {:>12}  {:>14.1f} us  {:>16.2f}".format(
                body,
                f"{a['rows']:,}" if a else "never",
                (f"{b['rows']:,}" if b else "never") if len(out) > 1 else "-",
                a["serial_med_s"] * 1e6 if a else float("nan"),
                a["serial_ns_per_row"] if a else float("nan"),
            )
        )
    print(
        "\n  'serial@xover' is total serial wall-clock of the kernel at the crossover row\n"
        "  count.  It is near-constant across bodies spanning 1500x in per-row cost."
    )


def t_floor(runs):
    hdr("3. the prange fixed cost (OpenMP fork/join floor), measured")
    for name, run in runs:
        vals = [c["prange_med_s"] for c in run["sweep"] if "skipped" not in c]
        small = [
            (c["body"], c["rows"], c["prange_med_s"] * 1e6)
            for c in run["sweep"]
            if "skipped" not in c and c["rows"] <= 2000
        ]
        print(f"\n  {name}: min prange time over all cells = {min(vals) * 1e6:.1f} us")
        print("   small-n prange times (us) — this is the floor, not work:")
        for b, n, us in small:
            print(f"     {b:>9} n={n:>6,}  {us:8.1f}")


def t_static_rule(runs):
    hdr("4. can a static rule 'prange iff predicted serial time > C' reproduce the picks?")
    obs = []
    for _, run in runs:
        for c in run["sweep"]:
            if "skipped" in c:
                continue
            obs.append((c["serial_med_s"], c["speedup_med"] > 1.0))
    print(f"  {len(obs)} (body, rows, run) cells total\n")
    print("  {:>12}  {:>10}  {:>10}  {:>8}".format("C (us)", "correct", "wrong", "acc %"))
    best = None
    for c_us in (20, 50, 75, 100, 125, 150, 200, 300, 500, 1000):
        thr = c_us * 1e-6
        ok = sum(1 for t, won in obs if (t > thr) == won)
        acc = ok / len(obs) * 100
        print("  {:>12}  {:>10}  {:>10}  {:>7.1f}".format(c_us, ok, len(obs) - ok, acc))
        if best is None or acc > best[1]:
            best = (c_us, acc)
    print(f"\n  best threshold in this grid: C = {best[0]} us  ({best[1]:.1f}% of cells)")
    thr = best[0] * 1e-6
    print("\n  cells the best static rule gets wrong:")
    for name, run in runs:
        for c in run["sweep"]:
            if "skipped" in c:
                continue
            pred = c["serial_med_s"] > thr
            if pred != (c["speedup_med"] > 1.0):
                print(
                    f"    {name:>18} {c['body']:>9} n={c['rows']:>9,}  "
                    f"serial={c['serial_med_s'] * 1e6:9.1f}us  speedup={c['speedup_med']:.3f}x"
                )


def t_warmup_cost(runs):
    hdr("5. what a warmup measurement costs")
    for name, run in runs:
        print(f"\n  {name}")
        print("   compile cost of the two variants (ms), phase 1:")
        for k, v in run["sweep_compile_ms"].items():
            print(f"     {k:>20}  {v:8.1f}")
        print("\n   probe cost = k reps of BOTH variants on a probe batch (ms):")
        print(
            "   {:>10} {:>9} {:>3} {:>12} {:>8} {:>9}".format(
                "body", "probe n", "k", "probe cost", "pick", "ratio"
            )
        )
        for r in run["warmup_probe"]:
            if "skipped" in r:
                continue
            print(
                "   {:>10} {:>9,} {:>3} {:>10.2f}ms {:>8} {:>8.3f}x".format(
                    r["body"], r["probe_rows"], r["k"], r["probe_cost_s"] * 1e3, r["pick"], r["ratio"]
                )
            )


def t_flip(runs):
    hdr("6. reproducibility of a cheap warmup pick (15 independent picks, k=3 reps)")
    print(
        "  {:>10} {:>10} {:>14} {:>14} {:>10} {:>10}".format(
            "body", "rows", "run1 %prange", "run2 %prange", "r1 ratio", "r1 spread%"
        )
    )
    idx = []
    for _, run in runs:
        idx.append({(r["body"], r["rows"]): r for r in run["flip_test"] if "skipped" not in r})
    keys = sorted(set(idx[0]) | (set(idx[1]) if len(idx) > 1 else set()))
    keys.sort(key=lambda k: (BODY_ORDER.index(k[0]), k[1]))
    for k in keys:
        a = idx[0].get(k)
        b = idx[1].get(k) if len(idx) > 1 else None
        print(
            "  {:>10} {:>10,} {:>13} {:>14} {:>9} {:>10.1f}".format(
                k[0],
                k[1],
                f"{a['frac_prange'] * 100:.0f}%" if a else "-",
                f"{b['frac_prange'] * 100:.0f}%" if b else "-",
                f"{a['ratio_med']:.3f}x" if a else "-",
                a["ratio_spread_pct"] if a else float("nan"),
            )
        )
    unstable = [
        (k, idx[0][k]) for k in idx[0] if 0.0 < idx[0][k]["frac_prange"] < 1.0
    ]
    print(f"\n  run1: {len(unstable)}/{len(idx[0])} cases flipped within one process.")
    if unstable:
        lo = min(idx[0][k]["ratio_med"] for k, _ in unstable)
        hi = max(idx[0][k]["ratio_med"] for k, _ in unstable)
        print(f"  their true ratios span {lo:.3f}x .. {hi:.3f}x")
    stable_near1 = [
        (k, v)
        for k, v in idx[0].items()
        if v["frac_prange"] in (0.0, 1.0) and 0.5 < v["ratio_med"] < 2.0
    ]
    for k, v in sorted(stable_near1, key=lambda kv: kv[1]["ratio_med"]):
        print(
            f"    unanimous but near 1.0: {k[0]:>9} n={k[1]:>8,} "
            f"ratio {v['ratio_med']:.3f}x  [{v['ratio_min']:.3f}-{v['ratio_max']:.3f}]"
        )


def t_compile(runs):
    hdr("7. parallel=True compile cost (doc 01 §4b claims 1.2-2.6x)")
    print(
        "  {:>12} {:>14} {:>16} {:>9} {:>9}".format(
            "shape", "serial (ms)", "parallel (ms)", "ratio", "run2"
        )
    )
    idx = [{r["shape"]: r for r in run["compile_cost"] if "skipped" not in r} for _, run in runs]
    for shape in idx[0]:
        a = idx[0][shape]
        b = idx[1].get(shape) if len(idx) > 1 else None
        print(
            "  {:>12} {:>14.1f} {:>16.1f} {:>8.2f}x {:>8}".format(
                shape,
                a["serial_compile_s"] * 1e3,
                a["parallel_compile_s"] * 1e3,
                a["ratio"],
                f"{b['ratio']:.2f}x" if b else "-",
            )
        )
    allr = [r["ratio"] for i in idx for r in i.values()]
    print(f"\n  observed ratio range over both runs: {min(allr):.2f}x .. {max(allr):.2f}x")


def t_identity(runs):
    hdr("8. correctness: prange output vs serial output")
    bad = [
        (name, c["body"], c["rows"])
        for name, run in runs
        for c in run["sweep"]
        if "skipped" not in c and not c["bitwise_identical"]
    ]
    total = sum(1 for _, run in runs for c in run["sweep"] if "skipped" not in c)
    print(f"  {total - len(bad)}/{total} cells bitwise identical.")
    for b in bad:
        print(f"  MISMATCH: {b}")


def main():
    paths = sys.argv[1:] or ["results.json"]
    runs = load(paths)
    print("runs:", ", ".join(n for n, _ in runs))
    m = runs[0][1]["meta"]
    print(
        f"python {m['python']}  numba {m['numba']}  llvmlite {m['llvmlite']}  "
        f"numpy {m['numpy']}\ncpus {m['cpu_count']}  NUMBA_NUM_THREADS {m['numba_num_threads']}  "
        f"threading layer {runs[0][1].get('threading_layer')}"
    )
    t_speedup(runs)
    t_crossover(runs)
    t_floor(runs)
    t_static_rule(runs)
    t_warmup_cost(runs)
    t_flip(runs)
    t_compile(runs)
    t_identity(runs)


if __name__ == "__main__":
    main()
