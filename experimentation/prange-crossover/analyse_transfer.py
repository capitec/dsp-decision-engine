#!/usr/bin/env python
"""
E-F follow-up analysis: the two questions the first two runs left open.

The existing analyse.py scores the sweep, the fixed prange floor, a static
threshold rule, within-process flip rate, and compile cost.  It does NOT
answer either half of the actual deliverable:

  (A) PROBE TRANSFER.  Doc 05 sec 5.1 proposes "measure both variants per kernel
      at warmup".  Warmup does not know the production batch size, so the
      natural implementation probes a small batch.  Does a pick made on a
      small probe batch still hold at the production row count?
      analyse.py prints probe picks and truth picks in separate tables and
      never joins them.  This joins them and scores the error.

  (B) CROSS-PROCESS PICK STABILITY.  Doc 08 sec 8 wants the chosen variant in
      the audit record.  analyse.py's flip test repeats picks *inside one
      process*.  The audit question is whether two processes -- different
      day, different machine load -- record the same variant.  That needs
      independent runs, which is what results*.json are.

  (C) The static rule "prange iff predicted serial wall-clock > C" fitted on
      runs 1+2 and scored OUT OF SAMPLE on run 3.  analyse.py fits and scores
      on the same pooled cells, which cannot detect overfitting of C.

Usage:
    /path/to/.venv/bin/python analyse_transfer.py [results*.json ...]
Defaults to results.json results_run2.json results_run3.json in this dir.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(paths):
    runs = {}
    for p in paths:
        p = Path(p)
        if p.exists():
            runs[p.name] = json.load(open(p))
    return runs


def truth_pick(speedup: float) -> str:
    return "prange" if speedup > 1.0 else "serial"


def rule(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# ---------------------------------------------------------------------------
# (A) probe transfer
# ---------------------------------------------------------------------------
def probe_transfer(runs):
    rule("A. PROBE TRANSFER -- does a pick made on a small probe batch hold at\n"
         "   the production row count?  (doc 05 sec 5.1)")

    # ground truth per run: (body, rows) -> pick, from the 5-trial sweep
    print("\n  probe pick vs TRUE pick at each production row count.")
    print("  '.' = probe agrees with truth,  X = probe picks the WRONG variant.\n")

    targets = [1_000, 10_000, 100_000, 1_000_000, 5_000_000]
    total = wrong = 0
    wrong_detail = []

    for rname, d in runs.items():
        truth = {}
        for r in d["sweep"]:
            truth[(r["body"], r["rows"])] = (
                truth_pick(r["speedup_med"]), r["speedup_med"], r["serial_med_s"]
            )
        print(f"  --- {rname} ---")
        hdr = "  {:>9s} {:>8s} {:>2s} {:>9s}  ".format("body", "probe n", "k", "probe pick")
        hdr += " ".join(f"{t:>9,d}" for t in targets)
        print(hdr)
        for pr in d["warmup_probe"]:
            if "skipped" in pr:
                continue
            cells = []
            for t in targets:
                key = (pr["body"], t)
                if key not in truth:
                    cells.append("{:>9s}".format("-"))
                    continue
                tp, sp, _ = truth[key]
                ok = tp == pr["pick"]
                total += 1
                if not ok:
                    wrong += 1
                    wrong_detail.append(
                        (rname, pr["body"], pr["probe_rows"], pr["k"], pr["pick"], t, tp, sp)
                    )
                cells.append("{:>9s}".format(("." if ok else "X") + f" {sp:.2f}x"))
            print("  {:>9s} {:>8,d} {:>2d} {:>10s}  ".format(
                pr["body"], pr["probe_rows"], pr["k"], pr["pick"]) + " ".join(cells))
        print()

    print(f"  OVERALL: {total-wrong}/{total} probe->target predictions correct "
          f"({100*(total-wrong)/total:.1f}%), {wrong} wrong.")

    # the decisive slice: cheap probe (n<=10k) predicting a large production batch
    rule("A2. the slice that matters: a CHEAP probe (n <= 10k) predicting the\n"
         "    variant for a LARGE production batch (n >= 100k)")
    tot2 = wrong2 = 0
    per_body = {}
    for rname, d in runs.items():
        truth = {(r["body"], r["rows"]): truth_pick(r["speedup_med"]) for r in d["sweep"]}
        for pr in d["warmup_probe"]:
            if "skipped" in pr or pr["probe_rows"] > 10_000:
                continue
            for t in (100_000, 1_000_000, 5_000_000):
                if (pr["body"], t) not in truth:
                    continue
                tot2 += 1
                ok = truth[(pr["body"], t)] == pr["pick"]
                b = per_body.setdefault(pr["body"], [0, 0])
                b[0] += 1
                if not ok:
                    wrong2 += 1
                    b[1] += 1
    print(f"\n  {'body':>10s} {'preds':>7s} {'wrong':>7s} {'acc %':>8s}")
    for b, (n, w) in per_body.items():
        print(f"  {b:>10s} {n:>7d} {w:>7d} {100*(n-w)/n:>7.1f}%")
    print(f"  {'ALL':>10s} {tot2:>7d} {wrong2:>7d} {100*(tot2-wrong2)/tot2:>7.1f}%")

    if wrong_detail:
        print("\n  every wrong prediction (probe -> target):")
        for rn, b, pn, k, pp, t, tp, sp in wrong_detail:
            print(f"    {rn:<20s} {b:>9s} probe n={pn:>7,d} k={k} said {pp:<6s} "
                  f"but n={t:>9,d} truth={tp:<6s} ({sp:.2f}x)")


# ---------------------------------------------------------------------------
# (B) cross-process stability of the recorded pick
# ---------------------------------------------------------------------------
def cross_run_stability(runs):
    rule("B. CROSS-PROCESS PICK STABILITY (doc 08 sec 8: the audit record)\n"
         "   Each run is an independent process.  Does the recorded variant agree?")

    if len(runs) < 2:
        print("  need >=2 runs; skipped")
        return

    names = list(runs)
    cells = {}
    for rname, d in runs.items():
        for r in d["sweep"]:
            cells.setdefault((r["body"], r["rows"]), {})[rname] = (
                truth_pick(r["speedup_med"]), r["speedup_med"], r["serial_med_s"],
                r.get("pick_unanimous"),
            )

    print(f"\n  {'body':>9s} {'rows':>10s} " + " ".join(f"{n[:14]:>15s}" for n in names)
          + f" {'agree':>7s} {'ser us':>9s}")
    disagree = []
    unanimous_but_disagree = []
    for (b, n), per in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if len(per) < len(names):
            continue
        picks = [per[nm][0] for nm in names]
        agree = len(set(picks)) == 1
        if not agree:
            disagree.append((b, n, per))
            if all(per[nm][3] for nm in names):
                unanimous_but_disagree.append((b, n, per))
        ser_us = statistics.median([per[nm][2] for nm in names]) * 1e6
        print(f"  {b:>9s} {n:>10,d} "
              + " ".join(f"{per[nm][0]:>8s}{per[nm][1]:>7.2f}x" for nm in names)
              + f" {'yes' if agree else 'NO':>7s} {ser_us:>9.1f}")

    tot = sum(1 for _, per in cells.items() if len(per) == len(names))
    print(f"\n  {tot - len(disagree)}/{tot} cells recorded the SAME variant in all "
          f"{len(names)} processes ({100*(tot-len(disagree))/tot:.1f}%).")
    if disagree:
        print(f"\n  {len(disagree)} cells disagreed ACROSS processes:")
        for b, n, per in disagree:
            sers = statistics.median([per[nm][2] for nm in names]) * 1e6
            sp = [f"{per[nm][1]:.2f}x" for nm in names]
            print(f"    {b:>9s} n={n:>9,d}  speedups {', '.join(sp)}  serial={sers:.0f}us")
    if unanimous_but_disagree:
        print(f"\n  *** {len(unanimous_but_disagree)} of those were UNANIMOUS within every")
        print("      process (all 5 trials agreed) yet still disagreed between")
        print("      processes.  Within-process agreement does NOT imply the audit")
        print("      record is reproducible:")
        for b, n, per in unanimous_but_disagree:
            print(f"        {b:>9s} n={n:>9,d}  "
                  + ", ".join(f"{nm.split('.')[0]}={per[nm][0]}({per[nm][1]:.2f}x)"
                              for nm in names))


# ---------------------------------------------------------------------------
# (C) static rule, fitted on runs 1+2, scored out of sample on run 3
# ---------------------------------------------------------------------------
def static_rule_oos(runs):
    rule("C. STATIC HEURISTIC 'prange iff predicted serial wall-clock > C',\n"
         "   FITTED on the first runs, scored OUT OF SAMPLE on the last run.")

    names = list(runs)
    if len(names) < 2:
        print("  need >=2 runs; skipped")
        return
    fit_names, test_name = names[:-1], names[-1]

    def cells_of(rns):
        out = []
        for rn in rns:
            for r in runs[rn]["sweep"]:
                out.append((r["serial_med_s"] * 1e6, truth_pick(r["speedup_med"]),
                            r["body"], r["rows"], r["speedup_med"]))
        return out

    fit = cells_of(fit_names)
    test = cells_of([test_name])

    grid = [20, 40, 50, 60, 75, 90, 100, 125, 150, 200, 300, 500, 1000]
    print(f"\n  fitted on: {', '.join(fit_names)}  ({len(fit)} cells)")
    print(f"  tested on: {test_name}  ({len(test)} cells)\n")
    print(f"  {'C (us)':>8s} {'fit acc %':>11s} {'test acc %':>12s}")
    best, best_acc = None, -1.0
    for C in grid:
        fa = sum(1 for s, t, *_ in fit if ("prange" if s > C else "serial") == t) / len(fit)
        ta = sum(1 for s, t, *_ in test if ("prange" if s > C else "serial") == t) / len(test)
        print(f"  {C:>8d} {100*fa:>10.1f}% {100*ta:>11.1f}%")
        if fa > best_acc:
            best, best_acc = C, fa
    ta = [(s, t, b, n, sp) for s, t, b, n, sp in test
          if ("prange" if s > best else "serial") != t]
    print(f"\n  best C on the FIT set = {best} us ({100*best_acc:.1f}% fit).")
    print(f"  OUT-OF-SAMPLE accuracy on {test_name}: "
          f"{100*(len(test)-len(ta))/len(test):.1f}%  ({len(ta)}/{len(test)} wrong)")
    for s, t, b, n, sp in ta:
        print(f"    MISS {b:>9s} n={n:>9,d} serial={s:>8.1f}us truth={t:<6s} ({sp:.2f}x)")

    print("\n  For comparison: a FIXED ROW THRESHOLD, best single row count,")
    print("  fitted the same way (this is the '~50k' style rule doc 01 rejects):")
    rowgrid = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 1_000_000]
    bestR, bestRa = None, -1.0
    for R in rowgrid:
        fa = sum(1 for s, t, b, n, sp in fit if ("prange" if n > R else "serial") == t) / len(fit)
        if fa > bestRa:
            bestR, bestRa = R, fa
    ra = sum(1 for s, t, b, n, sp in test if ("prange" if n > bestR else "serial") == t) / len(test)
    print(f"    best row threshold on fit = {bestR:,d} rows ({100*bestRa:.1f}% fit), "
          f"out of sample {100*ra:.1f}%")


# ---------------------------------------------------------------------------
# (D) what the warmup measurement actually costs, as a policy
# ---------------------------------------------------------------------------
def probe_budget(runs):
    rule("D. COST OF THE WARMUP MEASUREMENT AS A POLICY (doc 08 sec 4 staged activation)")
    print("\n  A warmup that probes AT the production row count, k reps of both")
    print("  variants, for all 4 kernels.  Sum over kernels = added startup time.\n")
    print(f"  {'run':>20s} {'probe n':>9s} {'k':>2s} {'sum over 4 kernels':>20s}")
    for rname, d in runs.items():
        agg = {}
        for pr in d["warmup_probe"]:
            if "skipped" in pr:
                continue
            agg.setdefault((pr["probe_rows"], pr["k"]), []).append(pr["probe_cost_s"])
        for (n, k), v in sorted(agg.items()):
            if len(v) < 4:
                continue
            print(f"  {rname:>20s} {n:>9,d} {k:>2d} {sum(v)*1e3:>17.1f} ms")

    print("\n  plus the unavoidable compile of BOTH variants (phase 1, ms):")
    for rname, d in runs.items():
        sc = d.get("sweep_compile_ms", {})
        tot = sum(sc.values())
        par = sum(v for k, v in sc.items() if k.endswith("prange"))
        print(f"  {rname:>20s} both variants total {tot:>8.0f} ms   "
              f"(the prange halves alone {par:>8.0f} ms)")


def compile_cost(runs):
    rule("E. parallel=True COMPILE COST (doc 01 sec 4b claims '1.2-2.6x')")
    names = list(runs)
    rows = {}
    for rname, d in runs.items():
        for c in d.get("compile_cost", []):
            rows.setdefault(c["shape"], {})[rname] = c
    print(f"\n  {'shape':>10s} " + " ".join(f"{n.split('.')[0][:12]:>22s}" for n in names))
    allr = []
    for shape, per in rows.items():
        line = f"  {shape:>10s} "
        for n in names:
            if n in per:
                c = per[n]
                line += (f" {c['serial_compile_s']*1e3:>7.0f}/"
                         f"{c['parallel_compile_s']*1e3:>7.0f}={c['ratio']:>4.2f}x")
                allr.append(c["ratio"])
            else:
                line += " " * 22
        print(line)
    if allr:
        print(f"\n  serial ms / parallel ms = ratio.  observed ratio range over all runs:"
              f"  {min(allr):.2f}x .. {max(allr):.2f}x")
        print(f"  median {statistics.median(allr):.2f}x")


def main() -> int:
    paths = sys.argv[1:] or [HERE / "results.json", HERE / "results_run2.json",
                             HERE / "results_run3.json"]
    runs = load(paths)
    if not runs:
        print("no results files found", file=sys.stderr)
        return 1
    print(f"runs loaded: {', '.join(runs)}")
    m = next(iter(runs.values()))["meta"]
    print(f"python {m['python']}  numba {m['numba']}  numpy {m['numpy']}  "
          f"cpus {m['cpu_count']}  NUMBA_NUM_THREADS {m['numba_num_threads']}")
    for rn, d in runs.items():
        print(f"  {rn:>20s} started {d['meta']['started']}")

    probe_transfer(runs)
    cross_run_stability(runs)
    static_rule_oos(runs)
    probe_budget(runs)
    compile_cost(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
