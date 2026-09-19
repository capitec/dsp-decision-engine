"""Print the tables from results.json. No measurement happens here."""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.json")))

for p in sorted({r["p_nomatch"] for r in d}):
    for m in ("first_match", "all"):
        rs = sorted([r for r in d if r["mode"] == m and r["p_nomatch"] == p],
                    key=lambda r: r["n_rules"])
        if not rs:
            continue
        print(f"\n=== mode={m}  p_nomatch={p} ===")
        print(f"{'rules':>6}{'lines':>7}{'reps':>5}{'compile_s':>11}{'ms/line':>9}"
              f"{'exec_ms@100k':>13}{'match':>8}")
        for r in rs:
            print(f"{r['n_rules']:6d}{r['emitted_lines']:7.0f}{r['compile_repeats']:5d}"
                  f"{r['compile_median_s']:11.3f}{r['ms_per_line']:9.2f}"
                  f"{r.get('exec_median_ms', float('nan')):13.3f}"
                  f"{r.get('match_rate', float('nan')):8.3f}")
        if len(rs) >= 4:
            xs = [math.log(r["emitted_lines"]) for r in rs]
            ys = [math.log(r["compile_median_s"]) for r in rs]
            n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
            b = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sum((x-mx)**2 for x in xs)
            a = my - b*mx
            ss = sum((y-my)**2 for y in ys)
            rss = sum((y-(a+b*x))**2 for x, y in zip(xs, ys))
            print(f"  global log-log fit: compile_s = {math.exp(a):.3e} * lines^{b:.3f}"
                  f"  (R2={1-rss/ss:.4f})")
            for i in range(len(rs)-1):
                lo, hi = rs[i], rs[i+1]
                e = (math.log(hi["compile_median_s"]/lo["compile_median_s"]) /
                     math.log(hi["emitted_lines"]/lo["emitted_lines"]))
                per = ((hi["compile_median_s"]-lo["compile_median_s"]) /
                       (hi["n_rules"]-lo["n_rules"]))
                print(f"    {lo['n_rules']:3d}->{hi['n_rules']:3d} rules:"
                      f" local exponent {e:5.2f}, marginal {per*1000:7.0f} ms/rule")
