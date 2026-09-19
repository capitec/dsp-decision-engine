#!/usr/bin/env python3
"""Render results.json from fusion_body_cost.py as the tables used in the write-up."""
import json, sys, pathlib

path = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).parent / "results.json")
d = json.load(open(path))
res, vec = d["results"], d["vec"]

Ms = sorted({r["modules"] for r in res})
Ns = sorted({r["rows"] for r in res})
bodies = list(dict.fromkeys(r["body"] for r in res))
branches = list(dict.fromkeys(r["branch"] for r in res))

print("### boundary floor (one extra split kernel, empty body)")
for f in d.get("floor", []):
    print(f"  n={f['rows']:<8d} dispatch={f['dispatch_us']:7.3f}us  "
          f"{f['ns_per_row']:6.3f} ns/row  {(f['gb_per_s'] or 0):.1f} GB/s")

for body in bodies:
    for br in branches:
        print(f"\n### ratio split/fused  (>1 = FUSION WINS)   body={body} branch={br}")
        print("| modules | " + " | ".join(f"{n:,}" for n in Ns) +
              " | fused ns/step @1M | split ns/step @1M |")
        print("|" + "---|" * (len(Ns) + 3))
        for M in Ms:
            cells = []
            f1m = s1m = None
            for n in Ns:
                r = next((r for r in res if r["body"] == body and r["branch"] == br
                          and r["modules"] == M and r["rows"] == n), None)
                cells.append(f"{r['ratio_split_over_fused']:.2f}" if r else "-")
                if r and n == max(Ns):
                    f1m, s1m = r["fused_ns_per_step_row"], r["split_ns_per_step_row"]
            print(f"| {M} | " + " | ".join(cells) +
                  f" | {f1m:.2f} | {s1m:.2f} |" if f1m is not None
                  else f"| {M} | " + " | ".join(cells) + " | - | - |")

print("\n### vectorisation of the FUSED kernel")
print("| body | branch | M | llvm vec values | llvm vec fp ops | asm scalar | asm packed xmm | asm packed ymm/zmm | simd frac |")
print("|" + "---|" * 9)
for v in vec:
    if v.get("shape") != "fused":
        continue
    print(f"| {v['body']} | {v['branch']} | {v['modules']} | "
          f"{v.get('llvm_vector_values')} | {v.get('llvm_vector_fp_ops')} | "
          f"{v.get('asm_fp_scalar')} | {v.get('asm_fp_packed_xmm')} | "
          f"{v.get('asm_fp_packed_wide')} | {v.get('asm_simd_frac')} |")

print("\n### vectorisation of a SPLIT (single-module) kernel")
print("| body | branch | llvm vec values | asm scalar | asm packed xmm | asm packed ymm/zmm | simd frac |")
print("|" + "---|" * 7)
for v in vec:
    if v.get("shape") != "split_module_0":
        continue
    print(f"| {v['body']} | {v['branch']} | {v.get('llvm_vector_values')} | "
          f"{v.get('asm_fp_scalar')} | {v.get('asm_fp_packed_xmm')} | "
          f"{v.get('asm_fp_packed_wide')} | {v.get('asm_simd_frac')} |")

print("\n### does the SIGN of the fusion decision flip with body cost?  (at 1M rows)")
print("| modules | " + " | ".join(f"{b}/{br}" for b in bodies for br in branches) + " |")
print("|" + "---|" * (1 + len(bodies) * len(branches)))
for M in Ms:
    cells = []
    for b in bodies:
        for br in branches:
            r = next((r for r in res if r["body"] == b and r["branch"] == br
                      and r["modules"] == M and r["rows"] == max(Ns)), None)
            cells.append(f"{r['ratio_split_over_fused']:.2f}" if r else "-")
    print(f"| {M} | " + " | ".join(cells) + " |")

bad = [r for r in res if r["equiv"] == "MISMATCH"]
print(f"\nequivalence: {len(res)-len(bad)}/{len(res)} bitwise-exact fused==split, "
      f"{len(bad)} mismatches")
