"""
EXPERIMENT A, part 3 - are Categorical physical codes a safe boundary representation?

Part 2 suggested string->Categorical->uint32 codes is the only njit-consumable string
path.  A decision engine would compile `sector_code == 1` as a constant.  That is only
sound if the code for a given string is the SAME in every frame the engine ever sees.
This isolates that question, including across processes.

Run:  <repo>/.venv/bin/python experimentation/dtype-boundary/part3_catcodes.py
      (also run twice and diff, to see cross-process stability)
"""
import os
import sys
import polars as pl


def codes(vals, dtype=pl.Categorical):
    s = pl.Series("s", vals).cast(dtype)
    return list(s._get_buffers()["values"].to_numpy()), list(s)


print(f"--- process pid={os.getpid()} polars {pl.__version__}")

print("\nA. codes assigned in order of first appearance, within one process:")
for vals in (["PRIVATE", "PUBLIC", "SME"],
             ["SME", "PRIVATE", "PUBLIC"],
             ["PUBLIC"],
             ["PRIVATE", "GOVERNMENT"],
             ["GOVERNMENT", "PUBLIC", "MUNICIPAL"],
             ["PRIVATE", "PUBLIC", "SME", "GOVERNMENT", "MUNICIPAL"]):
    c, v = codes(vals)
    print(f"   {str(v):<62} -> {c}")

print("\nB. THE TEST THAT MATTERS - a fresh frame whose first row is not PRIVATE:")
fresh = pl.Series("s", ["SME", "SME", "PRIVATE"]).cast(pl.Categorical)
print(f"   values={list(fresh)} codes={list(fresh._get_buffers()['values'].to_numpy())}")
print("   category mapping as polars reports it:", fresh.cat.get_categories().to_list()
      if hasattr(fresh, 'cat') else 'n/a')

print("\nC. does the code survive a round trip through a different frame's categories?")
f_a = pl.DataFrame({"sector": ["PRIVATE", "PUBLIC"]}).with_columns(pl.col("sector").cast(pl.Categorical))
f_b = pl.DataFrame({"sector": ["PUBLIC", "PRIVATE"]}).with_columns(pl.col("sector").cast(pl.Categorical))
ca = f_a["sector"]._get_buffers()["values"].to_numpy()
cb = f_b["sector"]._get_buffers()["values"].to_numpy()
print(f"   frame A {list(f_a['sector'])} -> {list(ca)}")
print(f"   frame B {list(f_b['sector'])} -> {list(cb)}")
map_a = dict(zip(list(f_a["sector"]), [int(x) for x in ca]))
map_b = dict(zip(list(f_b["sector"]), [int(x) for x in cb]))
print(f"   mapping A={map_a}  mapping B={map_b}  SAME={map_a == map_b}")

print("\nD. Enum, same questions (Enum has a declared, fixed category list):")
E = pl.Enum(["PRIVATE", "PUBLIC", "SME"])
for vals in (["PRIVATE", "PUBLIC", "SME"], ["SME"], ["PUBLIC", "PUBLIC"]):
    c, v = codes(vals, E)
    print(f"   {str(v):<40} -> {c}")
E2 = pl.Enum(["SME", "PUBLIC", "PRIVATE"])   # same strings, different declared order
c2, v2 = codes(["PRIVATE", "PUBLIC", "SME"], E2)
print(f"   SAME strings under a DIFFERENT Enum declaration order -> {c2}")
print("   => Enum codes are a function of the DECLARED list, so they are stable iff")
print("      the declaration is versioned with the compiled kernel.")

print("\nE. cross-process check: rerun this file and compare section B's codes.")
