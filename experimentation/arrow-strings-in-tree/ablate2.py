"""Second ablation: from the 13 ns flat micro-kernel (ablate.py K4) toward
walk_chunk's shape, one ingredient at a time, plus a count of NRT
incref/decref calls in each kernel's LLVM IR. probe=ablate2."""
from __future__ import annotations

import json
import re
import time

import numpy as np
import polars as pl
from numba import njit

import arrowc
from kernel import (CONTAINS, EXACT, LEAF, PREFIX, STR, _is_valid, _match_at, _str_node, _u8_at,
                    build_tree, pattern_table, string_tables, walk_chunk)

OUT = open("results.jsonl", "a")


def log(**rec):
    rec["probe"] = "ablate2"
    OUT.write(json.dumps(rec) + "\n"); OUT.flush(); print(json.dumps(rec), flush=True)


def nrt_calls(dispatcher):
    ir = next(iter(dispatcher.inspect_llvm().values()))
    return {"incref": len(re.findall(r"call void @NRT_incref", ir)),
            "decref": len(re.findall(r"call void @NRT_decref", ir)),
            "ir_lines": ir.count("\n")}


# A: flat loop, _str_node + validity, single column, no pc loop
@njit(cache=True)
def kA(views_addr, valid_addr, valid_len, valid_off, nd, base, daddr, dsize, pat_bytes, pat_bounds, n, out):
    for i in range(n):
        validity = _u8_at(valid_addr, valid_len)
        if not _is_valid(validity, i + valid_off):
            out[i] = 0
            continue
        views = _u8_at(views_addr, 16 * n)
        poff = pat_bounds[0, 0]
        ok, hit = _str_node(views, i, daddr, dsize, base, nd, pat_bytes, poff, pat_bounds[0, 1] - poff, EXACT)
        out[i] = 1 if hit else 0


# B: same but the addresses come out of 1-element arrays indexed by a runtime col (as walk_chunk)
@njit(cache=True)
def kB(views_addr, valid_addr, valid_len, valid_off, nd, base, daddr, dsize, pat_bytes, pat_bounds, feat_idx, n, out):
    for i in range(n):
        col = feat_idx[0]
        validity = _u8_at(valid_addr[col], valid_len[col])
        if not _is_valid(validity, i + valid_off[col]):
            out[i] = 0
            continue
        views = _u8_at(views_addr[col], 16 * n)
        poff = pat_bounds[0, 0]
        ok, hit = _str_node(views, i, daddr, dsize, base[col], nd[col], pat_bytes, poff, pat_bounds[0, 1] - poff, EXACT)
        out[i] = 1 if hit else 0


# C: B inside a `while True` pc loop over a kind array (walk_chunk minus CMP/IS_TRUE)
@njit(cache=True)
def kC(views_addr, valid_addr, valid_len, valid_off, nd, base, daddr, dsize, pat_bytes, pat_bounds,
       kind, feat_idx, op, thr_slot, then_, else_, leaf_value, n, out):
    for i in range(n):
        pc = 0
        while True:
            k = kind[pc]
            if k == LEAF:
                out[i] = leaf_value[pc]
                break
            else:
                col = feat_idx[pc]
                validity = _u8_at(valid_addr[col], valid_len[col])
                if not _is_valid(validity, i + valid_off[col]):
                    pc = else_[pc]
                    continue
                views = _u8_at(views_addr[col], 16 * n)
                p = thr_slot[pc]
                poff = pat_bounds[p, 0]
                ok, hit = _str_node(views, i, daddr, dsize, base[col], nd[col], pat_bytes, poff, pat_bounds[p, 1] - poff, op[pc])
                if not ok:
                    out[i] = -1
                    break
                pc = then_[pc] if hit else else_[pc]


# D: C but views/validity arrays are built ONCE before the row loop (single column)
@njit(cache=True)
def kD(views_addr, valid_addr, valid_len, valid_off, nd, base, daddr, dsize, pat_bytes, pat_bounds,
       kind, feat_idx, op, thr_slot, then_, else_, leaf_value, n, out):
    validity = _u8_at(valid_addr[0], valid_len[0])
    views = _u8_at(views_addr[0], 16 * n)
    for i in range(n):
        pc = 0
        while True:
            k = kind[pc]
            if k == LEAF:
                out[i] = leaf_value[pc]
                break
            else:
                col = feat_idx[pc]
                if not _is_valid(validity, i + valid_off[col]):
                    pc = else_[pc]
                    continue
                p = thr_slot[pc]
                poff = pat_bounds[p, 0]
                ok, hit = _str_node(views, i, daddr, dsize, base[col], nd[col], pat_bytes, poff, pat_bounds[p, 1] - poff, op[pc])
                if not ok:
                    out[i] = -1
                    break
                pc = then_[pc] if hit else else_[pc]


def best(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    N = 1_000_000
    rng = np.random.default_rng(1)
    pool = ["dog walker", "cat", "Capitec Bank", "Department of Education", "Shoprite Checkers",
            "self-employed", "dogma industries", "Transnet SOC Ltd", "unemployed", "SARS", "hotdog stand", "dog"]
    strings = [pool[i] for i in rng.integers(0, 12, N)]
    strings[::97] = [None] * len(strings[::97])
    s = pl.Series("s", strings)
    v = arrowc.export(s)
    tabs = string_tables([v.chunks[0]])
    views, valid, valid_len, valid_off, nd, base, daddr, dsize = tabs
    pb, pbounds = pattern_table(["dog"])
    ref = np.array([0 if x is None else int(x == "dog") for x in strings])
    out = np.zeros(N, np.int64)
    tree = build_tree([(STR, 0, EXACT, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])
    T = (tree["kind"], tree["feat_idx"], tree["op"], tree["thr_slot"], tree["then_"], tree["else_"], tree["leaf_value"])

    aA = (int(views[0]), int(valid[0]), int(valid_len[0]), int(valid_off[0]), int(nd[0]), int(base[0]), daddr, dsize, pb, pbounds, N, out)
    kA(*aA); assert np.array_equal(out, ref)
    log(k="A flat loop, scalar addresses, validity + _str_node", ns=best(lambda: kA(*aA)) / N * 1e9, **nrt_calls(kA))

    aB = (views, valid, valid_len, valid_off, nd, base, daddr, dsize, pb, pbounds, tree["feat_idx"], N, out)
    kB(*aB); assert np.array_equal(out, ref)
    log(k="B flat loop, addresses from arrays[col]", ns=best(lambda: kB(*aB)) / N * 1e9, **nrt_calls(kB))

    aC = (views, valid, valid_len, valid_off, nd, base, daddr, dsize, pb, pbounds, *T, N, out)
    kC(*aC); assert np.array_equal(out, ref)
    log(k="C + while-True pc loop (walk_chunk shape, STR only)", ns=best(lambda: kC(*aC)) / N * 1e9, **nrt_calls(kC))

    kD(*aC); assert np.array_equal(out, ref)
    log(k="D = C with views/validity arrays hoisted out of the row loop", ns=best(lambda: kD(*aC)) / N * 1e9, **nrt_calls(kD))

    feats = np.empty((N, 0))
    aW = (feats, *tabs, pb, pbounds, tree["thresholds"], *T, out, 0, N)
    walk_chunk(*aW); assert np.array_equal(out, ref)
    log(k="W walk_chunk as in kernel.py", ns=best(lambda: walk_chunk(*aW)) / N * 1e9, **nrt_calls(walk_chunk))
    OUT.close()


if __name__ == "__main__":
    main()
