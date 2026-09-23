"""Where do ~150 ns/row go in the STR node? Micro-kernels over the same 1M-row
low-cardinality column, each adding one ingredient. Written to results.jsonl
as probe=ablate.
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

import arrowc
from kernel import _u8_at, _u32, string_tables

OUT = open("results.jsonl", "a")


def log(**rec):
    rec["probe"] = "ablate"
    OUT.write(json.dumps(rec) + "\n"); OUT.flush(); print(json.dumps(rec), flush=True)


@intrinsic
def load_u8(typingctx, addr):
    if not isinstance(addr, types.Integer):
        return None

    def codegen(context, builder, sig, args):
        ptr = builder.inttoptr(args[0], ir.IntType(8).as_pointer())
        return builder.load(ptr)
    return types.uint8(addr), codegen


@intrinsic
def load_u32(typingctx, addr):
    if not isinstance(addr, types.Integer):
        return None

    def codegen(context, builder, sig, args):
        ptr = builder.inttoptr(args[0], ir.IntType(32).as_pointer())
        return builder.load(ptr, align=1)
    return types.uint32(addr), codegen


# K0: numpy array made ONCE in python, read the u32 length per row
@njit(cache=True)
def k0_len_from_numpy(views, n, out):
    acc = 0
    for i in range(n):
        acc += _u32(views, 16 * i)
    out[0] = acc


# K1: per-row _u8_at (array struct built per row), read u32 length
@njit(cache=True)
def k1_len_via_u8_at(views_addr, n, out):
    acc = 0
    for i in range(n):
        views = _u8_at(views_addr, 16 * n)
        acc += _u32(views, 16 * i)
    out[0] = acc


# K2: raw load intrinsic, read u32 length
@njit(cache=True)
def k2_len_via_load(views_addr_, n, out):
    views_addr = np.uint64(views_addr_)
    acc = 0
    for i in range(n):
        acc += load_u32(views_addr + np.uint64(16 * i))
    out[0] = acc


# K3: raw loads, full exact match of 'dog' inline-or-long
@njit(cache=True)
def k3_exact_via_load(views_addr_, data_addr, data_size, nd, pat, m, n, out):
    views_addr = np.uint64(views_addr_)
    for i in range(n):
        at = views_addr + np.uint64(16 * i)
        ln = np.int64(load_u32(at))
        if ln != m:
            out[i] = 0
            continue
        if ln <= 12:
            p = at + np.uint64(4)
        else:
            bi = np.int64(load_u32(at + np.uint64(8)))
            off = np.int64(load_u32(at + np.uint64(12)))
            if bi >= nd or off + ln > np.int64(data_size[bi]):
                out[i] = -1
                continue
            p = data_addr[bi] + np.uint64(off)
        hit = 1
        for j in range(m):
            if load_u8(p + np.uint64(j)) != pat[j]:
                hit = 0
                break
        out[i] = hit


# K4: _u8_at per row + exact match through array indexing (what kernel.py v2 does)
@njit(cache=True)
def k4_exact_via_u8_at(views_addr, data_addr, data_size, nd, pat, m, n, out):
    for i in range(n):
        views = _u8_at(views_addr, 16 * n)
        at = 16 * i
        ln = np.int64(_u32(views, at))
        if ln != m:
            out[i] = 0
            continue
        if ln <= 12:
            buf = views
            off = at + 4
        else:
            bi = np.int64(_u32(views, at + 8))
            off = np.int64(_u32(views, at + 12))
            if bi >= nd or off + ln > np.int64(data_size[bi]):
                out[i] = -1
                continue
            buf = _u8_at(data_addr[bi], np.int64(data_size[bi]))
        hit = 1
        for j in range(m):
            if buf[off + j] != pat[j]:
                hit = 0
                break
        out[i] = hit


# K5: _u8_at hoisted out of the loop (views once), data buffers still per row
@njit(cache=True)
def k5_exact_views_hoisted(views_addr, data_addr, data_size, nd, pat, m, n, out):
    views = _u8_at(views_addr, 16 * n)
    for i in range(n):
        at = 16 * i
        ln = np.int64(_u32(views, at))
        if ln != m:
            out[i] = 0
            continue
        if ln <= 12:
            buf = views
            off = at + 4
        else:
            bi = np.int64(_u32(views, at + 8))
            off = np.int64(_u32(views, at + 12))
            if bi >= nd or off + ln > np.int64(data_size[bi]):
                out[i] = -1
                continue
            buf = _u8_at(data_addr[bi], np.int64(data_size[bi]))
        hit = 1
        for j in range(m):
            if buf[off + j] != pat[j]:
                hit = 0
                break
        out[i] = hit


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
    s = pl.Series("s", [pool[i] for i in rng.integers(0, 12, N)])
    v = arrowc.export(s)
    c = v.chunks[0]
    _, _, _, _, nd_arr, _, daddr, dsize = string_tables([c])
    views_arr = np.ctypeslib.as_array(__import__("ctypes").cast(c.views, __import__("ctypes").POINTER(__import__("ctypes").c_uint8)), shape=(16 * N,))
    pat = np.frombuffer(b"dog", dtype=np.uint8)
    ref = np.array([int(x == "dog") for x in s.to_list()])
    o1 = np.zeros(1, np.int64); o = np.zeros(N, np.int64)

    k0_len_from_numpy(views_arr, N, o1); log(k="K0 numpy views made once, read len", ns=best(lambda: k0_len_from_numpy(views_arr, N, o1)) / N * 1e9)
    k1_len_via_u8_at(c.views, N, o1); log(k="K1 _u8_at per row, read len", ns=best(lambda: k1_len_via_u8_at(c.views, N, o1)) / N * 1e9)
    k2_len_via_load(c.views, N, o1); log(k="K2 raw load_u32 per row, read len", ns=best(lambda: k2_len_via_load(c.views, N, o1)) / N * 1e9)
    args = (c.views, daddr, dsize, int(nd_arr[0]), pat, 3, N, o)
    k3_exact_via_load(*args); assert np.array_equal(o, ref)
    log(k="K3 raw loads, exact match", ns=best(lambda: k3_exact_via_load(*args)) / N * 1e9)
    k4_exact_via_u8_at(*args); assert np.array_equal(o, ref)
    log(k="K4 _u8_at per row, exact match (kernel v2 shape)", ns=best(lambda: k4_exact_via_u8_at(*args)) / N * 1e9)
    k5_exact_views_hoisted(*args); assert np.array_equal(o, ref)
    log(k="K5 views _u8_at hoisted, data per row, exact match", ns=best(lambda: k5_exact_views_hoisted(*args)) / N * 1e9)
    OUT.close()


if __name__ == "__main__":
    main()
