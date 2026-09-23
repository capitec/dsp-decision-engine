"""Q3: what does raw-bytes-in-the-tree cost against today's two routes?

Same 1M-row data for every route, low (12 distinct) and high (one per row)
cardinality. Every route's cost is split into
    boundary : what has to happen to the column BEFORE the kernel runs
    kernel   : the per-row walk
and both are reported in ns/row. Routes:

    today/exact     cast(pl.Categorical) -> int32 codes -> float64 feature,
                    kernel does CMP EQ against the pattern's code
    today/nonexact  pl.col(...).str.{starts_with,ends_with,contains} before
                    the kernel -> bool -> float64 feature, kernel IS_TRUE
    mask/nonexact   today's encode + pattern over the distinct categories +
                    mask[code] -> float64 feature, kernel IS_TRUE
                    (SUMMARY §4 "per-category mask"; low cardinality only)
    arrow/<mode>    __arrow_c_stream__ export (O(1)) and the STR node reads
                    the row's bytes in-kernel. No encoding, no pass.

Also a GATED tree (x > t as root, ~1% of rows reach the string node) to show
what laziness buys when the kernel matches, and nothing else can.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import polars as pl

import arrowc
from kernel import (CMP, CONTAINS, EQ, EXACT, GT, IS_TRUE, LEAF, PREFIX, STR, SUFFIX,
                    build_tree, run_tree)

N = 1_000_000
REPS = 5
OUT = open("results.jsonl", "a")


def log(**rec):
    rec["probe"] = "bench"
    OUT.write(json.dumps(rec) + "\n")
    OUT.flush()
    print(json.dumps(rec), flush=True)


def best(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), r


def make_data(card: str):
    rng = np.random.default_rng(1)
    if card == "low":
        pool = ["dog walker", "cat", "Capitec Bank", "Department of Education", "Shoprite Checkers",
                "self-employed", "dogma industries", "Transnet SOC Ltd", "unemployed", "SARS",
                "hotdog stand", "dog"]
        strings = [pool[i] for i in rng.integers(0, 12, N)]
    else:
        strings = [f"merchant-{i:07d}-dog-{rng.integers(0, 1000):03d}" for i in range(N)]
        strings[::3] = [f"dog-{i}" for i in range(0, N, 3)]  # a third start with 'dog'
    strings[::97] = [None] * len(strings[::97])  # ~1% nulls
    x = rng.random(N)
    return pl.DataFrame({"s": strings, "x": x}), strings


PATTERN = "dog"
MODES = {"exact": EXACT, "prefix": PREFIX, "suffix": SUFFIX, "contains": CONTAINS}
POLARS_OP = {
    "prefix": lambda c: c.str.starts_with(PATTERN),
    "suffix": lambda c: c.str.ends_with(PATTERN),
    "contains": lambda c: c.str.contains(PATTERN, literal=True),
    "exact": lambda c: c == PATTERN,
}


def reference(strings, mode):
    f = {"exact": lambda s: s == PATTERN, "prefix": lambda s: s.startswith(PATTERN),
         "suffix": lambda s: s.endswith(PATTERN), "contains": lambda s: PATTERN in s}[mode]
    return np.array([0 if s is None else int(f(s)) for s in strings], dtype=np.int64)


# trees ---------------------------------------------------------------------
def tree_str(mode):   # root: STR node col 0 pattern 0
    return build_tree([(STR, 0, MODES[mode], 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])


def tree_cmp_eq(code):  # root: feats[:,0] == code
    return build_tree([(CMP, 0, EQ, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)],
                      thresholds=[float(code)])


def tree_is_true():  # root: feats[:,0] != 0
    return build_tree([(IS_TRUE, 0, 0, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])


def tree_gated_str(mode, thr):  # x > thr ? (STR ? 1 : 0) : 0   ; feats[:,0]=x
    return build_tree([(CMP, 0, GT, 0, 1, 3, 0), (STR, 0, MODES[mode], 0, 2, 3, 0),
                       (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)], thresholds=[thr])


def tree_gated_is_true(thr):  # feats[:,0]=x, feats[:,1]=bool
    return build_tree([(CMP, 0, GT, 0, 1, 3, 0), (IS_TRUE, 1, 0, 0, 2, 3, 0),
                       (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)], thresholds=[thr])


def main():
    for card in ("low", "high"):
        df, strings = make_data(card)
        s = df["s"]
        x = df["x"].to_numpy()
        empty_feats = np.empty((N, 0))
        n_distinct = s.n_unique()
        print(f"--- {card}: {n_distinct} distinct", file=sys.stderr)

        # ---------------- today / exact: dictionary-encode at the boundary --------
        def boundary_codes():
            enc = s.cast(pl.Categorical)
            codes = enc.to_physical().to_numpy().astype(np.float64)  # into the float64 slot, as today
            cats = enc.cat.get_categories().to_list()
            return codes, cats
        t_b, (codes, cats) = best(boundary_codes)
        code = cats.index(PATTERN) if PATTERN in cats else -1
        # a null code enters as NaN in the float slot (== never true) -- same as decider2
        feats = codes.reshape(-1, 1)
        tr = tree_cmp_eq(code)
        run_tree(feats, [], tr, [])  # compile
        t_k, out = best(lambda: run_tree(feats, [], tr, []))
        assert np.array_equal(out, reference(strings, "exact"))
        log(card=card, n=N, distinct=n_distinct, route="today/exact (encode + ==code)", mode="exact",
            boundary_ns=t_b / N * 1e9, kernel_ns=t_k / N * 1e9, total_ns=(t_b + t_k) / N * 1e9)

        # ---------------- today / non-exact: polars str op before the kernel ------
        for mode in ("prefix", "suffix", "contains"):
            def boundary_polars():
                return df.select(POLARS_OP[mode](pl.col("s")).fill_null(False).cast(pl.Float64))["s"].to_numpy()
            t_b, b = best(boundary_polars)
            feats = b.reshape(-1, 1)
            tr = tree_is_true()
            run_tree(feats, [], tr, [])
            t_k, out = best(lambda: run_tree(feats, [], tr, []))
            assert np.array_equal(out, reference(strings, mode)), mode
            log(card=card, n=N, distinct=n_distinct, route="today/nonexact (polars str op + is_true)", mode=mode,
                boundary_ns=t_b / N * 1e9, kernel_ns=t_k / N * 1e9, total_ns=(t_b + t_k) / N * 1e9)

        # ---------------- per-category mask (SUMMARY §4), low card only -----------
        if card == "low":
            for mode in ("prefix", "suffix", "contains"):
                def boundary_mask():
                    enc = s.cast(pl.Categorical)
                    codes = enc.to_physical().to_numpy()
                    cats = enc.cat.get_categories()
                    mask = POLARS_OP[mode](cats).to_numpy()
                    hit = np.zeros(len(codes), dtype=np.float64)
                    ok = ~np.isnan(codes) if codes.dtype.kind == "f" else np.ones(len(codes), bool)
                    hit[ok] = mask[codes[ok].astype(np.int64)]
                    return hit
                t_b, b = best(boundary_mask)
                feats = b.reshape(-1, 1)
                tr = tree_is_true()
                t_k, out = best(lambda: run_tree(feats, [], tr, []))
                assert np.array_equal(out, reference(strings, mode)), mode
                log(card=card, n=N, distinct=n_distinct, route="mask/nonexact (encode + category mask + is_true)",
                    mode=mode, boundary_ns=t_b / N * 1e9, kernel_ns=t_k / N * 1e9, total_ns=(t_b + t_k) / N * 1e9)

        # ---------------- arrow: raw bytes, matched at the node --------------------
        for mode in ("exact", "prefix", "suffix", "contains"):
            t_b, v = best(lambda: arrowc.export(s))
            tr = tree_str(mode)
            run_tree(empty_feats, [v], tr, [PATTERN])
            t_k, out = best(lambda: run_tree(empty_feats, [v], tr, [PATTERN]))
            assert np.array_equal(out, reference(strings, mode)), mode
            log(card=card, n=N, distinct=n_distinct, route="arrow/STR node (raw bytes in kernel)", mode=mode,
                boundary_ns=t_b / N * 1e9, kernel_ns=t_k / N * 1e9, total_ns=(t_b + t_k) / N * 1e9,
                n_data_buffers=len(v.chunks[0].data), n_chunks=len(v.chunks))

        # ---------------- gated: ~1% of rows reach the string node -----------------
        thr = 0.99
        for mode in ("prefix", "contains"):
            # today: polars pass over ALL rows regardless, then a gated is_true
            def boundary_polars():
                return df.select(POLARS_OP[mode](pl.col("s")).fill_null(False).cast(pl.Float64))["s"].to_numpy()
            t_b, b = best(boundary_polars)
            feats2 = np.stack([x, b], axis=1)
            tr = tree_gated_is_true(thr)
            t_k, out_today = best(lambda: run_tree(feats2, [], tr, []))
            log(card=card, n=N, distinct=n_distinct, route="today/nonexact GATED 1%", mode=mode,
                boundary_ns=t_b / N * 1e9, kernel_ns=t_k / N * 1e9, total_ns=(t_b + t_k) / N * 1e9)
            # arrow: the node only runs for the 1%
            t_b, v = best(lambda: arrowc.export(s))
            tr = tree_gated_str(mode, thr)
            feats1 = x.reshape(-1, 1)
            run_tree(feats1, [v], tr, [PATTERN])
            t_k, out_arrow = best(lambda: run_tree(feats1, [v], tr, [PATTERN]))
            assert np.array_equal(out_today, out_arrow)
            log(card=card, n=N, distinct=n_distinct, route="arrow/STR node GATED 1%", mode=mode,
                boundary_ns=t_b / N * 1e9, kernel_ns=t_k / N * 1e9, total_ns=(t_b + t_k) / N * 1e9)

        # ---------------- chunked frame, same answers ------------------------------
        df2 = pl.concat([df[: N // 3], df[N // 3: 2 * N // 3], df[2 * N // 3:]])
        s2 = df2["s"]
        v2 = arrowc.export(s2)
        tr = tree_str("contains")
        out2 = run_tree(empty_feats, [v2], tr, [PATTERN])
        ok = np.array_equal(out2, reference(strings, "contains"))
        t_k, _ = best(lambda: run_tree(empty_feats, [v2], tr, [PATTERN]))
        log(card=card, n=N, route="arrow/STR node on 3-chunk frame", mode="contains",
            polars_n_chunks=s2.n_chunks(), exported_chunks=[c.length for c in v2.chunks],
            answers_identical_to_single_chunk=bool(ok), kernel_ns=t_k / N * 1e9)

        del df, s, strings

    OUT.close()


if __name__ == "__main__":
    main()
