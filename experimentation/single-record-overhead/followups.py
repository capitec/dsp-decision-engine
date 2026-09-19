"""
EXPERIMENT N1 -- follow-up decomposition, run after n1_overhead.py's main
sweep showed phase 3 (marshal) and phase 5 (readback) dominating total cost.
This isolates WHY: is it numba, or is it numpy structured-dtype machinery
around a many-named-field array?

Answers, in order:
  1. Is "kernel dispatch" (39.9us in the main run) actually the njit call, or
     mostly np.empty() allocating the 633-field OUTPUT record array?
  2. Does a flat (positional, non-record) output representation allocate
     cheaper than a 633-field structured dtype?
  3. Is per-field named readback (673us) inherent, or a numpy-usage choice?
     (bulk .item() on the whole row vs field-by-field)
  4. Is per-field named marshal (220us) inherent, or the same choice?
     (assigning one tuple to the whole row vs field-by-field)
  5. What does reusing (pooling) the output buffer across calls cost --
     i.e. what's left once allocation is taken out of the picture entirely?

Reuses n1_overhead.py's shape, kernel, request, and timed_ns/pctiles/
pct_of_budget/jsonl_append helpers directly (imports the module, which also
re-triggers the one-time kernel compile -- ~11s, unavoidable without a
fixed-path cache, same tradeoff n1_overhead.py itself makes).

Run:
    /path/to/.venv/bin/python followups.py
Appends to the SAME results.jsonl as n1_overhead.py (run that first).
"""

from __future__ import annotations

import numpy as np

import n1_overhead as N


def main():
    n = N
    log, timed_ns, pctiles, pct_of_budget = n.log, n.timed_ns, n.pctiles, n.pct_of_budget
    jsonl_append, RESULTS_JSONL = n.jsonl_append, n.RESULTS_JSONL

    log("=== followups: decomposing marshal/readback ===")
    rec = n.marshal(n.REQUEST_DICT)
    rows = {}

    # 1. output-array allocation vs the njit call itself
    ns = timed_ns(lambda: np.empty(1, dtype=n.OUT_REC_DTYPE), 6000)
    rows["out_alloc_only_633field_record"] = pctiles(ns)

    pooled_out = np.empty(1, dtype=n.OUT_REC_DTYPE)
    ns = timed_ns(lambda: n.DRIVER(rec, n.COEF_F8, pooled_out), 6000)
    rows["njit_call_only_output_buffer_pooled"] = pctiles(ns)

    # 2. flat (3x 1D, positional) vs record (1x structured, named) allocation
    def alloc_3flat():
        return (
            np.empty(n.OUT_F8, dtype=np.float64),
            np.empty(n.OUT_I8, dtype=np.int64),
            np.empty(n.OUT_BOOL, dtype=np.bool_),
        )

    ns = timed_ns(alloc_3flat, 6000)
    rows["out_alloc_only_3flat_positional"] = pctiles(ns)

    ns = timed_ns(lambda: np.empty(1, dtype=n.IN_DTYPE), 6000)
    rows["in_alloc_only_400field_record"] = pctiles(ns)

    # 3. readback: per-field named vs bulk whole-row .item() vs flat .tolist()
    out = n.dispatch(rec)
    ns = timed_ns(lambda: n.readback(out), 6000)
    rows["readback_per_field_named"] = pctiles(ns)

    ns = timed_ns(lambda: n.readback_bulk(out), 6000)
    rows["readback_bulk_row_item"] = pctiles(ns)

    f8a, i8a, ba = alloc_3flat()

    def readback_flat_bulk():
        vals = f8a.tolist() + i8a.tolist() + ba.tolist()
        return dict(zip(n.OUT_ALL_NAMES, vals))

    ns = timed_ns(readback_flat_bulk, 6000)
    rows["readback_flat_tolist_bulk"] = pctiles(ns)

    # 4. marshal: per-field named vs one tuple assignment to the whole row
    def marshal_tuple(request):
        r = np.empty(1, dtype=n.IN_DTYPE)
        r[0] = tuple(request[nm] for nm in n.IN_ALL_NAMES)
        return r

    assert n.marshal(n.REQUEST_DICT) == marshal_tuple(n.REQUEST_DICT), "marshal variants disagree"
    ns = timed_ns(lambda: n.marshal(n.REQUEST_DICT), 6000)
    rows["marshal_per_field_named"] = pctiles(ns)
    ns = timed_ns(lambda: marshal_tuple(n.REQUEST_DICT), 6000)
    rows["marshal_tuple_whole_row"] = pctiles(ns)

    for name, p in rows.items():
        log(f"  {name}: p50={p['p50_us']:.3f}us ({pct_of_budget(p['p50_us']):.4f}% of 20ms)")
        jsonl_append(RESULTS_JSONL, {"measurement": "followup", "name": name, **p,
                                      "pct_of_20ms_budget_p50": pct_of_budget(p["p50_us"])})

    # 5. reconstructed "optimized" total: bulk-tuple marshal + pooled dispatch
    #    (no per-call output alloc) + bulk readback, everything else unchanged
    log("=== reconstructed optimized-path estimate (sum of bulk-variant p50s) ===")
    accept_p50 = pctiles(timed_ns(lambda: n.accept_kwargs(**n.REQUEST_DICT), 3000))["p50_us"]
    validate_p50 = pctiles(timed_ns(lambda: n.bind_params(n.PARAMS_RAW), 3000))["p50_us"]
    optimized_total_p50 = (
        accept_p50
        + validate_p50
        + rows["marshal_tuple_whole_row"]["p50_us"]
        + rows["njit_call_only_output_buffer_pooled"]["p50_us"]
        + rows["readback_bulk_row_item"]["p50_us"]
        + 0.35  # assemble, measured negligible in main run
    )
    log(f"  accept={accept_p50:.2f}us + validate={validate_p50:.2f}us + "
        f"marshal(tuple)={rows['marshal_tuple_whole_row']['p50_us']:.2f}us + "
        f"dispatch(pooled)={rows['njit_call_only_output_buffer_pooled']['p50_us']:.2f}us + "
        f"readback(bulk)={rows['readback_bulk_row_item']['p50_us']:.2f}us + assemble=0.35us "
        f"= {optimized_total_p50:.2f}us  ({pct_of_budget(optimized_total_p50):.4f}% of 20ms budget)")
    jsonl_append(RESULTS_JSONL, {
        "measurement": "followup_optimized_total_estimate",
        "optimized_total_p50_us": optimized_total_p50,
        "pct_of_20ms_budget": pct_of_budget(optimized_total_p50),
    })
    log("DONE")


if __name__ == "__main__":
    main()
