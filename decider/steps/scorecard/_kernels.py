"""The functions a scorecard's IR calls: plain numeric Python that numba also compiles."""
from __future__ import annotations

from numba.extending import register_jitable


@register_jitable
def _points(x, nb, vals, items, item_bin):
    # vals: nb lower bounds, nb upper bounds, nb bound-bin points, one points
    # per values bin, then the default points. A NaN bound is no bound.
    if x is None:
        return vals[-1]
    for k in range(len(items)):
        item = items[k]
        # NaN matches a NaN item, as polars' is_in does.
        if x == item or (x != x and item != item):
            return vals[3 * nb + item_bin[k]]
    for i in range(nb):
        lo, hi = vals[i], vals[nb + i]
        # `not x <= lo` rather than `x > lo`: a NaN input clears every lower bound, as in polars.
        if not x <= lo and (hi != hi or x <= hi):
            return vals[2 * nb + i]
    return vals[-1]


def score_bins(row, params, consts):
    nb, vals, items, item_bin = consts
    return (_points(row[0], nb, vals, items, item_bin),)


def score_bins_tuned(row, params, consts):
    # A separate function because numba can't index an empty params tuple.
    nb, vals, items, item_bin, slots = consts
    vals = vals.copy()
    for j in range(len(slots)):
        if slots[j] >= 0:
            vals[j] = params[slots[j]]
    return (_points(row[0], nb, vals, items, item_bin),)


def adjust(score, scale, offset):
    return score * scale + offset


def constant(score):
    return score


def total(row, params, consts):
    t = 0.0
    for v in row:
        t += v
    return (t,)
