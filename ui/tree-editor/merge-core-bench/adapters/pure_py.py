"""Reference adapter: the pure Python live ops as they were before any library replaced them (`reference/`)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("reference_live_ops", Path(__file__).parents[1] / "reference/live_ops.py")
_ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ops)

diff, apply, stale_ops, conflicts = _ops.diff, _ops.apply, _ops.stale_ops, _ops.conflicts
rebase_orders, is_related, touched, reconcile = _ops.rebase_orders, _ops.is_related, _ops.touched, _ops.reconcile

# Cases where the pure Python copy is the one that differs from the expected result.
XFAIL = {}


def roundtrip(old, new):
    return apply(old, diff(old, new))[0]


def online(base, first, second):
    # The server's rule: first is applied, then second's ops minus the stale ones, leniently.
    theirs, mine = diff(base, first), diff(base, second)
    doc = apply(base, theirs)[0]
    stale = stale_ops(mine, theirs)
    doc, failed = apply(doc, [op for op in mine if not any(op is s for s in stale)])
    return doc, sorted(touched(stale + failed))


def offline(base, mine, theirs):
    mine_ops, their_ops = diff(base, mine), diff(base, theirs)
    found = conflicts(mine_ops, their_ops, base)
    if found:
        return {"conflicts": found}
    return {"merged": apply(theirs, rebase_orders(mine_ops, base, theirs))[0]}


def load(doc):
    return doc


def export(state):
    return state
