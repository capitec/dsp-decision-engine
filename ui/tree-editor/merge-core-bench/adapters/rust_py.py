"""Approach 4 adapter (Python): the custom Rust crate `merge_core` (PyO3 extension), arguments crossing as JSON text."""

from merge_core import apply, conflicts, diff, is_related, rebase_orders, reconcile, stale_ops, touched

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
