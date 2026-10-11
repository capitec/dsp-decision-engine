"""Approach 4 adapter (Python): the custom Rust crate `merge_core` (PyO3 extension) built without bigint, arguments converted with pythonize."""

from merge_core import _native


def _run(name, *args):
    return _native.call_direct(name, list(args))


def _pick(ops, indices):
    return [ops[i] for i in indices]


def diff(old, new):
    return _run("diff", old, new)


def apply(doc, ops):
    doc, dropped = _run("apply", doc, ops)
    return doc, _pick(ops, dropped)


def stale_ops(mine, concurrent):
    return _pick(mine, _run("stale_ops", mine, concurrent))


def conflicts(mine, theirs, base):
    return _run("conflicts", mine, theirs, base)


def rebase_orders(mine, base, current):
    return _run("rebase_orders", mine, base, current)


def is_related(a, b):
    return _run("is_related", a, b)


def touched(ops):
    return _run("touched", ops)


def reconcile(current, given):
    return _run("reconcile", current, given)

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
