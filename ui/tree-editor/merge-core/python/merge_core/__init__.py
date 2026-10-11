"""The live-sync ops (diff, apply, stale_ops, conflicts, rebase_orders, ...) of the Rust core; the arguments cross as JSON."""

import json

from ._native import call as _call


def _run(name, *args):
    return json.loads(_call(name, json.dumps(args)))


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
