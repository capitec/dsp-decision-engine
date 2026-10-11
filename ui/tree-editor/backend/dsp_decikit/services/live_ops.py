"""Live-sync ops on tree documents: diff, lenient apply, and the online (stale) and offline (conflict) checks.

The Rust crate `ui/tree-editor/merge-core` is the one implementation; the browser runs the same code as WebAssembly.
`apply` and `stale_ops` return the caller's own op objects.

    >>> diff({"rules": [{"rule": {"id": "a"}}]}, {"rules": [{"rule": {"id": "a", "x": 1}}]})
    [{'op': 'add', 'path': '/rules/a/rule/x', 'value': 1}]
"""

from merge_core import apply, conflicts, diff, is_related, rebase_orders, reconcile, stale_ops, touched

__all__ = ["apply", "conflicts", "diff", "is_related", "rebase_orders", "reconcile", "stale_ops", "touched"]
