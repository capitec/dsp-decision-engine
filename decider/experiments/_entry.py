"""Import a flow by its `module:qualname` entry, and work out the entry of a live flow."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from decider import Step
from decider.ir import iter_nodes, to_ir


def import_flow(entry: str, tree: Path | None = None):
    """Import `entry` (`"module:qualname"`), from `tree` when given, and call it if it is a zero-argument factory."""
    module, _, qualname = entry.partition(":")
    if not module or not qualname:
        raise ValueError(f"entry {entry!r} must be 'module:qualname'")
    if tree is not None:
        sys.path.insert(0, str(tree))
    try:
        obj = importlib.import_module(module)
    finally:
        if tree is not None:
            sys.path.pop(0)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    # A step is callable too (a function step runs its function), so only a non-step is a factory.
    return obj if isinstance(obj, Step) or not callable(obj) else obj()


def infer_entry(step) -> str:
    """The `module:name` of the module-level variable that is `step`, or `""`.

    Looks in the module of the flow's first user step; never calls anything.
    """
    nodes = [n.origin for n in iter_nodes(to_ir(step))]
    module = next((o.source.partition(":")[0] for o in nodes if not o.source.startswith("decider")), "")
    mod = sys.modules.get(module)
    if mod is None or module == "__main__":
        return ""
    return next((f"{module}:{name}" for name, obj in vars(mod).items() if obj is step), "")
