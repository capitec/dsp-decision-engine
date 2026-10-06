from __future__ import annotations

import importlib
import sys
from pathlib import Path

from decider.steps._base import Step


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
