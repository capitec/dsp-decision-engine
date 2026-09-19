"""decider2's public surface.

Deliberately thin. Doc 03 §1.1's law is that the expensive path doesn't get
used, and the cheapest possible step is a plain function with a `param()`
default — importing `decider2` for that must not require the rest of the
framework (`compile/`, `runtime/`, `observe/`) to exist, let alone import
cleanly.

`param`/`missing_as`/`not_applicable_as` (doc 03 §1, §4.4) live in
`decider2.params`, which imports nothing but `decider2.types` and pydantic,
so they're loaded eagerly. `module`/`flow`/`step`/`compose`/`Pipeline` (doc
03 §1, §5, §5.3) live in `decider2.graph`, built separately (doc 00-BUILD.md,
Layer 3) and possibly not present yet — so they're resolved lazily (PEP 562),
on first access rather than at package-import time. That keeps
`from decider2 import param` working even while `graph/` is still being
built, and turns a missing `graph/` into an error only where one of these
names is actually used.
"""
from __future__ import annotations

from decider2.params import missing_as, not_applicable_as, param

__all__ = [
    "param", "missing_as", "not_applicable_as",
    "module", "flow", "step", "compose", "Pipeline",
]

_LAZY = {"module", "flow", "step", "compose", "Pipeline"}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from decider2 import graph
    except ImportError as exc:
        raise ImportError(
            f"decider2.{name} is not available yet: decider2.graph does not "
            "exist or fails to import (doc 00-BUILD.md Layer 3 — graph/ is "
            "built separately)."
        ) from exc
    try:
        return getattr(graph, name)
    except AttributeError as exc:
        raise ImportError(
            f"decider2.{name} is not available yet: decider2.graph does not "
            f"export {name!r} (doc 00-BUILD.md Layer 3 — graph/ is built "
            "separately)."
        ) from exc
