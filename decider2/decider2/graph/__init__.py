"""Layer 3 (doc 00-BUILD.md §3): the graph model — modules, pipelines,
interface inference and name resolution. "Specified but unvalidated" per
doc 00's build order; this package is the first attempt at building it.

`decider2/__init__.py` resolves `module`/`flow` lazily from here (PEP 562),
so importing this package is also what turns those two names on. Importing
`decider2.graph.module` below is what attaches `.interface`/`.bind()`/
`.relabel()`/etc. to `types.Module` as a side effect (doc 02 §2 keeps that
behaviour out of `types.py` itself).
"""
from __future__ import annotations

from decider2.graph.interface import effective_interface, raw_interface
from decider2.graph.module import module
from decider2.graph.pipeline import Pipeline, compose, flow
from decider2.graph.step import make_step, step

__all__ = [
    "module",
    "flow",
    "compose",
    "Pipeline",
    "step",
    "make_step",
    "raw_interface",
    "effective_interface",
]
