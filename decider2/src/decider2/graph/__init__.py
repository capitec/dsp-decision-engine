"""Layer 3 (doc 00-BUILD.md §3): the graph model — modules, pipelines,
interface inference and name resolution. "Specified but unvalidated" per
doc 00's build order; this package is the first attempt at building it.

`decider2/__init__.py` imports `module`/`flow`/`step`/`compose`/`Pipeline`
directly from here, so importing this package is also what turns those five
names on. Importing `decider2.graph.module` below is what attaches
`.interface`/`.bind()`/`.relabel()`/etc. to `types.Module` as a side effect
(doc 02 §2 keeps that behaviour out of `types.py` itself).

`make_step`, `raw_interface` and `effective_interface` are internal seams
between the files in this package (`module.py`/`pipeline.py` import them
straight from `.step`/`.interface`) with no caller outside it or its own
tests (over-engineering audit), so they are not re-exported here — import
them from `decider2.graph.step`/`decider2.graph.interface` directly if a
future caller genuinely needs them at that level.
"""
from __future__ import annotations

from decider2.graph.module import module
from decider2.graph.pipeline import Pipeline, compose, flow
from decider2.graph.step import step

__all__ = [
    "module",
    "flow",
    "compose",
    "Pipeline",
    "step",
]
