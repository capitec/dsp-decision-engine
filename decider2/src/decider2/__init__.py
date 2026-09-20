"""decider2's public surface.

Deliberately thin. Doc 03 §1.1's law is that the expensive path doesn't get
used, and the cheapest possible step is a plain function with a `param()`
default.

`param`/`missing_as`/`not_applicable_as` (doc 03 §1, §4.4) live in
`decider2.params`, which imports nothing but `decider2.types` and pydantic.
`module`/`flow`/`step`/`compose`/`Pipeline` (doc 03 §1, §5, §5.3) live in
`decider2.graph`. Both are imported directly and eagerly here: `graph/` (doc
00-BUILD.md Layer 3) is built and imports cleanly now, so the PEP 562 lazy
loader this module used to need — resolving these five names on first access
in case `decider2.graph` did not exist yet — was import-time dead weight
kept for a "not available yet" state that no longer occurs (over-engineering
audit).
"""
from __future__ import annotations

from decider2.graph import Pipeline, compose, flow, module, step
from decider2.params import missing_as, not_applicable_as, param

__all__ = [
    "param", "missing_as", "not_applicable_as",
    "module", "flow", "step", "compose", "Pipeline",
]
