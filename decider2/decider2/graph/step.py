"""A Step: declared inputs, output name and params (doc 02 §6's home for
"a Step"), harvested from a plain function's signature.

The harvesting itself — walking `inspect.signature`, applying doc 03 §2's
wiring table, splitting the docstring into description + `Implements:`
(§1.3) — already lives in `decider2.params.harvest_step`. It has to: a
module's params model (`decider2.params.build_params_model`) is assembled
from the very `ParamDecl.field_info` values that harvesting produces, so
there is exactly one place that computation happens, not two that must be
kept in sync (the same "one canonical location" rule doc 03 §4 states for a
tunable's value). This module adds the one piece of the authoring surface
that is genuinely `graph`'s: the optional `@step` decorator (doc 03 §1) that
overrides the derived output name, and `make_step`, the normalising entry
point `module()`/`flow()` call for every element they're given.
"""
from __future__ import annotations

from typing import Callable

from decider2.params import harvest_step
from decider2.types import Step

# Doc 03 §1: "`@step` is only needed when you want to override defaults" —
# and the only override that exists is the output name. Stashing it as a
# plain attribute (rather than wrapping `fn`) is what keeps the decorated
# function directly callable with no indirection (doc 00-BUILD.md §6: "a
# step is directly callable with no decorator, no pipeline and no import
# order").
_OUTPUT_OVERRIDE = "__decider2_output__"


def step(output: str | None = None) -> Callable[[Callable], Callable]:
    """`@step(output=...)` — the only decorator argument doc 03 §1 defines.

    `implements=` is deliberately not an option here: doc 03 §1.3 explains
    why putting `Implements:` on the decorator would make the decorator
    mandatory on every policy rule, directly against both §1's "only needed
    to override defaults" and §1.1's one-artefact rule. The docstring's
    trailing `Implements:` line (parsed by `decider2.params.parse_docstring`)
    is the only spelling.
    """

    def decorator(fn: Callable) -> Callable:
        setattr(fn, _OUTPUT_OVERRIDE, output)
        return fn

    return decorator


def make_step(element: Callable | Step) -> Step:
    """Normalise one pipeline element into a `Step`.

    Doc 03 §5.3: a bare function used in a pipeline expression *is* a
    module — "the engine cannot tell this from `module(fn, name=...)`,
    which is what it desugars to." This is that desugaring's first half:
    turning the bare callable into the `Step` a single-step `Module` wraps.
    Idempotent on an already-built `Step` (e.g. a step reused across two
    `module(...)` calls), so callers never need to branch on which they have.
    """
    if isinstance(element, Step):
        return element
    if not callable(element):
        raise TypeError(
            f"{element!r} is not a step: it is neither a Step nor callable "
            "(doc 03 §1 — a step is a plain, pure Python function)."
        )
    override = getattr(element, _OUTPUT_OVERRIDE, None)
    return harvest_step(element, name=override)
