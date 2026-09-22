"""Core types. Single source of truth for every seam in the package.

Nothing here imports numba or polars: these types are what the graph layer,
the boundary layer and the compile layer all agree on, and they must stay
importable without the heavy stack (doc 02 §2 — the graph is data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class NullPolicy(Enum):
    """Doc 03 §1 — four situations, declared in the signature."""

    REQUIRED = "required"              # tier 1: routed by on_missing_input
    MISSING_AS = "missing_as"          # tier 2: filled at extraction
    OPTIONAL = "optional"              # tier 3: step sees `| None`
    NOT_APPLICABLE_AS = "not_applicable_as"  # tier 4: filled, distinct reason code


@dataclass(frozen=True)
class Input:
    """One declared input of a step."""

    name: str
    annotation: Any
    null_policy: NullPolicy = NullPolicy.REQUIRED
    fill: Any = None                   # set for MISSING_AS / NOT_APPLICABLE_AS


@dataclass(frozen=True)
class ParamDecl:
    """A knob harvested from a signature default (doc 03 §4.4)."""

    name: str
    annotation: Any
    default: Any                       # the plain value, unwrapped from its carrier
    field_info: Any                    # pydantic FieldInfo, forwarded verbatim


@dataclass(frozen=True)
class Step:
    """A pure function over scalars. Doc 03 §1."""

    name: str                          # the OUTPUT name; function name by default
    fn: Callable
    inputs: tuple[Input, ...]
    params: tuple[ParamDecl, ...]
    doc: str | None = None             # the docstring — the reviewer-facing description
    implements: str | None = None      # parsed from the docstring's `Implements:` line
    reads_params: bool = False         # signature has a bare `params`
    reads_shared: bool = False         # signature has a bare `shared`
    nogil: bool = False                # @step(nogil=True); authored, never inferred
    packed: bool = False               # fn(args, params) instead of fn(a, b, ...)
    output_annotation: Any = None
    shared_fields: tuple[str, ...] | None = None
    # `shared_fields`: the `shared` keys this `reads_shared` step actually
    # reads, when it can say — a table's row/output steps read a fixed,
    # build-time-known set (`decider2.tables.encode`). The runtime then
    # hands the step a bundle of ONLY those fields (`decider2.runtime.
    # invoke.resolve_params`), so what numba types is the step's own
    # contract, never whatever else the caller merged into `shared=`.
    # Two things depend on that (measured, `decider2.runtime.bundles`):
    # numba's per-specialisation compile cost grows worse than linearly in
    # a bundle's field count — ten 12-key tables merged into one 120-key
    # `shared` had not finished their cold compile after the ten minutes
    # they were given, against 17s once each step saw only its own keys —
    # and a bundle's numba type IS its field set, so a step compiled
    # against the full bundle is recompiled (and its on-disk cache entry
    # missed) every time another table is added beside it. `None` (every
    # hand-written step, whose `shared.x` reads are not declared anywhere)
    # means the whole bundle, exactly as doc 03 §4.2 describes it.
    # The declared return type for a `packed` step, whose `fn` is a generic
    # closure with no `inspect.signature(fn).return_annotation` of its own
    # to read (every packed `fn` has the literal signature `(args, params)`
    # — see `packed` above). `None` means "read it off `fn`'s own signature
    # as usual" (every hand-written, non-packed step). Every caller that
    # used to do `inspect.signature(step.fn).return_annotation` — deciding
    # a materialised array's numpy dtype (doc 00 §2 / doc 05 §9 criterion
    # 4) — reads this first instead: `decider2.compile.driver.
    # step_return_annotation`.
    # `packed`: this Step's `fn` takes exactly two positional arguments —
    # `args` (a tuple, one entry per `inputs`, in order) and `params` (a
    # tuple, one entry per `params`, in order) — instead of one named
    # argument per input/param. Every calling convention in the codebase
    # (interpreted/stepped/fused/fallback) checks this flag and calls
    # accordingly (`decider2.compile.driver`'s packed-call helpers).
    #
    # Exists because a data-shaped interior (a tree, a table, a Branch/Loop
    # construct) has a SET of inputs/params only known once the document is
    # walked — a real, per-instance Python function with that many
    # individually-named parameters would have to be generated as source
    # text to get the names right, which is exactly what these interiors no
    # longer do (doc 08 §3.4). Fixing `fn`'s arity at two lets it be a
    # hand-written closure instead: `Step.inputs`/`.params` still carry
    # every name, type and null policy doc 03 §2's wiring table needs — only
    # the mechanical "how does the call happen" collapses to one shape.


@dataclass(frozen=True)
class Interface:
    """Inferred, then materialised. Doc 03 §5.1 — never declared."""

    inputs: tuple[Input, ...]          # leaves: needed but not produced within
    outputs: tuple[str, ...]           # every value produced
    terminals: tuple[str, ...]         # outputs nothing downstream reads
    params_model: Any                  # a pydantic model class, or None
    shared_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class Module:
    """One or more steps in a shared namespace. Doc 03 §5."""

    name: str
    steps: tuple[Step, ...]
    params_model: Any = None
    relabel_reads: Mapping[str, str] = field(default_factory=dict)
    relabel_writes: Mapping[str, str] = field(default_factory=dict)
    bound: Mapping[str, Any] = field(default_factory=dict)   # .bind()
    contract: str | None = None

    def __or__(self, other):
        from decider2.graph.pipeline import compose
        return compose(self, other)


class Decision(Enum):
    """Where a boundary null routes to, instead of raising. Doc 03 §1."""

    REFER = "refer"
    DECLINE = "decline"
    RAISE = "raise"


@dataclass(frozen=True)
class MissingInputPolicy:
    """Doc 03 §1 — a null must be able to produce a decision, not an exception."""

    default: Decision = Decision.REFER
    reason: int = 4101
    raise_for: tuple[str, ...] = ()


@dataclass(frozen=True)
class Emit:
    """One requested output column. Doc 03 §7. `at` qualifies a rewritten name."""

    name: str
    at: str | None = None              # producing module, or "*" for every version
