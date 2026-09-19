"""Parameter declaration surface — doc 03 §4, especially §4.4.

`param()` is a **thin adapter over pydantic's `Field`** (§4.4, verbatim):

    def param(default, **field_kwargs):
        info = Field(default, **field_kwargs)
        return _carrier_for(type(default))(default, info)

It forwards every keyword to `Field` untouched and owns no validation
vocabulary of its own, so `ge`/`le`/`description`/`alias`/anything pydantic
adds later all work on day one, with pydantic's own error messages and its
JSON Schema for free (doc 08 §6.2's config-UI contract). There is no second
validator to keep in step with the first, because there is no second
validator.

The return value **is a real value** — a subclass of the default's own type
(`float`/`int`/`str`/`list`/`dict`/`tuple`) carrying the `FieldInfo` alongside
— so a step stays directly callable with no decorator, no registration and no
import-order dependence (doc 03 §1.1, §4.4; the first three tests in
tests/test_flagship.py). An earlier draft had composition rewrite
`__defaults__` at import instead; that made a step's behaviour depend on
whether an unrelated line had executed, and was withdrawn (§4.4).

`missing_as()` and `not_applicable_as()` reuse the identical carrier trick for
null-policy tiers 2 and 4 (§1, "Null policy is declared in the signature").
They carry no `FieldInfo` — they own no validation vocabulary either — and
must be distinguishable from `param()` and from *each other*: tier 4 records a
different reason code than tier 2 (the fourth situation: not-applicable is
not missing), and neither is a params field.

Nothing here imports numba or polars — same reason as types.py: harvesting a
signature is pure Python/pydantic and must stay importable without the heavy
stack (doc 02 §2, "the graph is data").
"""
from __future__ import annotations

import inspect
import re
import types as _pytypes
import typing
from typing import Any, Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model

from decider2.types import Input, NullPolicy, ParamDecl, Step

__all__ = [
    "param",
    "missing_as",
    "not_applicable_as",
    "ParamSpec",
    "MissingAs",
    "NotApplicableAs",
    "harvest_signature",
    "harvest_step",
    "parse_docstring",
    "build_params_model",
]


# ---------------------------------------------------------------------------
# The carrier trick
# ---------------------------------------------------------------------------

# doc 03 §4.4: "a subclass of the default's own type" — these six, no others.
_CARRIER_TYPES: tuple[type, ...] = (float, int, str, list, dict, tuple)

# The mutable ones need __init__ to populate; the rest are fully built by
# __new__ (they're immutable, so construction and population are one step).
_MUTABLE_CARRIER_TYPES: tuple[type, ...] = (list, dict)


class ParamSpec:
    """Marks a default as a `param()` carrier.

    Doc 03 §4.4, verbatim: "The harvester finds params by
    `isinstance(default, ParamSpec)`." Any `param()` carrier — regardless of
    which builtin type it subclasses — is an instance of this class.
    """


class MissingAs:
    """Marks a default as a `missing_as()` carrier — null-policy tier 2 (§1)."""


class NotApplicableAs:
    """Marks a default as a `not_applicable_as()` carrier — tier 4 (§1).

    A distinct class from `MissingAs` on purpose: tiers 2 and 4 fill the same
    way but record different reason codes, and the two studies behind §1's
    "fourth situation" independently concluded that collapsing them silently
    approves an incomplete application.
    """


def _carrier_for(kind: type, value: Any) -> Any:
    """Build an instance that is-a `value`'s own type *and* is-a `kind`.

    `bool` and `NoneType` are rejected by the callers below (`param`,
    `missing_as`, `not_applicable_as`) before this is reached, with a message
    naming the alternative — both types are un-subclassable in CPython, so
    there is no carrier to build for them at all.
    """
    value_type = type(value)
    if value_type not in _CARRIER_TYPES:
        raise TypeError(
            f"cannot carry metadata on a {value_type.__name__} default "
            f"({value!r}); param()/missing_as()/not_applicable_as() support "
            f"defaults of type {', '.join(t.__name__ for t in _CARRIER_TYPES)}"
        )

    carrier_cls = type(
        f"_{kind.__name__}_{value_type.__name__}", (value_type, kind), {}
    )
    if value_type in _MUTABLE_CARRIER_TYPES:
        obj = value_type.__new__(carrier_cls)
        value_type.__init__(obj, value)
    else:
        obj = value_type.__new__(carrier_cls, value)
    # The instance already equals `value` (that's the whole trick), but store
    # it explicitly too so downstream unwrapping doesn't depend on knowing
    # which builtin type is under the carrier, or on MRO details of a
    # dynamically-built class.
    obj.__decider2_value__ = value
    return obj


def _unwrap(carrier: Any) -> Any:
    """The plain value underneath a carrier — what ParamDecl.default and
    Input.fill actually store (types.py: "the plain value, unwrapped from
    its carrier")."""
    return carrier.__decider2_value__


def _reject_uncarriable(
    value: Any, who: str, *, bool_alternative: str, none_alternative: str
) -> None:
    """doc 03 §4.4: "Two types cannot carry metadata this way: `bool` and
    `NoneType` are not subclassable in CPython." Shared by `param()`,
    `missing_as()` and `not_applicable_as()`; the alternatives differ per
    caller because the right fix differs — a boolean *param* is an enable
    mask, a boolean *input* just needs the ordinary `bool | None` tier, and
    "fill with None" is a contradiction rather than a missing feature.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"{who}() rejects bool: bool is not subclassable in CPython, so "
            f"it cannot carry metadata this way. {bool_alternative}"
        )
    if value is None:
        raise TypeError(
            f"{who}() rejects None: NoneType is not subclassable in CPython, "
            f"so it cannot carry metadata this way. {none_alternative}"
        )


# ---------------------------------------------------------------------------
# The three public constructors
# ---------------------------------------------------------------------------


def param(default: Any, **field_kwargs: Any) -> Any:
    """A knob declared in a step's signature (doc 03 §4.4).

    Forwards every keyword to pydantic's `Field` verbatim — `ge`, `le`, `gt`,
    `multiple_of`, `description`, `alias`, `examples`, `deprecated`, anything
    pydantic adds later. Returns `default` itself, wearing a `FieldInfo`:

        cap: float = param(48.0, ge=6, le=60)
        cap_by_income_band(term_cap=60.0, min_net_salary=4000.0)  # == 48.0

    so the step above is callable directly, with no decorator and no import
    order dependence — see the first three tests in tests/test_flagship.py.
    """
    _reject_uncarriable(
        default,
        "param",
        bool_alternative="A boolean knob is an enable mask instead (doc 08 §2.1).",
        none_alternative=(
            "An optional param needs an explicit pydantic model (doc 03 §4) — "
            "the signature shortcut can't represent a None default."
        ),
    )
    info = Field(default, **field_kwargs)
    carrier = _carrier_for(ParamSpec, default)
    carrier.field_info = info
    return carrier


def missing_as(value: Any) -> Any:
    """Null-policy tier 2 (doc 03 §1): the framework substitutes `value` at
    extraction, so the step body sees a plain value and there is nothing to
    forget. The carrier trick makes this, too, directly callable:

        bureau_score: float = missing_as(0.0)
    """
    _reject_uncarriable(
        value,
        "missing_as",
        bool_alternative="Declare `bool | None` directly (tier 3, doc 03 §1) instead.",
        none_alternative=(
            "Filling a null with null declares nothing — declare `T | None` "
            "directly (tier 3, doc 03 §1) if the step must see the absence "
            "itself."
        ),
    )
    carrier = _carrier_for(MissingAs, value)
    carrier.fill = value
    return carrier


def not_applicable_as(value: Any) -> Any:
    """Null-policy tier 4 (doc 03 §1, "the fourth situation: not-applicable
    is not missing"): fills exactly like `missing_as()` but is a distinct
    class, so the boundary layer can record a different reason code without
    the step body knowing the difference.

        spouse_income: float = not_applicable_as(0.0)
    """
    _reject_uncarriable(
        value,
        "not_applicable_as",
        bool_alternative="Declare `bool | None` directly (tier 3, doc 03 §1) instead.",
        none_alternative=(
            "Filling a null with null declares nothing — declare `T | None` "
            "directly (tier 3, doc 03 §1) if the step must see the absence "
            "itself."
        ),
    )
    carrier = _carrier_for(NotApplicableAs, value)
    carrier.fill = value
    return carrier


# ---------------------------------------------------------------------------
# Optional-annotation detection (tier 3 — "the step genuinely needs to
# distinguish", doc 03 §1). This tier is keyed on the *annotation*, not a
# default value, so `float | None` and `Optional[float]` both count.
# ---------------------------------------------------------------------------

_UNION_ORIGINS = (typing.Union, _pytypes.UnionType)


def _permits_none(annotation: Any) -> bool:
    if annotation is inspect.Parameter.empty or annotation is None:
        return False
    origin = typing.get_origin(annotation)
    if origin in _UNION_ORIGINS:
        return type(None) in typing.get_args(annotation)
    return False


# ---------------------------------------------------------------------------
# Docstring parsing — doc 03 §1.3
# ---------------------------------------------------------------------------

_IMPLEMENTS_RE = re.compile(r"^Implements:\s*(.+)$")


def parse_docstring(doc: str | None) -> tuple[str | None, str | None]:
    """Split a step's docstring into (description, implements).

    `Implements:` is a trailing line (doc 03 §1.3) — the policy join key doc
    04 §6.3c and doc 00 §6 refer to. It is optional; a rule without it still
    renders (§1.3: "the generated sheet shows 'policy section not
    declared'"), so absence is `None`, not an error.
    """
    if doc is None:
        return None, None

    lines = inspect.cleandoc(doc).splitlines()

    # Walk back from the end past trailing blank lines to find the last
    # non-blank line; if *that* is an `Implements:` line, it's the trailing
    # one §1.3 describes and everything above it is the description.
    i = len(lines) - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i >= 0:
        m = _IMPLEMENTS_RE.match(lines[i].strip())
        if m:
            implements = m.group(1).strip()
            description = "\n".join(lines[:i]).strip()
            return (description or None), implements

    description = "\n".join(lines).strip()
    return (description or None), None


# ---------------------------------------------------------------------------
# Signature harvesting — doc 03 §2's wiring table + §4.4
# ---------------------------------------------------------------------------


def harvest_signature(
    fn: Callable,
) -> tuple[tuple[Input, ...], tuple[ParamDecl, ...], bool, bool]:
    """Turn a plain function's signature into its leaf inputs and params.

    Doc 03 §2's wiring table keys the last two rows on the *default* rather
    than the name — `missing_as()`/`not_applicable_as()`/`param()` defaults —
    and the reserved names `params`/`shared` on the name itself. That is not
    a new axis; it is the existing one, applied consistently.

    Returns `(inputs, params, reads_params, reads_shared)`. `reads_params`/
    `reads_shared` are `Step.reads_params`/`Step.reads_shared` — signature
    has a bare `params` or `shared` argument (doc 03 §4.2).
    """
    try:
        sig = inspect.signature(fn, eval_str=True)
    except (NameError, TypeError):
        # A forward reference that can't resolve yet, or an eval_str-unable
        # callable (e.g. a builtin) — fall back rather than fail harvesting
        # outright; annotations then arrive as-authored (str or object).
        sig = inspect.signature(fn)

    inputs: list[Input] = []
    params: list[ParamDecl] = []
    reads_params = False
    reads_shared = False

    for pname, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"{getattr(fn, '__qualname__', fn)!s}: *args/**kwargs cannot "
                "be a step input — wiring is by name (doc 03 §1, §2)."
            )

        if pname == "params":
            reads_params = True
            continue
        if pname == "shared":
            reads_shared = True
            continue

        raw_annotation = p.annotation
        annotation: Any = raw_annotation if raw_annotation is not inspect.Parameter.empty else Any
        default = p.default

        if isinstance(default, ParamSpec):
            plain = _unwrap(default)
            params.append(
                ParamDecl(
                    name=pname,
                    annotation=annotation if annotation is not Any else type(plain),
                    default=plain,
                    field_info=default.field_info,
                )
            )
            continue

        if isinstance(default, MissingAs):
            plain = _unwrap(default)
            inputs.append(
                Input(
                    name=pname,
                    annotation=annotation if annotation is not Any else type(plain),
                    null_policy=NullPolicy.MISSING_AS,
                    fill=plain,
                )
            )
            continue

        if isinstance(default, NotApplicableAs):
            plain = _unwrap(default)
            inputs.append(
                Input(
                    name=pname,
                    annotation=annotation if annotation is not Any else type(plain),
                    null_policy=NullPolicy.NOT_APPLICABLE_AS,
                    fill=plain,
                )
            )
            continue

        if _permits_none(raw_annotation):
            inputs.append(
                Input(name=pname, annotation=annotation, null_policy=NullPolicy.OPTIONAL)
            )
            continue

        # Tier 1 (required), whether or not an ordinary Python default is
        # also present. Pipeline execution always supplies leaf inputs
        # explicitly (doc 03 §2); a plain default here only affects a
        # direct call, same as it would on any other Python function.
        inputs.append(Input(name=pname, annotation=annotation, null_policy=NullPolicy.REQUIRED))

    return tuple(inputs), tuple(params), reads_params, reads_shared


def _model_name(name: str) -> str:
    """`cap_by_income_band` -> `CapByIncomeBandParams` (doc 03 §4.4's
    worked example, verbatim)."""
    parts = re.split(r"[_\-\s]+", name)
    return "".join(part[:1].upper() + part[1:] for part in parts if part) + "Params"


def build_params_model(name: str, params: Sequence[ParamDecl]) -> type[BaseModel] | None:
    """The generated pydantic model for one module/step instance, namespaced
    by `name` (doc 03 §4.4). `None` when there are nothing to validate — a
    step with no `param()` defaults doesn't get an empty model forced on it.

    Built with `create_model` from the harvested `FieldInfo`s directly, so a
    hand-written model and a harvested one are indistinguishable downstream
    (§4.4: "the same object ... the same validators, the same fixed
    NamedTuple type").

    `extra="forbid"` is doc 03 §10, verbatim: "A misspelled param is a
    **hard error** (`extra="forbid"`), not silence." Pydantic's own default
    is `extra="ignore"`, which would make `params={"cap_by_income_band":
    {"capp": 36.0}}` return the untuned answer with no signal — doc 03
    §2.1's "no error, different decisions ... the worst failure mode the
    design can have". A tuning surface a bank retunes from config cannot
    have a silent one.
    """
    if not params:
        return None
    fields = {p.name: (p.annotation, p.field_info) for p in params}
    return create_model(_model_name(name), __config__=ConfigDict(extra="forbid"), **fields)


def harvest_step(fn: Callable, *, name: str | None = None) -> Step:
    """Build a fully-populated `Step` from a plain function.

    `name` defaults to `fn.__name__` (doc 03 §1: "the function name is the
    output name by default"); an override (`@step(output=...)`, §1) passes
    it explicitly. This is what a bare function used in a pipeline
    expression desugars to (§5.3: "the engine cannot tell this from
    `module(fn, name=...)`") — the intended integration point for
    `module()`/`flow()`.
    """
    inputs, params, reads_params, reads_shared = harvest_signature(fn)
    doc, implements = parse_docstring(fn.__doc__)
    return Step(
        name=name or fn.__name__,
        fn=fn,
        inputs=inputs,
        params=params,
        doc=doc,
        implements=implements,
        reads_params=reads_params,
        reads_shared=reads_shared,
    )
