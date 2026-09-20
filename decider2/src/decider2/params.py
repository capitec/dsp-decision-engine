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

import dis
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
    "attrs_read_from_local",
    "check_params_model_fields_are_read",
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


# `harvest_signature` below builds the same `Input(..., fill=plain)` shape for
# either marker and only needs to know which `NullPolicy` tier it means — one
# dict lookup instead of two near-identical `isinstance` branches.
_NULL_POLICY_BY_MARKER: dict[type, "NullPolicy"] = {
    MissingAs: NullPolicy.MISSING_AS,
    NotApplicableAs: NullPolicy.NOT_APPLICABLE_AS,
}


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


# `missing_as()` and `not_applicable_as()` are byte-identical apart from which
# marker class they attach (tier 2 vs tier 4, §1) — both refuse the same
# `bool | None` alternative text, so that text is written once here and keyed
# off `kind` rather than repeated at each call site.
_NULL_FILL_ALTERNATIVES = dict(
    bool_alternative="Declare `bool | None` directly (tier 3, doc 03 §1) instead.",
    none_alternative=(
        "Filling a null with null declares nothing — declare `T | None` "
        "directly (tier 3, doc 03 §1) if the step must see the absence "
        "itself."
    ),
)


def _fill(kind: type, who: str, value: Any) -> Any:
    """Shared body of `missing_as()`/`not_applicable_as()`: reject an
    uncarriable value, then wrap `value` in `kind`'s carrier with `.fill`
    set — the two functions differ only in which marker class `kind` is."""
    _reject_uncarriable(value, who, **_NULL_FILL_ALTERNATIVES)
    carrier = _carrier_for(kind, value)
    carrier.fill = value
    return carrier


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
    return _fill(MissingAs, "missing_as", value)


def not_applicable_as(value: Any) -> Any:
    """Null-policy tier 4 (doc 03 §1, "the fourth situation: not-applicable
    is not missing"): fills exactly like `missing_as()` but is a distinct
    class, so the boundary layer can record a different reason code without
    the step body knowing the difference.

        spouse_income: float = not_applicable_as(0.0)
    """
    return _fill(NotApplicableAs, "not_applicable_as", value)


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

    # `inspect.cleandoc` already strips leading/trailing blank lines (verified:
    # trailing all-whitespace lines never survive it), so the last element of
    # `lines`, when non-empty, is already the last non-blank line — no
    # walk-back needed to find it.
    lines = inspect.cleandoc(doc).splitlines()
    if lines:
        m = _IMPLEMENTS_RE.match(lines[-1].strip())
        if m:
            implements = m.group(1).strip()
            description = "\n".join(lines[:-1]).strip()
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

        null_policy = next(
            (policy for marker_cls, policy in _NULL_POLICY_BY_MARKER.items()
             if isinstance(default, marker_cls)),
            None,
        )
        if null_policy is not None:
            plain = _unwrap(default)
            inputs.append(
                Input(
                    name=pname,
                    annotation=annotation if annotation is not Any else type(plain),
                    null_policy=null_policy,
                    fill=plain,
                )
            )
            continue

        if _permits_none(raw_annotation):
            inputs.append(
                Input(name=pname, annotation=annotation, null_policy=NullPolicy.OPTIONAL)
            )
            continue

        # Review finding 5 (COLD-READ §1.3, "the Python default is a
        # decoy"): a bare Python default here is ambiguous between a
        # tunable knob and a fill-when-missing value, and doc 03 §4.4's
        # whole argument is that a declaration must be visible — silently
        # treating it as tier 1 (required) makes it LOOK optional at the
        # call site while `score()`/`apply()` still demand it and raise a
        # bare KeyError when it's absent, with nothing in the signature
        # warning that would happen. Rejected at harvest instead, naming
        # the two real spellings: `param()` for a tunable knob (the caller
        # never has to supply it), `missing_as()` for a fill applied only
        # when the value is absent/null. `p.default is inspect.Parameter.
        # empty` (genuinely no default at all) is unaffected — that is
        # ordinary tier 1, required with nothing to be ambiguous about.
        if default is not inspect.Parameter.empty:
            type_name = annotation.__name__ if annotation is not Any and hasattr(annotation, "__name__") else "..."
            raise TypeError(
                f"{getattr(fn, '__qualname__', fn)!s}: parameter '{pname}' "
                f"has a bare Python default ({default!r}). decider2 rejects "
                "this rather than silently treating it as required (doc 03 "
                "§4.4): a bare default is ambiguous between a tunable knob "
                "and a value to fill when the input is missing. Say which "
                f"one you mean — `{pname}: {type_name} = param({default!r})` "
                "for a tunable knob a caller never has to supply, or "
                f"`{pname}: {type_name} = missing_as({default!r})` to fill "
                "an absent/null value."
            )

        # Tier 1 (required): no default at all.
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


# ---------------------------------------------------------------------------
# Bytecode introspection — review findings 6b/6c, checkable at composition
# from the function objects already in hand, with no source file to open
# (contrast finding 6a, `decider2.lint.check_pipeline_file`, which genuinely
# needs one).
# ---------------------------------------------------------------------------


def _touched_local_names(code: Any) -> set[str]:
    """Every local/cell/free-variable name `code` — or a nested code object
    inside it (a lambda, a generator expression; an ordinary list/set/dict
    comprehension is inlined into the enclosing frame since PEP 709 and
    needs no recursion) — ever loads or stores.

    Keyed off a substring of `opname` rather than a fixed, per-Python-
    version opcode list: CPython 3.12+'s specializing interpreter fuses
    adjacent loads into opcodes like `LOAD_FAST_BORROW_LOAD_FAST_BORROW`,
    whose `argval` is a TUPLE of names rather than a single one, so every
    variant of `LOAD`/`STORE` × `FAST`/`DEREF`/`CLOSURE` is covered by
    matching on the fragment rather than enumerating exact opcode names.
    """
    names: set[str] = set()
    for instr in dis.get_instructions(code):
        opname = instr.opname
        if "FAST" not in opname and "DEREF" not in opname and "CLOSURE" not in opname:
            continue
        val = instr.argval
        if isinstance(val, tuple):
            names.update(val)
        elif isinstance(val, str):
            names.add(val)
    for const in code.co_consts:
        if isinstance(const, type(code)):
            names |= _touched_local_names(const)
    return names


def _check_every_input_is_referenced(fn: Callable, inputs: Sequence[Input]) -> None:
    """Review finding 6c: a step parameter never referenced in the body is
    an input the caller must still supply that nothing reads. Checked here,
    once, at harvest — the same place doc 03 §2's wiring table is applied —
    rather than left for a reviewer to notice by hand.
    """
    if not inputs:
        return
    touched = _touched_local_names(fn.__code__)
    for inp in inputs:
        if inp.name not in touched:
            raise ValueError(
                f"{getattr(fn, '__qualname__', fn)!s}: parameter "
                f"'{inp.name}' is declared but never referenced in the "
                "body. A caller must still supply it for nothing to ever "
                "read (doc 03 §1, §2) — drop the parameter, or use it."
            )


def attrs_read_from_local(fn: Callable, local_name: str) -> frozenset[str]:
    """Every attribute name `fn`'s body accesses on its `local_name`
    argument (`params.cap`, `shared.base_rate`, ...) — bytecode scan, not
    source text. `decider2.graph.module` uses this to check review finding
    6b (a `params=` model field no step reads).

    An ordinary LOAD of `local_name` immediately followed by a
    `LOAD_ATTR`/`LOAD_METHOD` names the field actually consulted. A fused
    load instruction (`LOAD_FAST_BORROW_LOAD_FAST_BORROW`, Python 3.13+)
    applies the FOLLOWING attribute access to whichever of its two names was
    pushed LAST — the rightmost element of its `argval` tuple, matching
    source order for a left-to-right-evaluated call argument list such as
    `min(term_cap, params.cap)`.
    """
    attrs: set[str] = set()
    instrs = list(dis.get_instructions(fn.__code__))
    for prev, cur in zip(instrs, instrs[1:]):
        if cur.opname not in ("LOAD_ATTR", "LOAD_METHOD"):
            continue
        if "FAST" not in prev.opname and "DEREF" not in prev.opname:
            continue
        val = prev.argval
        last = val[-1] if isinstance(val, tuple) else val
        if last == local_name:
            attrs.add(cur.argval)
    return frozenset(attrs)


def check_params_model_fields_are_read(module_name: str, steps: Sequence[Step], model: Any) -> None:
    """Review finding 6b: a `params=` model field that no step reads is
    advertised at `GET /params/schema`, accepted by `POST /params`, and
    completely inert — doc 03 §4: "Exactly one canonical location per
    parameter." `decider2.graph.module.module()` calls this for the
    explicit `params=` model path (the harvested path can't have this bug:
    every field it has came FROM a step declaring it, doc 03 §4.4).
    """
    fields = set(model.model_fields)
    if not fields:
        return
    read: set[str] = set()
    for s in steps:
        if s.reads_params:
            read |= attrs_read_from_local(s.fn, "params")
    unread = sorted(fields - read)
    if unread:
        raise ValueError(
            f"module '{module_name}' params= model declares field(s) "
            f"{unread} that no step reads (doc 03 §4: 'exactly one "
            "canonical location per parameter'). Advertised at GET "
            "/params/schema and accepted by POST /params while being "
            "completely inert — drop the field, or read it from a step's "
            "bare `params` argument (`params.<field>`)."
        )


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
    _check_every_input_is_referenced(fn, inputs)
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
