"""The module registry (doc 08 §1.1, §7.1).

Doc 08 §1.1 draws the line config is allowed to cross: a **reference**
(`{"type": "credit_scorer", ...}`, resolved through a discriminated union in
one native pydantic pass) is admissible; a **pointer**
(`{"module_name": "x", "function_name": "y"}`, resolved with `import_module`
+ `getattr`) is not — it has no declared interface, no schema, nothing to
validate. §7.1 then says where new vocabulary comes from: not a wider
admission policy, an **extension** — "write the thing in code, register it,
reference it by id" — and that this must hold for the framework's own base
types too: "even the base modules were just core extensions" is decider 1's
shape (`decider._ext.TypeDiscriminatedBaseModule` +
`register_graph_module`), and it is the one this module ports.

**A registered type is one class, in one file.** The class is simultaneously
the pydantic schema config validates against *and* the thing that knows how
to become a pipeline element (`.build()`). There is no second place —
a union entry maintained elsewhere, an `isinstance`/`match` dispatching on
`.type` — that a new module kind has to also touch. `register(name)` is the
only step that makes a class reachable from `from_config`, and it is also
the only place `name` (the id config spells) and the class's own declared
`type: Literal[name]` are cross-checked, so the two can never quietly drift
apart.

    from decider2.registry import RegisteredModule, register, from_config

    @register("credit_scorer")
    class CreditScorer(RegisteredModule):
        type: t.Literal["credit_scorer"]
        dti_weight: float = 200.0

        def build(self) -> Module:
            return module(_score, name="credit_scorer").bind(dti_weight=self.dti_weight)

    flow(from_config({"type": "credit_scorer", "dti_weight": 180.0}), ...)

`decision_tree` and `decision_table` (below) register through the exact same
call a third party would use — no privileged path, per the owner's own
framing of decider 1's registry. `tree_module(doc)` / `table_module(doc)`
(`decider2.trees.build` / `decider2.tables.build`) are untouched and remain
the direct Python API for anyone holding a `Tree`/`DecisionTable` already;
`from_config` is the config-facing door onto the same two builders.

Doc 08 §7.1 also flags a production concern this module does not yet
address: sealing the union once startup registration is done, so a
*late* registration raises rather than silently widening what config can
already reference. Nothing here needs that guarantee yet (registration
only ever happens at import time, never mid-request), so it is left as the
doc already leaves it — open, not invented ahead of a real need.
"""
from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from decider2.resolve import suggest_name
from decider2.tables.schema import DecisionTable
from decider2.trees.schema import Tree
from decider2.types import Module

__all__ = [
    "RegisteredModule",
    "register",
    "registered_types",
    "from_config",
    "DecisionTreeConfig",
    "DecisionTableConfig",
]


class RegisteredModule(BaseModel, ABC):
    """Base for anything config may address by registered id.

    `type` is the discriminator every subclass narrows to a single literal
    (checked by `register()`, not here — `register(name)` is where `name`
    and the class meet). `build()` is the behaviour decider 1 put on
    `BaseModule` itself: the class turns *itself* into a pipeline element,
    rather than handing its fields to a dispatcher that knows every kind by
    name.
    """

    type: str

    @abstractmethod
    def build(self) -> Module:
        """Resolve this validated config into an ordinary `types.Module`.

        Whatever `flow(...)` already accepts — nothing about admitting a
        registered type changes what a pipeline is built from (doc 08
        §7's "config references code by registered id" is the whole
        widening; the result is exactly as ordinary as a hand-written
        `module(...)`).
        """


_REGISTRY: dict[str, type[RegisteredModule]] = {}
_adapter: TypeAdapter | None = None


def _declared_tag(cls: type[RegisteredModule]) -> t.Any:
    """The `Literal[...]` `cls` declares for its `type` field, pydantic-
    resolved (so this works regardless of `from __future__ import
    annotations` deferring string evaluation) — or `None` if `type` isn't a
    single-value `Literal` at all.
    """
    field = cls.model_fields.get("type")
    if field is None:
        return None
    annotation = field.annotation
    if t.get_origin(annotation) is not t.Literal:
        return None
    args = t.get_args(annotation)
    return args[0] if len(args) == 1 else None


def register(name: str) -> t.Callable[[type[RegisteredModule]], type[RegisteredModule]]:
    """Class decorator: `@register("decision_tree")` on a `RegisteredModule`
    subclass makes `{"type": "decision_tree", ...}` resolve to it from
    `from_config` — one discriminated union, native pydantic validation,
    per doc 08 §1.1's table.

    The class must declare `type: t.Literal[name]` — restating `name` in
    the annotation is what lets a config document be validated (and its
    `type` field typed) without consulting the registry at all; `register`
    only checks the two were kept honest, it does not infer one from the
    other.
    """

    def decorator(cls: type[RegisteredModule]) -> type[RegisteredModule]:
        if not (isinstance(cls, type) and issubclass(cls, RegisteredModule)):
            raise TypeError(
                f"@register({name!r}) needs a RegisteredModule subclass, got {cls!r}."
            )
        tag = _declared_tag(cls)
        if tag != name:
            raise TypeError(
                f"{cls.__name__} must declare `type: t.Literal[{name!r}]` to be "
                f"registered as {name!r} — found `type` annotated {tag!r}."
            )
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"{name!r} is already registered to {existing.__name__} — "
                f"cannot also register {cls.__name__} (doc 08 §7.1: one id, "
                f"one class; rename one of them)."
            )
        _REGISTRY[name] = cls
        global _adapter
        _adapter = None  # invalidate: rebuilt lazily, on next from_config()
        return cls

    return decorator


def registered_types() -> dict[str, type[RegisteredModule]]:
    """A snapshot of `id -> class` for every currently-registered type.

    Introspection only — nothing in this module reads it back; it exists so
    a caller (a `/module-types` endpoint, a test, a `--list-modules` CLI)
    can answer "what can config reference?" without reaching into a private
    dict.
    """
    return dict(_REGISTRY)


def _get_adapter() -> TypeAdapter:
    global _adapter
    if _adapter is None:
        if not _REGISTRY:
            raise LookupError("no module types are registered — nothing for from_config to resolve")
        classes = tuple(_REGISTRY.values())
        annotation = (
            classes[0]
            if len(classes) == 1
            else t.Annotated[t.Union[classes], Field(discriminator="type")]
        )
        _adapter = TypeAdapter(annotation)
    return _adapter


def from_config(doc: t.Mapping[str, t.Any]) -> Module:
    """Resolve `doc["type"]` to its registered class, validate the rest of
    `doc` against that class's own schema, and build.

    This is doc 08 §1.1's reference row end to end: one discriminated-union
    validation (typed and bounded, `extra="forbid"` wherever a registered
    class sets it, no code executed to get there), then `.build()` returns
    the ordinary `types.Module` a pipeline was always built from — so
    `flow(from_config(doc), ...)` needs nothing new from `flow`.
    """
    try:
        model = _get_adapter().validate_python(doc)
    except ValidationError as exc:
        tag = doc.get("type") if isinstance(doc, t.Mapping) else None
        if isinstance(tag, str) and tag not in _REGISTRY and any(
            e.get("type") in ("union_tag_invalid", "union_tag_not_found")
            for e in exc.errors()
        ):
            known = sorted(_REGISTRY)
            hint = suggest_name(tag, known)
            message = f"unknown module type {tag!r} — registered: {known}."
            if hint:
                message += f" Did you mean {hint!r}?"
            raise LookupError(message) from exc
        raise
    return model.build()


# --- base module types register identically to a user's (doc 08 §7.1) -----


class DecisionTreeConfig(RegisteredModule):
    """`{"type": "decision_tree", "tree": {...v3 tree document...}}`.

    A thin registration wrapper, not a new tree representation: `tree` is
    exactly the `Tree` document `trees/schema.py` already defines and the
    ported migration tests already pin, and `build()` delegates to
    `trees.build.tree_module` — the same function the direct Python API
    calls. `name`/`build_dir`/`params` mirror that function's own keyword
    arguments one for one.
    """

    type: t.Literal["decision_tree"]
    tree: Tree
    name: str | None = None
    build_dir: str | None = None
    params: dict[str, t.Any] | None = None

    def build(self) -> Module:
        from decider2.trees.build import tree_module

        return tree_module(
            self.tree, name=self.name, build_dir=self.build_dir, params=self.params
        ).module


class DecisionTableConfig(RegisteredModule):
    """`{"type": "decision_table", "table": {...decision-table document...}}`.

    Same shape as `DecisionTreeConfig`: `table` is `tables.schema
    .DecisionTable` unchanged, and `build()` delegates to
    `tables.build.table_module`.
    """

    type: t.Literal["decision_table"]
    table: DecisionTable
    name: str | None = None
    build_dir: str | None = None

    def build(self) -> Module:
        from decider2.tables.build import table_module

        return table_module(self.table, name=self.name, build_dir=self.build_dir).module


register("decision_tree")(DecisionTreeConfig)
register("decision_table")(DecisionTableConfig)
