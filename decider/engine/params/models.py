from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from pydantic import ConfigDict, Field, TypeAdapter, create_model

from decider.engine.ir.decls import ParamDecl
from decider.engine.params.bundles import bundle_class
from decider.exceptions import IRError
from decider.types import is_raw, raw_base, string_code


def type_name(annotation: Any) -> str:
    """A readable name for a param's annotation.

    >>> type_name(float), type_name(list[int])
    ('float', 'list[int]')
    """
    return annotation.__name__ if isinstance(annotation, type) else repr(annotation)


def record_shared_type(types: dict[str, tuple[Any, str]], decl: ParamDecl, path: str) -> None:
    """Record the type of a shared param in `types`; raise if another node declared it differently.

    Example::

        types = {}
        record_shared_type(types, ParamDecl("base_rate", float, 5.0, shared_key="base_rate"), "term/cap")
    """
    seen_type, seen_path = types.setdefault(decl.shared_key, (decl.annotation, path))
    if seen_type != decl.annotation:
        raise IRError(
            f"shared param '{decl.shared_key}' is declared {type_name(seen_type)} by {seen_path} "
            f"and {type_name(decl.annotation)} by {path}"
        )


def _tuned_type(annotation: Any) -> Any:
    # A `Raw[str]` param is tuned, validated and documented as a plain string; only the
    # bundle a step receives holds the code.
    return raw_base(annotation) if is_raw(annotation) else annotation


def _as_code(value: str) -> np.int32:
    # int32 whatever the value, so retuning the string never moves the kernel's signature.
    return np.int32(string_code(value))


def _conversion(annotation: Any) -> Any:
    # A `Raw[str]` param reaches a step as the same int32 code a `Raw[str]` input does.
    return _as_code if is_raw(annotation) and raw_base(annotation) is str else None


def _placeholder(annotation: Any) -> Any:
    # A required param has no default, but the defaults bundle still needs a
    # value of the right type so a compiled kernel sees one signature.
    try:
        return _tuned_type(annotation)()
    except (TypeError, ValueError):
        return None


def _field(decl: ParamDecl) -> Any:
    if decl.field_info is not None:
        return decl.field_info
    return Field() if decl.required else Field(decl.default)


class NodeParams:
    """The params of one node: its pydantic model, its bundle class and its bundle of defaults.

    The model covers the node's local params and the shared keys it uses, each
    under the param's own name, with the node's own bounds and defaults.

    Example::

        node = NodeParams("term/cap_by_income", params)
        node.defaults.cap  # 48.0
    """

    __slots__ = ("path", "decls", "model", "bundle_type", "conversions", "defaults", "local_names")

    def __init__(self, path: str, decls: Iterable[ParamDecl]):
        self.path = path
        self.decls = tuple(decls)
        self.local_names = frozenset(d.name for d in self.decls if d.shared_key is None)
        fields = {d.name: (_tuned_type(d.annotation), _field(d)) for d in self.decls}
        self.model = create_model(
            path, __config__=ConfigDict(extra="forbid", validate_default=True), **fields
        )
        self.bundle_type = bundle_class(tuple(fields))
        self.conversions = tuple(_conversion(d.annotation) for d in self.decls)
        self.defaults = self.bundle(
            _placeholder(d.annotation) if d.required
            else TypeAdapter(_tuned_type(d.annotation)).validate_python(d.default)
            for d in self.decls
        )

    def bundle(self, values: Iterable[Any]) -> tuple:
        """`values`, in declaration order, as the namedtuple the node's function receives."""
        return self.bundle_type(*(v if c is None else c(v) for c, v in zip(self.conversions, values)))
