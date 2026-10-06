from __future__ import annotations

from typing import Any, Iterable

import decimal as stdlib
from typing import Annotated

from pydantic import BeforeValidator, ConfigDict, Field, TypeAdapter, create_model

from decider._engine.ir.decls import ParamDecl
from decider._engine.params.bundles import bundle_class
from decider.exceptions import IRError
from decider._engine.ir.decls import nullable
from decider._engine.types import decimal_info, to_raw_input


def type_name(annotation: Any) -> str:
    """A readable name for a param's annotation.

    >>> type_name(float), type_name(list[int])
    ('float', 'list[int]')
    """
    if (info := decimal_info(annotation)) is not None:
        return info.cls.__name__ if info.cls.__name__ == "Currency" and info.units is None else f"Decimal[{info.precision}, {info.scale}]"
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


def _std(value: Any) -> Any:
    return stdlib.Decimal(value.raw).scaleb(-value.scale) if hasattr(value, "raw") and hasattr(value, "scale") else value


def tuned_type(annotation: Any) -> Any:
    """The type a param's value is validated and documented as.

    A Decimal param is a number or string pydantic reads as a `decimal.Decimal`
    (so `ge=` and `le=` work); only the bundle a step receives holds the int64.
    """
    if decimal_info(annotation) is not None:
        checked = Annotated[stdlib.Decimal, BeforeValidator(_std)]
        return checked | None if nullable(annotation) else checked
    return annotation


def _conversion(annotation: Any) -> Any:
    # A Decimal reaches a step as its int64, so no object crosses the dispatcher.
    if (info := decimal_info(annotation)) is not None:
        return lambda v: None if v is None else to_raw_input(v, info.scale)
    return None


def _placeholder(annotation: Any) -> Any:
    # A required param has no default, but the defaults bundle still needs a
    # value of the right type so a compiled kernel sees one signature.
    try:
        return tuned_type(annotation)()
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
        fields = {d.name: (tuned_type(d.annotation), _field(d)) for d in self.decls}
        self.model = create_model(
            path, __config__=ConfigDict(extra="forbid", validate_default=True), **fields
        )
        self.bundle_type = bundle_class(tuple(fields))
        self.conversions = tuple(_conversion(d.annotation) for d in self.decls)
        self.defaults = self.bundle(
            _placeholder(d.annotation) if d.required
            else TypeAdapter(tuned_type(d.annotation)).validate_python(d.default)
            for d in self.decls
        )

    def bundle(self, values: Iterable[Any]) -> tuple:
        """`values`, in declaration order, as the namedtuple the node's function receives."""
        return self.bundle_type(*(v if c is None else c(v) for c, v in zip(self.conversions, values)))
