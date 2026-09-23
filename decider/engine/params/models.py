from __future__ import annotations

from typing import Any, Iterable

from pydantic import ConfigDict, Field, TypeAdapter, create_model

from decider.engine.ir.decls import ParamDecl
from decider.engine.params.bundles import bundle_class


def _type_name(annotation: Any) -> str:
    return annotation.__name__ if isinstance(annotation, type) else repr(annotation)


def record_shared_type(types: dict[str, tuple[Any, str]], decl: ParamDecl, path: str) -> None:
    """Record the type of a shared param in `types`; raise if another node declared it differently.

    Example::

        types = {}
        record_shared_type(types, ParamDecl("base_rate", float, 5.0, shared_key="base_rate"), "term/cap")
    """
    seen_type, seen_path = types.setdefault(decl.shared_key, (decl.annotation, path))
    if seen_type != decl.annotation:
        raise TypeError(
            f"shared param '{decl.shared_key}' is declared {_type_name(seen_type)} by {seen_path} "
            f"and {_type_name(decl.annotation)} by {path}"
        )


def _placeholder(annotation: Any) -> Any:
    # A required param has no default, but the defaults bundle still needs a
    # value of the right type so a compiled kernel sees one signature.
    try:
        return annotation()
    except TypeError:
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

    __slots__ = ("path", "decls", "model", "bundle_type", "defaults", "local_names")

    def __init__(self, path: str, decls: Iterable[ParamDecl]):
        self.path = path
        self.decls = tuple(decls)
        self.local_names = frozenset(d.name for d in self.decls if d.shared_key is None)
        fields = {d.name: (d.annotation, _field(d)) for d in self.decls}
        self.model = create_model(
            path, __config__=ConfigDict(extra="forbid", validate_default=True), **fields
        )
        self.bundle_type = bundle_class(tuple(fields))
        self.defaults = self.bundle_type(*(
            _placeholder(d.annotation) if d.required else TypeAdapter(d.annotation).validate_python(d.default)
            for d in self.decls
        ))
