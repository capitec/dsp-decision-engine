from __future__ import annotations

import dataclasses
from typing import Annotated, Any

from pydantic import TypeAdapter

from decider.engine.ir.decls import ParamDecl
from decider.engine.ir.nodes import CallNode, IRNode, iter_nodes
from decider.engine.params.models import type_name


def _info(decl: ParamDecl) -> dict[str, Any]:
    if decl.schema is not None:
        return {"type": "table", "schema": dict(decl.schema)}
    info: dict[str, Any] = {"type": type_name(decl.annotation)}
    if decl.required:
        info["required"] = True
    else:
        info["default"] = decl.default
    for constraint in getattr(decl.field_info, "metadata", ()):
        if dataclasses.is_dataclass(constraint):
            info.update(dataclasses.asdict(constraint))
    return info


def _json_schema(decl: ParamDecl) -> dict[str, Any]:
    if decl.schema is not None:
        return {"type": "array", "items": {"type": "object", "required": [column for column, _ in decl.schema]}}
    # ponytail: types that need $defs (enums, models) keep dangling refs; hoist $defs when such a param appears.
    if decl.field_info is not None:
        return TypeAdapter(Annotated[decl.annotation, decl.field_info]).json_schema()
    schema = TypeAdapter(decl.annotation).json_schema()
    return schema if decl.required else {**schema, "default": decl.default}


class ParamsSchema(dict):
    """Every param a step needs, keyed by node path, plus shared params under `"shared"`.

    Example::

        schema = pipeline.parameters()
        schema["term/cap_by_income"]["cap"]   # {"type": "float", "default": 48.0, "ge": 6, "le": 60}
        schema["shared"]["min_ratio"]["used_by"]  # ["affordability/affordable"]
        schema.defaults()                     # a params document of every default
        schema.json_schema()                  # JSON Schema of the params document
    """

    def __init__(self, decls: dict[str, dict[str, ParamDecl]], used_by: dict[str, list[str]]):
        super().__init__()
        self.decls = decls
        for path, params in decls.items():
            self[path] = {name: _info(d) for name, d in params.items()}
        for key, paths in used_by.items():
            self["shared"][key]["used_by"] = paths

    def defaults(self) -> dict[str, Any]:
        """A params document holding every default, nested by path. Required params are left out."""
        doc: dict[str, Any] = {}
        for path, params in self.decls.items():
            values = {name: d.default for name, d in params.items() if not d.required}
            if values:
                target = doc
                for part in path.split("/"):
                    target = target.setdefault(part, {})
                target.update(values)
        return doc

    def json_schema(self) -> dict[str, Any]:
        """JSON Schema of the params document, for rendering forms. Shared params use their first declaration."""
        schema: dict[str, Any] = {"type": "object", "properties": {}}
        for path, params in self.decls.items():
            target = schema
            for part in path.split("/"):
                target = target["properties"].setdefault(part, {"type": "object", "properties": {}})
            for name, d in params.items():
                target["properties"][name] = _json_schema(d)
                if d.required:
                    target.setdefault("required", []).append(name)
        return schema


def parameters(root: IRNode) -> ParamsSchema:
    """Collect the params of every `CallNode` in an IR tree.

    Example::

        parameters(engine.to_ir(pipeline))
    """
    shared: dict[str, ParamDecl] = {}
    used_by: dict[str, list[str]] = {}
    local: dict[str, dict[str, ParamDecl]] = {}
    for node in iter_nodes(root):
        if not isinstance(node, CallNode):
            continue
        for d in node.params:
            if d.shared_key is None:
                local.setdefault(node.origin.path, {})[d.name] = d
            else:
                # Shared defaults may differ per node; the document shows the first.
                shared.setdefault(d.shared_key, d)
                used_by.setdefault(d.shared_key, []).append(node.origin.path)
    return ParamsSchema({"shared": shared, **local} if shared else local, used_by)
