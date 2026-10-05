from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Mapping

from pydantic import TypeAdapter

from decider._engine.ir.decls import ParamDecl
from decider._engine.ir.nodes import CallNode, IRNode, ScatterGatherNode
from decider._engine.params.models import NodeParams, type_name
from decider._engine.params.validate import Status, Validation, check_namespaces, validate_node
from decider.exceptions import ParamsError


_JSON_TYPES = {"int": "integer", "float": "number", "bool": "boolean"}


def _info(decl: ParamDecl) -> dict[str, Any]:
    table = decl.schema is not None
    info: dict[str, Any] = {"type": "table", "schema": dict(decl.schema)} if table else {"type": type_name(decl.annotation)}
    if decl.required:
        info["required"] = True
    else:
        info["default"] = decl.default
    for constraint in () if table else getattr(decl.field_info, "metadata", ()):
        if dataclasses.is_dataclass(constraint):
            info.update(dataclasses.asdict(constraint))
    return info


def _json_schema(decl: ParamDecl) -> dict[str, Any]:
    if decl.schema is not None:
        columns = {c: {"type": _JSON_TYPES[d]} if d in _JSON_TYPES else {} for c, d in decl.schema}
        items = {"type": "object", "properties": columns, "required": list(columns), "additionalProperties": False}
        return {"type": "array", "items": items, **({} if decl.required else {"default": decl.default})}
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
        schema.validate(doc).check()          # raises ParamsError listing every problem in `doc`
    """

    def __init__(self, decls: dict[str, dict[str, ParamDecl]], used_by: dict[str, list[str]],
                 nodes: dict[str, tuple[ParamDecl, ...]] | None = None):
        super().__init__()
        self.decls = decls
        self._nodes = nodes or {}
        for path, params in decls.items():
            self[path] = {name: _info(d) for name, d in params.items()}
        for key, paths in used_by.items():
            self["shared"][key]["used_by"] = paths

    def defaults(self) -> dict[str, Any]:
        """A params document holding every default, nested by path.

        Required params are left out, except a required table, which shows
        as `[]` (no rows) so the document says where its rows go.
        """
        doc: dict[str, Any] = {}
        for path, params in self.decls.items():
            values = {name: [] if d.required else d.default
                      for name, d in params.items() if not d.required or d.schema is not None}
            if values:
                target = doc
                for part in path.split("/"):
                    target = target.setdefault(part, {})
                target.update(values)
        return doc

    def validate(self, doc: Mapping[str, Any]) -> Validation:
        """Check a params document without running anything.

        Returns a `Validation` over the whole document: `errors` lists every
        problem (unknown step or param, wrong type, out of bounds, missing
        required), `warnings` the values replaced by their default, and `bundle`
        is `None`. `.check()` raises `ParamsError` when there are errors.

        Example::

            problems = pipeline.parameters().validate({"term": {"cap_by_income": {"cap": 99.0}}})
            problems.errors   # ("term/cap_by_income: param 'cap': ... (got 99.0)",)
        """
        nodes = {path: NodeParams(path, decls) for path, decls in self._nodes.items()}
        try:
            check_namespaces(doc, nodes)
        except ParamsError as e:
            return Validation(Status.INVALID, None, (str(e),))
        results = [validate_node(n, doc) for n in nodes.values()]
        errors = tuple(e for r in results for e in r.errors)
        warnings = tuple(w for r in results for w in r.warnings)
        return Validation(Status.INVALID if errors else Status.OK, None, errors, warnings)

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
    """Collect the params of every `CallNode` in an IR tree, and a `ScatterGatherNode`'s hoisted child params.

    Example::

        parameters(to_ir(pipeline))
    """
    shared: dict[str, ParamDecl] = {}
    used_by: dict[str, list[str]] = {}
    local: dict[str, dict[str, ParamDecl]] = {}
    nodes: dict[str, tuple[ParamDecl, ...]] = {}
    for node in _param_nodes(root):
        if isinstance(node, ScatterGatherNode):
            decls, path = node.params, node.origin.path
        elif isinstance(node, CallNode):
            decls, path = node.params, node.origin.path
        else:
            continue
        if decls:
            nodes[path] = tuple(decls)
        for d in decls:
            if d.shared_key is None:
                local.setdefault(path, {})[d.name] = d
            else:
                # Shared defaults may differ per node; the document shows the first.
                shared.setdefault(d.shared_key, d)
                used_by.setdefault(d.shared_key, []).append(path)
    return ParamsSchema({"shared": shared, **local} if shared else local, used_by, nodes)


def _param_nodes(node: IRNode):
    """Every param-bearing node of an IR tree; a `ScatterGatherNode` is opaque (its child's params are hoisted)."""
    yield node
    if isinstance(node, ScatterGatherNode):
        if node.accumulate is not None:
            yield from _param_nodes(node.accumulate)
        return
    for child in node.children():
        yield from _param_nodes(child)
