from __future__ import annotations

import json
from typing import Any

from decider.contract.refs import (CONTRACT_VERSION, EdgeRef, FlowDescription, FlowRef, RecordRef, StepRef,
                                   ValueSlotRef)

_JSON_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"

# Every contract reference type, so the generated schema and TypeScript cover
# the ones `FlowDescription` doesn't embed (RecordRef).
_MODELS = (FlowRef, StepRef, EdgeRef, ValueSlotRef, RecordRef)


def _full_schema() -> dict[str, Any]:
    schema = FlowDescription.model_json_schema()
    for model in _MODELS:
        schema["$defs"].setdefault(model.__name__, model.model_json_schema())
    return schema


def json_schema() -> dict[str, Any]:
    """The versioned JSON Schema of the static-flow contract, generated from the Python models.

    `FlowDescription` and every reference type it names appear under `$defs`;
    `contract_version` marks the schema version. FastMCP tools declare their
    input and output with this same schema, so MCP and any other consumer
    share one contract.

    Example::

        schema = json_schema()
        schema["contract_version"]  # "1"
    """
    schema = _full_schema()
    schema["$schema"] = _JSON_SCHEMA_URI
    schema["title"] = "decider static-flow contract"
    schema["contract_version"] = CONTRACT_VERSION
    return schema


def mcp_schema() -> dict[str, Any]:
    """The MCP schema for describing a flow: the same versioned JSON Schema, used as a tool's output.

    An MCP server hosted in `decider` declares its `describe_flow` tool's
    result with this schema, so a client's structural-metadata reads match the
    Python models exactly rather than a hand-written approximation.

    Example::

        mcp_schema()["$defs"]["StepRef"]
    """
    return json_schema()


def typescript() -> str:
    """TypeScript interfaces for the contract, generated from the Python models.

    Emits one `export interface` per reference type plus `FlowDescription`.
    The output is deterministic (fields in declaration order) so it can be
    committed and diffed.

    Example::

        ts = typescript()
        "export interface StepRef" in ts
    """
    schema = _full_schema()
    defs = schema.get("$defs", {})
    names = sorted(defs) + ["FlowDescription"]
    return "\n\n".join(_interface(name, schema if name == "FlowDescription" else defs[name], defs)
                       for name in names) + "\n"


def _interface(name: str, node: dict[str, Any], defs: dict[str, Any]) -> str:
    required = set(node.get("required", []))
    fields = [f"  {field}{'' if field in required else '?'}: {_ts(schema, defs)};"
              for field, schema in node.get("properties", {}).items()]
    return f"export interface {name} {{\n" + "\n".join(fields) + "\n}"


def _ts(node: dict[str, Any], defs: dict[str, Any]) -> str:
    if "$ref" in node:
        return node["$ref"].rsplit("/", 1)[-1]
    if "const" in node:
        return _literal(node["const"])
    if "enum" in node:
        return " | ".join(_literal(v) for v in node["enum"])
    if "anyOf" in node:
        parts = [_ts(branch, defs) for branch in node["anyOf"]]
        nullable = "null" in parts
        parts = [p for p in parts if p != "null"]
        joined = " | ".join(dict.fromkeys(parts)) or "unknown"
        return f"{joined} | null" if nullable else joined
    kind = node.get("type")
    if kind == "string":
        return "string"
    if kind in ("integer", "number"):
        return "number"
    if kind == "boolean":
        return "boolean"
    if kind == "null":
        return "null"
    if kind == "array":
        if "prefixItems" in node:
            return "[" + ", ".join(_ts(item, defs) for item in node["prefixItems"]) + "]"
        items = node.get("items")
        return f"Array<{_ts(items, defs) if items else 'any'}>"
    if kind == "object":
        if "properties" in node:
            return _inline_object(node, defs)
        additional = node.get("additionalProperties")
        if additional in (None, {}, True):
            return "Record<string, any>"
        return f"Record<string, {_ts(additional, defs)}>"
    return "any"


def _inline_object(node: dict[str, Any], defs: dict[str, Any]) -> str:
    required = set(node.get("required", []))
    fields = [f"{field}{'' if field in required else '?'}: {_ts(schema, defs)}"
              for field, schema in node.get("properties", {}).items()]
    return "{ " + "; ".join(fields) + " }"


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)
