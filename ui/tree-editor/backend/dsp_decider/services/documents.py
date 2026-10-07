"""Pure helpers for project documents (no DB): format checks, validation, defaults."""

import json
import typing as t

from decider.steps.trees import TreeConfig
from decider.steps.trees.schema import TreeDocument, detect_format, load_document
from pydantic import TypeAdapter, ValidationError

from dsp_decider.db.models.projects import ProjectType

FORMATS = {
    ProjectType.TREE: "v3",
    ProjectType.FLAT_RULE: "prioritized_flat_rule",
}


def check_document(data: t.Any, project_type: ProjectType) -> None:
    """Raise ValueError if data is not a supported document of the given project type."""
    if not isinstance(data, dict):
        raise ValueError("project_data must be a JSON object")
    fmt = detect_format(data)
    if fmt not in FORMATS.values():
        if fmt.startswith("v") and fmt[1:].isdigit():
            raise ValueError(f"{fmt} tree documents are no longer supported")
        raise ValueError(f"Unsupported document format '{fmt}'")
    if fmt != FORMATS[project_type]:
        raise ValueError(
            f"project_type is '{project_type.value}' but project_data is a '{fmt}' document"
        )


def detect_project_type(data: t.Any) -> ProjectType:
    """Project type of a supported document; raises ValueError otherwise."""
    fmt = detect_format(data) if isinstance(data, dict) else None
    for project_type, f in FORMATS.items():
        if f == fmt:
            return project_type
    check_document(data, ProjectType.TREE)  # always raises here
    raise AssertionError


def validation_errors(data: t.Dict[str, t.Any]) -> t.List[t.Dict[str, t.Any]]:
    try:
        load_document(data)
    except ValidationError as e:
        return json.loads(e.json(include_url=False))
    # A valid document can still fail to build (unsupported output types, out-of-range
    # result_idx, ...); building takes milliseconds and catches what running would.
    try:
        TreeConfig(name="tree", tree=data).parameters()
    except (ValueError, TypeError) as e:
        return [{"loc": [], "msg": str(e), "type": "build_error"}]
    return []


def default_project_data(project_type: ProjectType) -> t.Dict[str, t.Any]:
    if project_type == ProjectType.FLAT_RULE:
        return {
            "type": "prioritized_flat_rule",
            "parameters": {},
            "output": {"dtypes": [], "type_defs": {}, "data": [], "default": None},
            "rules": [],
            "mode": "first_match",
        }
    return {
        "type": "ui-tree",
        "formatVersion": 3,
        "parameters": {},
        "output": {
            "dtypes": [["Action", "string"], ["Description", "string"]],
            "type_defs": {},
            "data": [
                {
                    "Action": "Alert",
                    "Description": "A rule suggests that there is an alert",
                },
                {
                    "Action": "Block",
                    "Description": "This is more serious and must be blocked",
                },
            ],
            "default": {"Action": "Allow", "Description": "Default Action"},
        },
        "nodes": [
            {
                "id": "node-1",
                "position": {"x": 100.0, "y": 100.0},
                "data": {"type": "leaf", "result_idx": -1},
            }
        ],
        "edges": [],
        "subtrees": [],
    }


def add_document_schemas(openapi: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    """Add the document models to an OpenAPI spec so clients can generate types for them."""
    doc = TypeAdapter(TreeDocument).json_schema(
        ref_template="#/components/schemas/{model}"
    )
    schemas = openapi.setdefault("components", {}).setdefault("schemas", {})
    schemas.update(doc.pop("$defs"))
    schemas["TreeDocument"] = doc
    return openapi
