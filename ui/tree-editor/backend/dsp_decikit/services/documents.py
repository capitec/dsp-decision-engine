"""Pure helpers for project documents (no DB): format checks, validation, defaults."""

import json
import typing as t

from decikit.steps.trees import TreeConfig
from decikit.steps.trees.schema import TreeDocument, detect_format, load_document
from pydantic import TypeAdapter, ValidationError

from dsp_decikit.db.models.projects import ProjectType

FORMATS = {
    ProjectType.TREE: "v3",
    ProjectType.FLAT_RULE: "prioritized_flat_rule",
}


def check_document(data: t.Any, project_type: ProjectType) -> None:
    """Raise ValueError if data is not a supported document of the given project type."""
    if not isinstance(data, dict):
        raise ValueError("project_data must be a JSON object")
    fmt = detect_format(data)
    if fmt == "v3" and "nodes" not in data:
        kind = data.get("type")
        raise ValueError(f"a {kind!r} document is not a tree or flat-rule document" if kind
                         else "the file is not a tree or flat-rule document: it has no nodes or rules")
    if fmt not in FORMATS.values():
        if fmt.startswith("v") and fmt[1:].isdigit():
            raise ValueError(f"{fmt} tree documents are no longer supported")
        raise ValueError(f"Unsupported document format '{fmt}'")
    if fmt != FORMATS[project_type]:
        raise ValueError(
            f"project_type is '{project_type.value}' but project_data is a '{fmt}' document"
        )


def imported_document(data: t.Any) -> t.Any:
    """The tree a file holds: the file itself, or the `tree` of a TreeConfig step document (`{"type": "tree", ...}`),
    whose step settings (path_output, feature_types, ...) the editor does not hold."""
    if isinstance(data, dict) and data.get("type") == "tree" and isinstance(data.get("tree"), dict):
        return data["tree"]
    return data


def detect_project_type(data: t.Any) -> ProjectType:
    """Project type of a supported document; raises ValueError otherwise."""
    fmt = detect_format(data) if isinstance(data, dict) else None
    for project_type, f in FORMATS.items():
        if f == fmt:
            check_document(data, project_type)
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
        params = TreeConfig(name="tree", tree=data).parameters()
    except (ValueError, TypeError) as e:
        return [{"loc": [], "msg": str(e), "type": "build_error"}]
    # The editor lets a parameter's default be any type, but a run converts it to the type
    # of the slot that references it (a threshold is a number), so check that conversion here.
    return [
        {"loc": ["parameters", name], "msg": f"parameter '{name}': default {d.default!r} is not a valid "
                                             f"{getattr(d.annotation, '__name__', d.annotation)}, which is how the tree uses it",
         "type": "param_default"}
        for decls in params.decls.values() for name, d in decls.items()
        if not d.required and not _converts(d.annotation, d.default)
    ]


def _converts(annotation: t.Any, value: t.Any) -> bool:
    try:
        TypeAdapter(annotation).validate_python(value)
    except ValidationError:
        return False
    return True


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
