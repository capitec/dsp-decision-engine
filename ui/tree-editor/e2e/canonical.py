"""A tree document with only what the editor may legitimately add or normalise taken out (data/README.md lists
each and why), so a source and its export compare equal unless something was lost or changed."""

import json
import re

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
EXPORT_STAMP = ("project_id", "name", "project_type", "exported_at", "decikit_version")
OWN_PARTS = ("input_schema", "subtrees")


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None and not (k == "id" and UUID.match(str(v)))}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def canonical(document: dict) -> dict:
    """`document` without the editor's own additions, numbers as floats; compare with `as_lines`."""
    doc = _clean(document)
    metadata = doc.pop("metadata", {})
    if "exported_at" in metadata:  # the export stamps its own metadata, with the project's name as `name`
        metadata = {k: v for k, v in metadata.items() if k not in EXPORT_STAMP}
    if metadata:
        doc["metadata"] = metadata
    # Absent at source the editor fills these in; present, `own_parts` compares them as they are.
    for key in OWN_PARTS:
        doc.pop(key, None)
    doc.pop("parameters_col", None)  # a field decikit never reads
    if doc.get("parameters") == {}:
        del doc["parameters"]
    if doc.get("output", {}).get("type_defs") == {}:
        del doc["output"]["type_defs"]
    for rule in doc.get("rules", []):
        rule.get("meta", {}).pop("position", None)
    if "nodes" in doc:
        if doc.get("type") in ("ui-tree", "v3-tree"):
            del doc["type"]
        if doc.get("formatVersion") == 3:
            del doc["formatVersion"]
        for node in doc["nodes"]:
            node.pop("position", None)
        for edge in doc.get("edges", []):
            if not isinstance(edge["data"]["sourceIndex"], list):
                edge["data"]["sourceIndex"] = [edge["data"]["sourceIndex"]]
    return doc


def as_lines(document: dict) -> list[str]:
    """The canonical form as sorted-key JSON lines: a failing comparison shows the differing lines."""
    return json.dumps(canonical(document), sort_keys=True, indent=1).splitlines()


def own_parts(document: dict) -> dict:
    """The parts the editor fills in when they are missing (`input_schema`, `subtrees`), where the document has them."""
    return {key: document[key] for key in OWN_PARTS if document.get(key)}
