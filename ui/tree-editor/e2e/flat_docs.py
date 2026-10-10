"""Flat-rule documents and decikit runs shared by the flat-rule editor tests."""

import polars as pl
from decikit import TreeConfig

from flat_rules_ui import open_flat_rules
from seed import create_project
from test_flat_rule_params import flat_document

APPROVE = {"type": "leaf", "result_idx": 0}
DEFAULT = {"type": "leaf", "result_idx": -1}
SCHEMA = [["amount", "Float64"], ["count", "Int64"], ["flag", "Boolean"], ["band", "String"]]


def document(*rules: dict, parameters: dict | None = None, mode: str = "first_match", output: dict | None = None,
             schema: list | None = None) -> dict:
    """`flat_document` with a wider input schema (amount, count, flag, band) and an optional output block."""
    doc = flat_document(*rules, parameters=parameters, mode=mode)
    doc["input_schema"] = schema or SCHEMA
    if output:
        doc["output"] = output
    return doc


def run_frame(doc: dict, frame: dict, params: dict | None = None) -> list[dict]:
    """Run the exported `doc` in decikit over `frame` (column -> values); returns the output rows only."""
    tree = TreeConfig.load({"type": "tree", "name": "rules", "tree": doc})
    out = tree.run(pl.DataFrame(frame), **({"params": {"rules": params}} if params else {}))
    return out.drop(list(frame)).to_dicts()


def decisions_for(doc: dict, frame: dict, params: dict | None = None) -> list[str]:
    return [row["decision"] for row in run_frame(doc, frame, params)]


def open_doc(page, *rules: dict, name: str = "Rules", **kwargs) -> str:
    """Seed a project holding `document(*rules, **kwargs)` through the API, open it in the editor; returns its id."""
    project = create_project(page, name, document(*rules, **kwargs))["id"]
    open_flat_rules(page, project)
    return project

OUTPUT3 = {
    "dtypes": [["decision", "String"]],
    "type_defs": {},
    "data": [{"decision": "approve"}, {"decision": "review"}],
    "default": {"decision": "decline"},
}
