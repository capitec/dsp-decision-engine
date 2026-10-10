import json
from pathlib import Path

import pytest

from dsp_decikit.db.models.projects import ProjectType
from dsp_decikit.services.documents import (
    add_document_schemas,
    check_document,
    default_project_data,
    detect_project_type,
    validation_errors,
)

STORED = json.loads((Path(__file__).parent / "data/stored_documents.json").read_text())


@pytest.mark.parametrize("name", [n for n in STORED if n != "default_flat"])
def test_stored_document_is_valid(name):
    assert validation_errors(STORED[name]) == []


@pytest.mark.parametrize("project_type", list(ProjectType))
def test_default_project_data_matches_type(project_type):
    data = default_project_data(project_type)
    check_document(data, project_type)
    assert detect_project_type(data) == project_type


def test_default_tree_is_valid():
    assert validation_errors(default_project_data(ProjectType.TREE)) == []


def test_flat_rules_without_output_columns_cannot_build():
    errors = validation_errors(default_project_data(ProjectType.FLAT_RULE))
    assert [e["type"] for e in errors] == ["build_error"]
    assert "declares no columns" in errors[0]["msg"]


def test_custom_struct_output_is_valid():
    doc = default_project_data(ProjectType.TREE)
    doc["output"] = {
        "dtypes": [["Action", {"type": "Custom", "type_id": "t1"}]],
        "type_defs": {
            "t1": {
                "name": "Act",
                "type": "struct",
                "definition": {
                    "fields": [["Outcome", "String"]],
                    "display_field": "Outcome",
                    "data": [{"Outcome": "Approve"}],
                },
            }
        },
        "data": [],
        "default": {"Action": {"$key": "Approve"}},
    }
    assert validation_errors(doc) == []


def test_output_type_the_tree_cannot_build_is_reported():
    doc = default_project_data(ProjectType.TREE)
    doc["output"]["dtypes"] = [["Action", {"type": "Custom", "type_id": "missing"}]]
    errors = validation_errors(doc)
    assert errors and "not in type_defs" in errors[0]["msg"]
    json.dumps(errors)


def test_editor_threshold_without_value_is_reported_under_nodes():
    doc = default_project_data(ProjectType.TREE)
    doc["nodes"][0]["data"] = {
        "type": "numerical",
        "variable": "x",
        "threshold": {"name": "hi"},
    }
    errors = validation_errors(doc)
    assert errors and any("nodes" in e["loc"] for e in errors)
    json.dumps(errors)
    check_document(doc, ProjectType.TREE)


def test_type_mismatch_is_rejected():
    with pytest.raises(ValueError, match="flat_rule"):
        check_document(default_project_data(ProjectType.TREE), ProjectType.FLAT_RULE)


def test_v1_document_is_rejected():
    with pytest.raises(ValueError, match="v1 tree documents are no longer supported"):
        check_document({"type": "v1-tree", "root": {}}, ProjectType.TREE)


@pytest.mark.parametrize("data", [[], "x", None])
def test_non_dict_is_rejected(data):
    with pytest.raises(ValueError):
        check_document(data, ProjectType.TREE)
    with pytest.raises(ValueError):
        detect_project_type(data)


def test_openapi_contains_document_types():
    schemas = add_document_schemas({})["components"]["schemas"]
    assert {
        "V3TreeDocument",
        "PrioritizedFlatRuleDocument",
        "ParamRef",
        "TreeDocument",
    } <= set(schemas)
