import json
from pathlib import Path

import pytest
from decikit.steps.trees.schema import load_document

from dsp_decikit.db.models.projects import ProjectType
from dsp_decikit.services.documents import (
    add_document_schemas,
    check_document,
    default_project_data,
    detect_project_type,
    imported_document,
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


def _flat_with_param(default):
    doc = default_project_data(ProjectType.FLAT_RULE)
    doc["output"]["dtypes"] = [["decision", "String"]]
    doc["parameters"] = {"limit": {"type": "String", "default_value": default}}
    doc["rules"] = [{"rule": {"type": "unary", "then": {"type": "leaf", "result_idx": -1},
                              "condition": {"op": "<=", "feature": "amount", "threshold": {"param": "limit"}}}}]
    return doc


def test_parameter_default_the_tree_cannot_use_is_reported_under_parameters():
    errors = validation_errors(_flat_with_param(""))
    assert [e["loc"] for e in errors] == [["parameters", "limit"]]
    assert "limit" in errors[0]["msg"] and "float" in errors[0]["msg"]


@pytest.mark.parametrize("default", [1000, 999.5, "1000", None])
def test_parameter_default_the_tree_can_use_is_valid(default):
    assert validation_errors(_flat_with_param(default)) == []


def test_a_step_document_imports_as_its_tree():
    tree = default_project_data(ProjectType.TREE)
    step = {"type": "tree", "name": "campaign", "path_output": "path", "tree": tree}
    assert imported_document(step) == tree
    assert imported_document(tree) == tree


@pytest.mark.parametrize("data", [{"type": "rule_set", "rules": {}}, {"a": 1}])
def test_a_document_that_is_no_tree_is_refused_naming_what_it_is(data):
    with pytest.raises(ValueError, match="not a tree") as e:
        detect_project_type(data)
    assert data.get("type", "not a tree") in str(e.value)


def test_a_one_rule_flat_document_imports_as_a_prioritized_set_that_runs_the_same():
    rule = {"meta": {"name": "only"}, "rule": {"type": "leaf", "result_idx": 0}}
    output = {"dtypes": [["d", "String"]], "data": [{"d": "x"}]}
    flat = {"type": "flat_rule", "rule": rule, "output": output}
    imported = imported_document(flat)
    assert detect_project_type(imported) == ProjectType.FLAT_RULE
    assert load_document(imported).to_tree().model_dump() == load_document(flat).to_tree().model_dump()
