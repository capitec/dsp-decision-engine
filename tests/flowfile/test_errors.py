"""Flow-file errors: typo'd keys, unresolved names, cycles and ambiguous references."""
import json

import pytest
import yaml

from decider.exceptions import WiringError
from decider.steps.flowfile import load_flow


def _write(tmp_path, name, doc):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc) if name.endswith((".yaml", ".yml")) else json.dumps(doc))
    return path


def test_unknown_step_name_suggests_a_close_match(tmp_path):
    path = _write(tmp_path, "flow.yaml", {
        "flow": "f", "components": "tests.flowfile.components", "steps": [{"step": "full_limti"}]})
    with pytest.raises(WiringError, match="Did you mean 'full_limit'"):
        load_flow(path)


def test_name_in_both_components_and_documents_is_an_error(tmp_path):
    path = _write(tmp_path, "flow.yaml", {
        "flow": "f", "components": "tests.flowfile.components", "steps": [{"step": "full_limit"}]})
    with pytest.raises(WiringError, match="both components and documents"):
        load_flow(path, documents={"full_limit": {"type": "scorecard", "name": "full_limit", "variables": []}})


def test_root_level_typo_key_is_an_error(tmp_path):
    path = _write(tmp_path, "flow.yaml", {"flow": "f", "stpes": [{"step": "x"}]})
    with pytest.raises(WiringError, match="stpes"):
        load_flow(path)


def test_node_with_no_recognized_kind_key_is_an_error(tmp_path):
    path = _write(tmp_path, "flow.yaml", {"flow": "f", "steps": [{"stpe": "x"}]})
    with pytest.raises(WiringError, match="step/flow/branch/loop"):
        load_flow(path)


def test_missing_modifies_on_a_branch_is_an_error(tmp_path):
    path = _write(tmp_path, "flow.yaml", {"flow": "f", "components": "tests.flowfile.components", "steps": [
        {"branch": "b", "when": "eligible", "then": [{"step": "full_limit"}]}]})
    with pytest.raises(WiringError, match="modifies"):
        load_flow(path)


def test_flow_reference_cycle_is_an_error(tmp_path):
    _write(tmp_path, "flows/a.yaml", {"flow": "a", "steps": [{"flow": "b"}]})
    _write(tmp_path, "flows/b.yaml", {"flow": "b", "steps": [{"flow": "a"}]})
    path = _write(tmp_path, "flow.yaml", {"flow": "root", "steps": [{"flow": "a"}]})
    with pytest.raises(WiringError, match="a -> b -> a"):
        load_flow(path)


def test_referenced_file_flow_name_mismatch_is_an_error(tmp_path):
    _write(tmp_path, "flows/sub.yaml", {"flow": "other", "steps": [{"step": "x"}]})
    path = _write(tmp_path, "flow.yaml", {"flow": "root", "steps": [{"flow": "sub"}]})
    with pytest.raises(WiringError, match="declares flow 'other'"):
        load_flow(path)


def test_inline_steps_and_an_existing_file_is_ambiguous(tmp_path):
    _write(tmp_path, "flows/sub.yaml", {"flow": "sub", "steps": [{"step": "x"}]})
    path = _write(tmp_path, "flow.yaml", {
        "flow": "root", "components": "tests.flowfile.components",
        "steps": [{"flow": "sub", "steps": [{"step": "full_limit"}]}]})
    with pytest.raises(WiringError, match="also exists"):
        load_flow(path)
