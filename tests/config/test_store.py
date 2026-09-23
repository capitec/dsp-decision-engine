from __future__ import annotations

import json

import pytest
from pydantic import TypeAdapter

from decider.config import JsonFileStore, Version, VersionPart
from decider.engine.ir.decls import Input, Output
from decider.engine.ir.nodes import CallNode
from decider.settings import DeciderConfigSettings
from decider.steps import ConfigurableStep, ParamRef, StepRef, Value


class Cutoff(ConfigurableStep):
    column: str
    cut: Value[float] = 0.5

    def to_ir(self, ctx):
        return CallNode(ctx.origin(self), "row", _above, (Input(self.column, float),), (Output(self.name, bool),), ())


def _above(row, params, consts):
    return (row[0] > 0.5,)


def test_create_version_writes_one_json_file_per_key(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    versioned = store.create_version({"params": {"cut": 0.7}})
    assert versioned.version == Version(0, 0, 0)
    assert json.loads((tmp_path / "0.0.0" / "params.json").read_text()) == {"cut": 0.7}


def test_fresh_store_loads_what_another_wrote(tmp_path):
    JsonFileStore(basepath=str(tmp_path)).create_version({"params": {"cut": 7.0}})
    fresh = JsonFileStore(basepath=str(tmp_path))
    assert fresh.read(fresh.latest_version()).config == {"params": {"cut": 7.0}}


def test_versions_increment_and_latest_wins(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version({"params": {"cut": 1}})
    store.create_version({"params": {"cut": 2}})
    store.create_version({"params": {"cut": 3}}, bump=VersionPart.MAJOR)
    fresh = JsonFileStore(basepath=str(tmp_path))
    assert fresh.versions() == [Version(0, 0, 0), Version(0, 1, 0), Version(1, 0, 0)]
    assert fresh.read(fresh.latest_version()).config["params"] == {"cut": 3}
    assert fresh.read("0.1.0").config["params"] == {"cut": 2}


def test_versions_sort_numerically_and_ignore_other_entries(tmp_path):
    for name in ("0.10.0", "0.9.0", "notes", ".staging"):
        (tmp_path / name).mkdir()
    (tmp_path / "1.0.0").write_text("a file, not a version")
    assert JsonFileStore(basepath=str(tmp_path)).versions() == [Version(0, 9, 0), Version(0, 10, 0)]


def test_empty_store_has_no_latest(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path / "missing"))
    assert store.versions() == [] and store.latest_version() is None


def test_dotted_keys_are_subdirectories(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version({"credit.bureau": {"a": 1}, "main": {"b": 2}})
    assert (tmp_path / "0.0.0" / "credit" / "bureau.json").exists()
    assert store.read("0.0.0").config == {"credit.bureau": {"a": 1}, "main": {"b": 2}}


def test_stored_version_is_unaffected_by_later_edits_to_the_input(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    doc = {"params": {"cut": 1}}
    store.create_version(doc)
    doc["params"]["cut"] = 99
    assert store.read("0.0.0").config["params"] == {"cut": 1}


def test_failed_write_leaves_no_version(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    with pytest.raises(TypeError):
        store.create_version({"ok": {"a": 1}, "bad": {"a": object()}})
    assert store.versions() == []


def test_read_errors_propagate(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version({"params": {}})
    (tmp_path / "0.0.0" / "params.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        JsonFileStore(basepath=str(tmp_path)).read("0.0.0")
    with pytest.raises(FileNotFoundError):
        store.read("9.9.9")
    assert not hasattr(store, "subscribe_version_updates")


def test_configurable_step_and_params_round_trip(tmp_path):
    cfg = Cutoff(name="hi", column="ratio", cut=ParamRef(param="hi_cut", default=0.7))
    params = {"shared": {"rate": 0.1}, "scoring": {"hi_cut": 0.8}}
    JsonFileStore(basepath=str(tmp_path)).create_version({"cutoff": cfg.model_dump(mode="json"), "params": params})

    loaded = JsonFileStore(basepath=str(tmp_path)).read("0.0.0").config
    step = TypeAdapter(StepRef).validate_python(loaded["cutoff"])
    assert type(step) is Cutoff and step == cfg
    assert ConfigurableStep.resolve(loaded["cutoff"]["type"]) is Cutoff
    assert loaded["params"] == params


def test_settings_build_the_store(tmp_path):
    store = DeciderConfigSettings(basepath=str(tmp_path)).get()
    assert isinstance(store, JsonFileStore) and store.basepath == str(tmp_path)
    with pytest.raises(LookupError, match="file:yaml"):
        DeciderConfigSettings(type="file:yaml").get()
