"""Engines are registry documents: resolved by tag, built from settings, and installable from a package."""
import sys

import pytest

from decikit import Engine, FusedEngine, InterpretedEngine, NumbaEngine, SteppedEngine, flow
from decikit._registry import import_path
from decikit.exceptions import RegistryError
from test_registry import _install_plugin


def ratio(income: float, debt: float) -> float:
    return debt / income


def test_built_in_engines_resolve_by_alias_and_by_import_path():
    for tag, cls in {"interpreted": InterpretedEngine, "stepped": SteppedEngine, "fused": FusedEngine}.items():
        assert Engine.resolve(tag) is cls
        assert Engine.resolve(import_path(cls)) is cls
        assert Engine.registered()[tag] is cls


def test_numba_engine_is_abstract_and_never_registered():
    assert NumbaEngine not in Engine.registered().values()
    with pytest.raises(RegistryError):
        Engine.resolve(import_path(NumbaEngine))
    with pytest.raises(TypeError):
        NumbaEngine()


def test_an_unknown_engine_tag_suggests_the_closest_one():
    with pytest.raises(RegistryError, match="Did you mean 'fused'"):
        Engine.resolve("fsued")


def test_compile_takes_a_tag_or_an_instance_and_the_executable_keeps_it():
    pipeline = flow(ratio, name="p")
    strict = FusedEngine(strict_compile=True)
    assert pipeline.compile("stepped").engine == SteppedEngine()
    assert pipeline.compile(strict).engine is strict
    assert pipeline.compile(strict).run({"income": 4.0, "debt": 1.0})["ratio"] == 0.25


def test_an_engine_is_a_frozen_settings_document():
    engine = FusedEngine.model_validate({"type": "fused", "strict_compile": "true", "params_validation": "lazy"})
    assert engine == FusedEngine(strict_compile=True, params_validation="lazy")
    assert engine.model_dump() == {"type": "fused", "params_validation": "lazy", "strict_compile": True}
    with pytest.raises(ValueError):
        engine.strict_compile = False
    with pytest.raises(ValueError, match="extra"):
        FusedEngine(strict=True)


def test_an_engine_loads_from_an_entry_point_and_compiles_by_its_tag(tmp_path, monkeypatch):
    _install_plugin(tmp_path, monkeypatch, "engines", "ep_engine", "ep_plugin:EpEngine", '''
from typing import Literal
from decikit import Engine, FusedEngine

class EpEngine(Engine):
    type: Literal["ep_engine"] = "ep_engine"
    factor: int = 3

    def bind(self, step_or_ir):
        return FusedEngine().bind(step_or_ir)
''')
    assert "ep_plugin" not in sys.modules
    assert flow(ratio, name="p").compile("ep_engine").run({"income": 4.0, "debt": 1.0})["ratio"] == 0.25
    cls = Engine.resolve("ep_engine")
    assert cls.__name__ == "EpEngine" and cls.model_validate({"type": "ep_engine", "factor": 5}).factor == 5
