def test_package_imports():
    import decikit
    import decikit.serving
    import decikit.serving._handler
    import decikit._settings


def test_engine_and_config_value_types_are_top_level():
    import decikit
    from decikit._engine import Engine
    from decikit.steps import ParamRef, Value

    assert (decikit.Engine, decikit.ParamRef, decikit.Value) == (Engine, ParamRef, Value)
    assert {"Engine", "ParamRef", "Value"} <= set(decikit.__all__)


def test_polars_schema_and_version_are_top_level():
    import polars as pl

    import decikit

    assert isinstance(decikit.__version__, str)
    assert decikit.PolarsSchema.from_polars(pl.Schema({"a": pl.Float64})).schema == pl.Schema({"a": pl.Float64})
    assert decikit.PolarsSchema.model_validate({"a": "Float64"}).schema == pl.Schema({"a": pl.Float64})


def test_ir_building_blocks_are_exported_from_engine_ir():
    import decikit._engine.ir as ir

    assert {"CallNode", "Input", "IRContext", "Output", "ParamDecl"} <= set(ir.__all__)


def test_import_decikit_does_not_import_numba():
    import subprocess
    import sys

    code = "import decikit, sys; assert 'numba' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lazy_names_load_on_first_use():
    import decikit
    from decikit.steps.trees import TreeConfig

    assert decikit.TreeConfig is TreeConfig
    assert {"Engine", "Executable", "TreeConfig", "DecisionTableConfig", "ScorecardConfig"} <= set(dir(decikit))


def test_the_offline_libraries_live_under_lab():
    from pathlib import Path

    import decikit
    from decikit.lab import checks, data, describe, diff, experiments, testing

    assert (checks.run, data.read, describe, diff.structure, experiments.run, testing.assert_equivalent)
    # Untracked leftovers (`__pycache__`) may keep an old directory on disk, so assert on source.
    root = Path(decikit.__file__).parent
    for old in ("checks", "diff", "testing", "data", "experiments", "contract", "verify"):
        assert not list((root / old).rglob("*.py")), old


def test_the_registry_and_the_json_forms_are_private_foundations():
    from pathlib import Path

    import decikit
    from decikit._registry import BaseRegistryModule
    from decikit._serde import FrameDoc

    assert (BaseRegistryModule, FrameDoc)
    root = Path(decikit.__file__).parent
    for old in ("registry", "_engine/params/polars_docs", "steps/_polars_docs"):
        assert not list((root / old).rglob("*.py")), old
