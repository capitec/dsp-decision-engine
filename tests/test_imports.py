def test_package_imports():
    import decider
    import decider.serving
    import decider.serving._handler
    import decider._settings


def test_engine_and_config_value_types_are_top_level():
    import decider
    from decider._engine import Engine
    from decider.steps import ParamRef, Value

    assert (decider.Engine, decider.ParamRef, decider.Value) == (Engine, ParamRef, Value)
    assert {"Engine", "ParamRef", "Value"} <= set(decider.__all__)


def test_ir_building_blocks_are_exported_from_engine_ir():
    import decider._engine.ir as ir

    assert {"CallNode", "Input", "IRContext", "Output", "ParamDecl"} <= set(ir.__all__)


def test_import_decider_does_not_import_numba():
    import subprocess
    import sys

    code = "import decider, sys; assert 'numba' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lazy_names_load_on_first_use():
    import decider
    from decider.steps.trees import TreeConfig

    assert decider.TreeConfig is TreeConfig
    assert {"Engine", "Executable", "TreeConfig", "DecisionTableConfig", "ScorecardConfig"} <= set(dir(decider))


def test_the_offline_libraries_live_under_lab():
    from pathlib import Path

    import decider
    from decider.lab import checks, data, describe, diff, experiments, testing

    assert (checks.run, data.read, describe, diff.structure, experiments.run, testing.assert_equivalent)
    # Untracked leftovers (`__pycache__`) may keep an old directory on disk, so assert on source.
    root = Path(decider.__file__).parent
    for old in ("checks", "diff", "testing", "data", "experiments", "contract", "verify"):
        assert not list((root / old).rglob("*.py")), old
