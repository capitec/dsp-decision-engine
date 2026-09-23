def test_package_imports():
    import decider
    import decider.serving
    import decider.serving.handler
    import decider.settings


def test_engine_and_config_value_types_are_top_level():
    import decider
    from decider.engine import Engine
    from decider.steps import ParamRef, Value

    assert (decider.Engine, decider.ParamRef, decider.Value) == (Engine, ParamRef, Value)
    assert {"Engine", "ParamRef", "Value"} <= set(decider.__all__)


def test_ir_building_blocks_are_exported_from_engine_ir():
    import decider.engine.ir as ir

    assert {"CallNode", "Input", "IRContext", "Output", "ParamDecl"} <= set(ir.__all__)
