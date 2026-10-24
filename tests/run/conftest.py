import pytest

from decikit._engine import Engine

MODES = ("interpreted", "stepped", "fused")


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


@pytest.fixture
def bind(mode):
    """`bind(step, **engine_settings)`: an Executable on the test's engine."""
    return lambda step, **kw: Engine.resolve(mode)(**kw).bind(step)
