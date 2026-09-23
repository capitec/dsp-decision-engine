import pytest

from decider.engine import Engine

MODES = ("interpreted", "stepped", "fused")


@pytest.fixture(params=MODES)
def mode(request):
    return request.param


@pytest.fixture
def bind(mode):
    """`bind(step, **engine_kwargs)`: an Executable in the test's mode."""
    return lambda step, **kw: Engine(**kw).bind(step, mode=mode)
