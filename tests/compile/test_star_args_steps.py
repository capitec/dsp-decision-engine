import warnings
from contextlib import nullcontext
from typing import Any, Literal

import polars as pl
import pytest

from decikit import ConfigurableStep, FallbackWarning, allow_fallback, flow
from decikit._engine.ir.decls import Input, Output
from decikit._engine.ir.nodes import CallNode


def total(*, seed: float, **columns: Any) -> float:
    return seed + sum(columns.values())


class Total(ConfigurableStep):
    type: Literal["total"] = "total"
    declared: bool = False

    def to_ir(self, ctx):
        inputs = (Input("a", float, arg="seed"), Input("b", float))
        fn = allow_fallback(lambda *, seed, **columns: total(seed=seed, **columns)) if self.declared else total
        return CallNode(ctx.origin(self), "scalar", fn, inputs, (Output(self.name, float),), ())


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_a_step_declaring_star_kwargs_falls_back_to_python_with_a_warning_in_fused(mode):
    exe = flow(Total(name="s"), name="p").compile(mode)
    with pytest.warns(FallbackWarning, match="p/s runs in Python") if mode == "fused" else nullcontext():
        out = exe.run(pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]}))
    assert out["s"].to_list() == [3.0, 5.0]
    assert "'**columns'" in exe.fallbacks()["p/s"]


def test_allow_fallback_silences_the_star_kwargs_warning():
    exe = flow(Total(name="s", declared=True), name="p").compile("fused")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert exe.run({"a": 1.0, "b": 2.0})["s"] == 3.0
