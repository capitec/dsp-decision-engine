"""Params reach nodes by path, the same way in run and score; unknown entries are hard errors."""
from __future__ import annotations

import polars as pl
import pytest

from decider import flow, param
from decider.engine.params import ParamsError

FRAME = pl.DataFrame({"term_cap": [60.0, 60.0]})


def cap(term_cap: float, cap: float = param(48.0, ge=6, le=60)) -> float:
    return min(term_cap, cap)


def net_of_floor(term_cap: float, floor: float = param(10.0)) -> float:
    return term_cap - floor


def plus_headroom(net_of_floor: float, headroom: float = param(3.0), base: float = param(0.0, shared_key="base")) -> float:
    return net_of_floor + headroom + base


band = flow(net_of_floor, plus_headroom, name="band")


def test_a_named_flow_is_tuned_through_its_path(bind):
    exe = bind(band)
    assert exe.run(FRAME)["plus_headroom"].to_list() == [53.0, 53.0]
    tuned = exe.run(FRAME, params={"band": {"net_of_floor": {"floor": 20.0}, "plus_headroom": {"headroom": 5.0}}})
    assert tuned["plus_headroom"].to_list() == [45.0, 45.0]


def test_score_is_namespaced_the_same_way_as_run(bind):
    exe = bind(band)
    assert exe.score({"term_cap": 60.0}, params={"band": {"net_of_floor": {"floor": 20.0}}})["plus_headroom"] == 43.0


def test_a_shared_param_is_read_from_the_shared_entry(bind):
    assert bind(band).run(FRAME, params={"shared": {"base": 1.0}})["plus_headroom"].to_list() == [54.0, 54.0]


def test_an_unknown_params_namespace_is_a_hard_error_with_a_suggestion(bind):
    with pytest.raises(ParamsError, match="no step with params at 'kap'. Did you mean 'cap'?"):
        bind(flow(cap)).run(FRAME, params={"kap": {"cap": 12.0}})


def test_an_unknown_nested_namespace_is_a_hard_error_with_a_suggestion(bind):
    with pytest.raises(ParamsError, match="'band/net_of_flor'. Did you mean 'band/net_of_floor'?"):
        bind(band).run(FRAME, params={"band": {"net_of_flor": {"floor": 1.0}}})


def test_an_unknown_shared_key_is_a_hard_error_with_a_suggestion(bind):
    with pytest.raises(ParamsError, match="shared param 'bse'. Did you mean 'base'?"):
        bind(band).run(FRAME, params={"shared": {"bse": 1.0}})


def test_the_namespace_is_checked_in_lazy_mode_too(bind):
    with pytest.raises(ParamsError, match="'kap'"):
        bind(flow(cap), params_validation="lazy").run(FRAME, params={"kap": {"cap": 12.0}})
