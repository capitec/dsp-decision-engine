"""Sessions stepping into branch arms and loop iterations, and over them, in every mode."""
import polars as pl
import pytest

from decider import branch, flow, loop, step
from decider.engine.debug import EVENT_LOG, NodeStarted, Paused
from decider.testing import assert_equivalent


def is_private(sector_code: int) -> bool:
    return sector_code == 1


@step(output="term_cap")
def cap_private(term_cap: float) -> float:
    return min(term_cap, 54.0)


@step(output="term_cap")
def cap_public(term_cap: float) -> float:
    return min(term_cap, 60.0)


@step(output="term_cap")
def cap_band(term_cap: float) -> float:
    return term_cap - 1.0


by_sector = branch(is_private, flow(cap_private, cap_band, name="private"), cap_public,
                   modifies=["term_cap"], name="by")
TERM = pl.DataFrame({"term_cap": [72.0, 72.0, 50.0], "sector_code": [1, 2, 1]})


def keep_going(best: float) -> bool:
    return best < 10


@step(output="best")
def improve(best: float, step_size: float) -> float:
    return best + step_size


search = loop(keep_going, improve, carries=["best"], max_iterations=10, name="search")
BEST = pl.DataFrame({"best": [0.0, 9.0, 20.0], "step_size": [3.0, 3.0, 1.0]})


def walk(session, until):
    """Every checkpoint `step_into` reaches until the one `until` accepts, as (when, path, arm, iteration)."""
    seen = []
    while not until(at := session.step_into()):
        seen.append((at.when, at.origin.path, at.arm, at.iteration))
    return seen


@pytest.mark.parametrize("mode", ["interpreted", "stepped"])
def test_step_into_a_branch_visits_each_taken_arm_in_order_and_says_which(mode):
    s = by_sector.session(TERM, mode=mode)
    s.step_into()                                                    # before the branch
    seen = walk(s, lambda cp: cp.origin.path == "by")
    assert seen == [
        ("before", "by/is_private", None, None), ("after", "by/is_private", None, None),
        ("before", "by/private", 0, None),
        ("before", "by/private/cap_private", 0, None), ("after", "by/private/cap_private", 0, None),
        ("before", "by/private/cap_band", 0, None), ("after", "by/private/cap_band", 0, None),
        ("after", "by/private", 0, None),
        ("before", "by/cap_public", 1, None), ("after", "by/cap_public", 1, None),
    ]
    started = [e for e in s.events if isinstance(e, NodeStarted) and e.origin.path == "by/cap_public"]
    assert [e.arm for e in started] == [1]


@pytest.mark.parametrize("mode", ["interpreted", "stepped"])
def test_an_arm_no_row_takes_is_never_entered(mode):
    s = by_sector.session(TERM.with_columns(sector_code=pl.lit(2)), mode=mode)
    s.step_into()
    paths = [p for _, p, _, _ in walk(s, lambda cp: cp.origin.path == "by")]
    assert "by/private" not in paths and "by/cap_public" in paths


@pytest.mark.parametrize("mode", ["interpreted", "stepped"])
def test_step_into_a_loop_visits_each_iteration_and_numbers_it(mode):
    s = search.session(BEST, mode=mode)
    s.step_into()
    seen = walk(s, lambda cp: cp.origin.path == "search")
    # Row 0 needs 4 iterations, row 1 one; the fifth check finds no row going on.
    assert [(w, p, i) for w, p, _, i in seen if w == "before"] == [
        ("before", "search/keep_going", 1), ("before", "search/improve", 1),
        ("before", "search/keep_going", 2), ("before", "search/improve", 2),
        ("before", "search/keep_going", 3), ("before", "search/improve", 3),
        ("before", "search/keep_going", 4), ("before", "search/improve", 4),
        ("before", "search/keep_going", 5),
    ]


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_step_runs_a_whole_branch_or_loop(mode):
    for pipeline, frame, path, name, values in [(by_sector, TERM, "by", "term_cap", [53.0, 60.0, 49.0]),
                                                (search, BEST, "search", "best", [12.0, 12.0, 20.0])]:
        s = pipeline.session(frame, mode=mode)
        assert s.step_into().origin.path == path
        at = s.step()
        assert (at.when, at.origin.path) == ("after", path)
        assert s.value(name).to_list() == values


def test_a_paused_event_says_which_iteration_and_round_trips_as_json():
    s = search.session(BEST)
    s.break_at("search/improve")
    s.resume()
    s.resume()
    paused = [e for e in s.events if isinstance(e, Paused)]
    assert [(p.origin.path, p.iteration) for p in paused] == [("search/improve", 1), ("search/improve", 2)]
    assert EVENT_LOG.validate_json(EVENT_LOG.dump_json(s.events)) == s.events


def test_a_packed_branch_is_one_fused_checkpoint_covering_every_step_inside():
    s = by_sector.session(TERM, mode="fused")
    s.break_at("by/private/cap_band")
    at = s.resume()
    assert (at.when, at.origin.path) == ("before", "by")
    (paused,) = [e for e in s.events if isinstance(e, Paused)]
    assert paused.kernel == ("by", "by/is_private", "by/private/cap_private", "by/private/cap_band", "by/cap_public")
    assert s.step_into().origin.path == "by"
    assert s.current.when == "after"
    with pytest.raises(KeyError, match="mode='stepped'"):
        s.value("term_cap@by/private/cap_private")
    s.resume()
    assert s.output()["term_cap"].to_list() == [53.0, 60.0, 49.0]


def test_a_fused_branch_that_cannot_pack_is_stepped_into_like_stepped_mode():
    unpackable = flow(by_sector).emit("is_private@by/is_private")
    s = unpackable.session(TERM, mode="fused")
    s.break_at("by/cap_public")
    at = s.resume()
    assert (at.origin.path, at.arm) == ("by/cap_public", 1)


def test_a_breakpoint_and_override_inside_a_loop_agree_across_modes():
    def override_step_size(s):
        s.break_at("search")
        s.resume()
        s.set("step_size", 5.0)

    out = assert_equivalent(flow(search), BEST, script=override_step_size)
    assert out["best"].to_list() == [12.0, 12.0, 20.0]
