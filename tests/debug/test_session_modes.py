"""Debug sessions in stepped and fused modes: same answers as interpreted, fused pauses at kernel boundaries."""
import polars as pl
import pytest

from decider import branch, flow, step
from decider.engine.debug import EVENT_LOG, NodeFinished, Paused, RunFinished

MODES = ("interpreted", "stepped", "fused")


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def headroom(affordability_ratio: float) -> float:
    return affordability_ratio - 1.0


pipeline = flow(disposable_income, affordability_ratio, headroom)
KERNEL = ("disposable_income", "affordability_ratio", "headroom")
FRAME = pl.DataFrame({
    "net_income": [9200.0, 4100.0, 15000.0],
    "expenses":   [3100.0, 1500.0,  6000.0],
    "instalment": [1200.0,  800.0,  2500.0],
})


def term_cap(requested_term: float) -> float:
    return min(requested_term, 60.0)


@step(output="term_cap")
def cap_private(term_cap: float) -> float:
    return min(term_cap, 54.0)


@step(output="term_cap")
def cap_public(term_cap: float) -> float:
    return min(term_cap, 48.0)


def is_private(sector_code: int) -> bool:
    return sector_code == 1


term = flow(term_cap, branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"), name="term")
TERM_FRAME = pl.DataFrame({"requested_term": [72.0, 50.0, 84.0], "sector_code": [1, 2, 1]})


@pytest.mark.parametrize("mode", MODES)
def test_break_set_resume_gives_the_same_output_in_every_mode(mode):
    s = pipeline.session(FRAME, mode=mode)
    s.break_at("affordability_ratio")
    assert s.resume().when == "before"
    s.set("instalment", 100.0)
    assert s.resume() is None
    assert s.output()["headroom"].to_list() == [60.0, 25.0, 89.0]
    assert isinstance(s.events[-1], RunFinished)


@pytest.mark.parametrize("mode", MODES)
def test_an_override_reaches_the_kernels_after_it(mode):
    s = term.session(TERM_FRAME, mode=mode)
    s.break_at("term/by_sector")
    assert s.resume().origin.path == "term/by_sector"
    s.set("term_cap", 10.0)
    s.resume()
    assert s.output()["term_cap"].to_list() == [10.0, 10.0, 10.0]


@pytest.mark.parametrize("mode", MODES)
def test_an_untouched_session_outputs_what_run_does(mode):
    s = term.session(TERM_FRAME, mode=mode)
    s.resume()
    assert s.output().equals(term.run(TERM_FRAME))


@pytest.mark.parametrize("mode", MODES)
def test_pause_and_rewind_work_in_every_mode(mode):
    s = term.session(TERM_FRAME, mode=mode)
    s.pause()
    assert (s.resume().origin.path, s.current.when) == ("term", "before")
    s.resume()
    s.set("requested_term", 30.0)
    at = s.rewind("term/term_cap")
    assert (at.origin.path, at.when) == ("term/term_cap", "before")
    s.resume()
    assert s.output()["term_cap"].to_list() == [30.0, 30.0, 30.0]


@pytest.mark.parametrize("mode", ("interpreted", "stepped"))
def test_stepped_mode_pauses_at_every_node_like_interpreted(mode):
    s = term.session(TERM_FRAME, mode=mode)
    s.break_at("term/by_sector")
    s.resume()
    seen = []
    while (at := s.step_into()).origin.path != "term/by_sector":
        seen.append(at.origin.path)
    assert seen == ["term/by_sector/is_private"] * 2 + ["term/by_sector/cap_private"] * 2 + \
        ["term/by_sector/cap_public"] * 2


def test_a_breakpoint_inside_a_fused_kernel_pauses_before_the_kernel_and_says_so():
    s = pipeline.session(FRAME, mode="fused")
    s.break_at("affordability_ratio")
    at = s.resume()
    assert (at.origin.path, at.when) == ("disposable_income", "before")
    (paused,) = [e for e in s.events if isinstance(e, Paused)]
    assert (paused.reason, paused.kernel) == ("breakpoint", KERNEL)
    assert EVENT_LOG.validate_json(EVENT_LOG.dump_json(s.events)) == s.events


def test_step_runs_a_whole_fused_kernel_and_reports_what_it_stores():
    s = pipeline.session(FRAME, mode="fused")
    s.step_into()                                     # the flow
    assert s.step_into().origin.path == "disposable_income"
    at = s.step()
    assert (at.origin.path, at.when) == ("disposable_income", "after")
    (finished,) = [e for e in s.events if isinstance(e, NodeFinished) and e.origin.path == "disposable_income"]
    assert set(finished.outputs) == {"headroom"}


def test_rewind_to_a_step_inside_a_fused_kernel_rewinds_to_the_kernel():
    s = pipeline.session(FRAME, mode="fused")
    s.resume()
    s.set("instalment", 100.0)
    at = s.rewind("headroom")
    assert at.origin.path == "disposable_income"
    s.resume()
    assert s.output()["headroom"].to_list() == [60.0, 25.0, 89.0]


def test_a_value_inside_a_fused_kernel_is_an_error_suggesting_stepped_mode():
    s = pipeline.session(FRAME, mode="fused")
    s.resume()
    assert s.value("headroom").to_list() == pipeline.run(FRAME)["headroom"].to_list()
    for spec in ("affordability_ratio", "affordability_ratio@affordability_ratio"):
        with pytest.raises(KeyError, match="fused kernel.*mode='stepped'"):
            s.value(spec)
    with pytest.raises(KeyError, match="mode='stepped'"):
        s.set("disposable_income", 1.0)
