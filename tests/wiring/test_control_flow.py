"""Branches, loops and unknown-lineage frame steps: nested structure kept, versions merged."""
from __future__ import annotations

import pytest

from decider import branch, flow, frame_step, loop, step
from decider.engine.wiring import Branch, Call, Loop, Sequence, interface, resolve
from decider.engine import to_ir


def is_private(sector_code: int) -> bool:
    return sector_code == 1


@step(output="term_cap")
def cap_private(term_cap: float) -> float:
    return min(term_cap, 54.0)


@step(output="term_cap")
def cap_public(term_cap: float) -> float:
    return min(term_cap, 60.0)


def note(term_cap: float) -> float:
    return term_cap


def seed(requested_term: float) -> float:
    return requested_term


term_cap = step(seed, output="term_cap")


def test_a_branch_merges_each_arms_version_of_what_it_modifies():
    plan = resolve(flow(term_cap, branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by")))
    seeded, fork = plan.root.children
    assert isinstance(fork, Branch) and fork.condition.node.origin.path == "by/is_private"
    (merge,) = fork.merges
    assert merge.version.producer == "by"
    assert [v.producer for v in merge.arms] == ["by/cap_private", "by/cap_public"]
    assert merge.prior is None
    assert fork.arms[0].reads[0] is seeded.writes[0]
    assert plan.outputs["term_cap"] is merge.version


def test_an_arm_that_leaves_a_modified_name_alone_keeps_the_prior_version():
    def keep(term_cap: float) -> float:
        return term_cap

    plan = resolve(flow(term_cap, branch(is_private, cap_private, keep, modifies=["term_cap"], name="by")))
    seeded, fork = plan.root.children
    (merge,) = fork.merges
    assert merge.arms[1] is None
    assert merge.prior is seeded.writes[0]


def test_a_modified_name_nothing_produced_before_is_an_input_column_when_an_arm_leaves_it():
    plan = resolve(branch(is_private, cap_private, note, modifies=["term_cap"], name="by"))
    assert plan.root.merges[0].prior.producer is None
    assert {i.name for i in plan.inputs} == {"sector_code", "term_cap"}


def test_every_version_includes_each_arm_and_the_merge():
    plan = resolve(flow(term_cap, branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by"))
                   .emit("term_cap@*"))
    assert [k for k in plan.outputs if "@" in k] == [
        "term_cap@seed", "term_cap@by/cap_private", "term_cap@by/cap_public", "term_cap@by",
    ]


def test_arm_writes_outside_modifies_and_the_condition_stay_inside_the_branch():
    def extra(term_cap: float) -> float:
        return term_cap

    arm = flow(cap_private, extra, name="private")
    plan = resolve(flow(term_cap, branch(is_private, arm, cap_public, modifies=["term_cap"], name="by")))
    assert set(plan.outputs) == {"requested_term", "sector_code", "term_cap"}
    assert isinstance(plan.root.children[1].arms[0], Sequence)


def test_a_modified_name_no_arm_writes_is_an_error():
    with pytest.raises(ValueError, match="branch by: modifies 'band', but no arm writes it"):
        resolve(flow(term_cap, branch(is_private, cap_private, cap_public, modifies=["band"], name="by")))


def keep_going(best: float) -> bool:
    return best < 10


@step(output="best")
def improve(best: float, step_size: float) -> float:
    return best + step_size


def test_a_loop_carries_one_version_through_its_iterations():
    plan = resolve(loop(keep_going, improve, carries=["best"], max_iterations=5, name="search"))
    lp = plan.root
    assert isinstance(lp, Loop)
    (carry,) = lp.carries
    assert carry.version.producer == "search"
    assert carry.initial.producer is None
    assert carry.last.producer == "search/improve"
    assert lp.condition.reads[0] is carry.version
    assert lp.body.reads[0] is carry.version
    assert plan.outputs["best"] is carry.version
    assert [i.name for i in plan.inputs] == ["best", "step_size"]


def test_a_carry_the_body_never_writes_is_an_error():
    def other(best: float) -> float:
        return best

    with pytest.raises(ValueError, match="loop search: carries 'best', but the body never writes it"):
        resolve(loop(keep_going, other, carries=["best"], max_iterations=5, name="search"))


def test_a_loop_condition_reading_what_only_the_body_produces_is_an_error():
    def done(score: float) -> bool:
        return score > 1

    @step(outputs=("best", "score"))
    def body(best: float) -> tuple[float, float]:
        return best, best

    with pytest.raises(ValueError, match="search/done reads 'score' as an input column"):
        resolve(loop(done, body, carries=["best"], max_iterations=5, name="search"))


def enrich(df):
    return df


def test_names_read_after_an_unknown_lineage_frame_come_from_it():
    barrier = frame_step(enrich)

    def ratio(disposable_income: float, instalment: float) -> float:
        return disposable_income / instalment

    plan = resolve(flow(barrier, ratio))
    frame, call = plan.calls
    assert frame.reads is None
    assert [v.name for v in frame.writes] == ["disposable_income", "instalment"]
    assert all(v.producer == "enrich" for v in call.reads)
    assert plan.inputs == ()


def test_a_name_produced_after_the_frame_is_read_from_its_producer():
    def a(x: float) -> float:
        return x

    def b(a: float) -> float:
        return a

    plan = resolve(flow(frame_step(enrich), a, b))
    _, first, second = plan.calls
    assert second.reads[0] is first.writes[0]


def test_a_frame_with_known_columns_is_no_barrier():
    known = frame_step(enrich, name="join", reads=["client_id"], writes=["bureau_score"])

    def risky(bureau_score: float) -> bool:
        return bureau_score < 600

    plan = resolve(flow(known, risky))
    frame, call = plan.calls
    assert [v.producer for v in frame.reads] == [None]
    assert call.reads[0] is frame.writes[0]


def test_a_barrier_inside_an_arm_may_write_what_the_branch_modifies():
    arm = flow(frame_step(enrich), note, name="framed")
    plan = resolve(flow(term_cap, branch(is_private, arm, cap_public, modifies=["term_cap"], name="by")))
    (merge,) = plan.root.children[1].merges
    assert merge.arms[0].producer == "by/framed/enrich"


def test_interface_of_a_branch_is_what_it_reads_and_modifies():
    reads, writes = interface(to_ir(branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by")))
    assert reads == {"sector_code", "term_cap"}
    assert writes == {"term_cap"}


def test_interface_of_a_loop_is_what_it_reads_and_carries():
    reads, writes = interface(to_ir(loop(keep_going, improve, carries=["best"], max_iterations=5, name="s")))
    assert reads == {"best", "step_size"}
    assert writes == {"best"}


def test_interface_of_a_flow_reads_only_from_outside():
    def a(x: float) -> float:
        return x

    def b(a: float, y: float) -> float:
        return a + y

    assert interface(to_ir(flow(a, b))) == ({"x", "y"}, {"a", "b"})


def test_every_call_in_the_tree_is_in_the_flat_list_once():
    plan = resolve(flow(term_cap, branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by")))
    seen = []

    def calls(r):
        if isinstance(r, Call):
            seen.append(r)
        elif isinstance(r, Sequence):
            for c in r.children:
                calls(c)
        elif isinstance(r, Branch):
            calls(r.condition)
            for arm in r.arms:
                calls(arm)

    calls(plan.root)
    assert seen == list(plan.calls)
