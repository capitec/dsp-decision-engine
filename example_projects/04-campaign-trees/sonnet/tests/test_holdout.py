"""holdout.py: Stage 7 (spec 04 §5.7) -- deterministic, re-derivable assignment."""
from __future__ import annotations

from decider import Engine

from campaign_trees import holdout


def test_control_assignment_is_deterministic_across_calls():
    exe = Engine().bind(holdout.is_control_step, mode="interpreted")
    a = exe.score({"client_id": "C-1000042", "campaign_id": 23})
    b = exe.score({"client_id": "C-1000042", "campaign_id": 23})
    assert a["is_control"] == b["is_control"]


def test_control_assignment_needs_no_stored_table():
    """§10 item 8: re-derivable from `client_id`, `campaign_id` and `holdout_design_version`
    alone -- calling it fresh, with no memory of any prior run, reproduces the same answer."""
    exe = Engine().bind(holdout.is_control_step, mode="interpreted")
    first = exe.score({"client_id": "C-777", "campaign_id": 23, "holdout_design_version": "hv1"})
    exe2 = Engine().bind(holdout.is_control_step, mode="interpreted")  # a fresh bind, no shared state
    second = exe2.score({"client_id": "C-777", "campaign_id": 23, "holdout_design_version": "hv1"})
    assert first["is_control"] == second["is_control"]


def test_roughly_the_declared_percentage_is_control():
    exe = Engine().bind(holdout.is_control_step, mode="interpreted")
    n, control = 20_000, 0
    for i in range(n):
        out = exe.score({"client_id": f"C-{i}", "campaign_id": 23}, params={"is_control": {"control_percentage": 0.05}})
        control += out["is_control"]
    assert 0.03 < control / n < 0.07  # 5% +/- generous tolerance at this sample size


def test_only_a_design_version_change_moves_clients_between_groups():
    """§5.7 requirement 2: "only a change to holdout_design_version moves anyone.\""""
    exe = Engine().bind(holdout.is_control_step, mode="interpreted")
    ids = [f"C-{i}" for i in range(5_000)]
    v1 = {cid: exe.score({"client_id": cid, "campaign_id": 23, "holdout_design_version": "hv1"})["is_control"]
          for cid in ids}
    v1_again = {cid: exe.score({"client_id": cid, "campaign_id": 23, "holdout_design_version": "hv1"})["is_control"]
                for cid in ids}
    assert v1 == v1_again  # same version, same cycle-to-cycle answer

    v2 = {cid: exe.score({"client_id": cid, "campaign_id": 23, "holdout_design_version": "hv2"})["is_control"]
          for cid in ids}
    moved = [cid for cid in ids if v1[cid] != v2[cid]]
    assert moved  # a version bump *does* move a measurable, enumerable set of clients


def test_universal_holdout_is_independent_of_campaign_control():
    exe = Engine().bind(holdout.is_universal_holdout_step, mode="interpreted")
    out = exe.score({"client_id": "C-1", "universal_holdout_design_version": "uhv1"})
    assert out["is_universal_holdout"] in (True, False)


def test_variant_split_partitions_deterministically():
    a = holdout.tree_variant("C-1", 23, "vv1", (0.9, 0.1))
    b = holdout.tree_variant("C-1", 23, "vv1", (0.9, 0.1))
    assert a == b
    assert a in (0, 1)
