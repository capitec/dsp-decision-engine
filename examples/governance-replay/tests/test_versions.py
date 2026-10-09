from governance.flows import FLOWS, FRAUD, LOAN
from governance.versions import version_diff

CARD = "rate_card_flex_loan"


def raise_rates(by, grade=None):
    def edit(config):
        for row in config[CARD]["rows"]:
            if grade in (None, row["grade"]):
                row["rate"] += by
    return edit


def reverse_rows(config):
    config[CARD]["rows"].reverse()


def test_a_rate_card_refresh_is_summarised_not_listed(tmp_path):
    d = version_diff(LOAN, LOAN.replayer(), LOAN.candidate(raise_rates(1.0), tmp_path, "interpreted"))
    assert d.documents == {CARD: {"cells": 63360, "changed": 63360, "mean_move": 1.0, "any_down": False}}
    assert d.params == ()


def test_reordering_the_rows_of_a_rate_card_changes_nothing(tmp_path):
    d = version_diff(LOAN, LOAN.replayer(), LOAN.candidate(reverse_rows, tmp_path, "interpreted"))
    assert d.documents[CARD]["changed"] == 0 and d.documents[CARD]["any_down"] is False


def test_a_changed_rule_is_found_by_its_id_whatever_its_position(tmp_path):
    def edit(config):
        rules = config["rule_set"]["rules"]
        rules.insert(0, rules.pop(rules.index(next(r for r in rules if r["rule_id"] == "MS-0021"))))
        rules[0]["action_code"] = 50

    d = version_diff(FRAUD, FRAUD.replayer(), FRAUD.candidate(edit, tmp_path, "interpreted"))
    assert d.documents == {"rule_set": {"added": [], "removed": [], "changed": ["MS-0021"]}}


def test_a_params_change_lists_the_leaf_that_moved(tmp_path):
    def edit(config):
        config["params"]["fraud_interdiction"]["adjustment_stack_enabled"]["adjustment_stack_enabled"] = False

    d = version_diff(FRAUD, FRAUD.replayer(), FRAUD.candidate(edit, tmp_path, "interpreted"))
    assert [(p["path"], p["old"], p["new"]) for p in d.params] == [
        ("/fraud_interdiction/adjustment_stack_enabled/adjustment_stack_enabled", True, False)]
    assert d.documents == {}


def test_the_same_version_adds_removes_and_changes_nothing():
    d = version_diff(LOAN, LOAN.replayer(), LOAN.replayer())
    s = d.structure
    assert (s.added, s.removed, s.renamed, s.moved, s.edges, d.params, d.documents) == ((), (), (), (), (), (), {})


def test_pipelines_built_from_the_same_documents_are_provably_the_same(flow_code):
    flow = FLOWS[flow_code]
    d = version_diff(flow, flow.replayer(), flow.replayer())
    assert d.proof is True and d.structure.identical
