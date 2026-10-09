"""The pipeline reads no hidden objects: a release check can compare it, and an edit to a rule or a document shows."""
import copy
import json
import math
import warnings

from decider import ConfigurableStep, Engine, flow, step
from decider.exceptions import HiddenStateWarning
from decider.lab import diff

from retail_credit import caps, disclosure, eligibility, routing, validation
from retail_credit.documents import SHIPPED, shipped
from retail_credit.pipeline import build


def test_building_the_pipeline_emits_no_hidden_state_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Engine().bind(build(), mode="interpreted")
    assert [str(w.message) for w in caught if issubclass(w.category, HiddenStateWarning)] == []


def test_two_builds_of_the_pipeline_diff_identical():
    assert diff.structure(build(), build()).identical


def amount_rules(rules=caps.RULES):
    return flow(step(caps.seed, outputs=caps.AMOUNT), shipped("appetite_by_grade"),
                *(step(rule, outputs=caps.AMOUNT) for rule in rules), name="amount_rules").emit(*caps.AMOUNT)


def test_a_changed_threshold_in_one_rule_shows_in_the_diff_as_that_rule():
    @caps.allow_fallback
    def cap_0131(amount_value: float, amount_chain: list[dict], amount_declined: int,
                 months_employed: float = caps.missing_as(0.0)) -> tuple[float, list[dict], int]:
        return caps.narrow(amount_value, amount_chain, amount_declined, "CAP-0131", True,
                           30_000.0 if months_employed < 9 else 90_000.0 if months_employed < 24 else math.inf)

    edited = amount_rules([cap_0131 if rule.__name__ == "cap_0131" else rule for rule in caps.RULES])
    changed = diff.structure(amount_rules(), edited).changed_steps
    assert [c.ref.path for c in changed] == ["amount_rules/cap_0131"]


def test_the_order_of_two_rules_is_part_of_the_diff():
    swapped = list(caps.RULES)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    assert not diff.structure(amount_rules(), amount_rules(swapped)).identical


def test_a_changed_appetite_cap_shows_in_the_diff():
    document = json.loads((SHIPPED / "appetite_by_grade.json").read_text())
    document["rows"][6]["appetite_amount_cap"] = 170_000.0
    edited = flow(step(caps.seed, outputs=caps.AMOUNT), ConfigurableStep.load(document),
                  *(step(rule, outputs=caps.AMOUNT) for rule in caps.RULES), name="amount_rules").emit(*caps.AMOUNT)
    assert not diff.structure(amount_rules(), edited).identical


def test_a_changed_overlay_effect_shows_in_the_diff_and_a_changed_rationale_does_not():
    document = json.loads((SHIPPED / "amount_cap_overlays.json").read_text())
    tighter = copy.deepcopy(document)
    tighter["adjustments"][0]["effect"]["value"] = 0.80
    reworded = copy.deepcopy(document)
    reworded["adjustments"][0]["rationale"] = "Reworded"
    base = build()
    assert not diff.structure(base, build(amount_cap_overlays=ConfigurableStep.load(tighter))).identical
    assert diff.structure(base, build(amount_cap_overlays=ConfigurableStep.load(reworded))).identical


def test_every_reason_the_pipeline_can_raise_is_in_the_reason_codes_document():
    raised = {value for module in (disclosure, eligibility, routing, validation, caps) for name, value in vars(module).items()
              if name.startswith(("R_", "A_")) and isinstance(value, int)}
    documented = {code.code for code in shipped("reason_codes").codes}
    assert raised <= documented


def test_the_cap_fold_columns_are_not_in_the_response(decide):
    assert not set(caps.AMOUNT) & set(decide())
