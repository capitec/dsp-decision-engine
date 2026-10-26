"""The rule set as a config document: each rule's governance record and its condition, side by side."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from decikit import ConfigurableStep, TreeConfig, flow
from pydantic import BaseModel, ConfigDict, model_validator

from fraud_interdiction import vocab
from fraud_interdiction.firing import firing_step


class Rule(BaseModel):
    """One rule. `condition` is a decikit tree-rule condition; `base_condition` is the same rule at its
    authored thresholds, set only on rules whose condition reads an overlay-adjusted feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    family: Literal["CF", "AT", "MS", "FP", "AA"]
    status: Literal["live", "shadow"]
    severity: int
    action_code: int
    priority: int
    reason_code: int
    critical: bool = False
    event_types: tuple[int, ...] = (vocab.EVENT_TYPE_INSTANT_PAYMENT,)
    segments: tuple[str, ...] | None = None  # None: every segment
    effective_from: date
    effective_to: date | None = None
    # What to do when the velocity aggregate the rule reads is stale or absent.
    velocity_feature: Literal["velocity_count_1min", "velocity_count_10min", "velocity_sum_amount_1h"] | None = None
    stale_absent: Literal["evaluate_false", "last_known_good", "suppress"] = "last_known_good"
    overlay_exempt: bool = False
    condition: dict[str, Any]
    base_condition: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _exempt_rules_are_not_adjusted(self) -> Rule:
        if self.overlay_exempt and self.base_condition is not None:
            raise ValueError(f"{self.rule_id} is overlay-exempt but reads an overlay-adjusted threshold")
        return self


def _tree(name: str, rules: list[Rule], column: str, *, base: bool = False) -> TreeConfig:
    leaf = {"type": "leaf", "result_idx": 0}

    def root(r: Rule) -> dict:
        c = r.base_condition if base else r.condition
        node = {**c, "then": leaf} if c.get("type") == "composite" else {"type": "unary", "condition": c, "then": leaf}
        return {"meta": {"name": r.rule_id}, "rule": node}

    return TreeConfig.load({"type": "tree", "name": name, "tree": {
        "type": "prioritized_flat_rule", "mode": "all", "rules": [root(r) for r in rules],
        "output": {"data": [{column: True}], "default": {column: False}, "dtypes": [[column, "Boolean"]]}}})


class RuleSet(ConfigurableStep):
    """Every live and shadow rule. Writes `fired_rule_ids`, `unevaluable_rule_ids`, `fired_on_overlay_ids`,
    `counterfactual_fired_rule_ids` and `shadow_fired_rule_ids`.

    Example::

        rule_set = RuleSet.load("configs/0.1.0/rule_set.json")
        flow(rule_set, name="p").run(df)
    """

    type: Literal["rule_set"] = "rule_set"
    rules: tuple[Rule, ...]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> RuleSet:
        ids = [r.rule_id for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate rule ids: {sorted({i for i in ids if ids.count(i) > 1})}")
        return self

    def to_ir(self, ctx: Any) -> Any:
        live = [r for r in self.rules if r.status == "live"]
        shadow = [r for r in self.rules if r.status == "shadow"]
        adjusted = [r for r in live if r.base_condition is not None]
        trees = [_tree(name, rules, column, base=base)
                 for name, rules, column, base in (("live", live, "fired", False), ("shadow", shadow, "fired", False),
                                                   ("overlay_base", adjusted, "fired_base", True)) if rules]
        return ctx.expand(self, flow(*trees, firing_step(self.rules), name=self.name))
