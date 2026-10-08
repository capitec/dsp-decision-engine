"""The flows the harness runs against, and everything that differs between them.

Adding a flow is one more `Flow` here; no other module names a flow.
"""
from __future__ import annotations

import copy
import functools
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from decider.config import JsonFileStore

from governance.replayer import Replayer


def _fraud_rules(docs: dict) -> set[str]:
    return {r["rule_id"] for r in docs["rule_set"]["rules"] if r["status"] == "live"}


def _loan_rules(docs: dict) -> set[str]:
    return {rule.rule_id for rule in pkgutil.resolve_name("loan_granting.waterfall:REGISTER")}


def _business_rules(docs: dict) -> set[str]:
    rollup = pkgutil.resolve_name("business_nested.rollup")
    return {getattr(rollup, name) for name in dir(rollup) if name.startswith("AE_R_")}


def _loan_fired(response: dict) -> set[str]:
    return {r["rule_id"] for r in response["waterfall_rules"] if r["status"] in ("bound", "raised", "declined")}


@dataclass(frozen=True)
class Flow:
    code: str
    entry: str
    outcome: str                      # the response field that is "the decision"
    reasons: str                      # the field holding every reason code, most severe first
    primary: str
    registry: str                     # the flow's ReasonCodeRegistry
    registers: tuple[str, ...]        # its overlay registers
    detail: tuple[str, ...]           # the fields an analyst reads to see how it was decided
    rules: Callable[[dict], set[str]]  # every rule id, from the config documents
    fired: Callable[[dict], set[str]]  # the rule ids one response says bound

    def project(self) -> Path:
        return Path(pkgutil.resolve_name(self.entry.partition(":")[0]).__file__).parents[1]

    def store(self) -> JsonFileStore:
        return JsonFileStore(basepath=str(self.project() / "configs"))

    def replayer(self, version: str | None = None, mode: str = "fused", store: JsonFileStore | None = None,
                 params: dict | None = None) -> Replayer:
        """The flow staged at `version` (default: latest) in `store` (default: its own configs).

            fraud.replayer("0.1.0").served(request)
        """
        store = store or self.store()
        handler = copy.copy(_staged(self.entry, store.basepath, str(version or store.latest_version()), mode,
                                    str(self.project() / "sample_request.json")))
        handler.params = copy.deepcopy(params)
        return handler

    def candidate(self, edit: Callable[[dict], None], root: Path, mode: str = "fused") -> Replayer:
        """A new version of the flow's latest config, changed by `edit(config)`, written under `root`.

            loan.candidate(lambda config: config["params"].update(...), tmp_path)
        """
        store = self.store()
        config = copy.deepcopy(store.read(store.latest_version()).config)
        edit(config)
        scratch = JsonFileStore(basepath=str(root))
        scratch.create_version(config)
        return self.replayer(store=scratch, mode=mode)

    def pipeline(self, version: str | None = None):
        store = self.store()
        return self.replayer(version).pipeline_fn(store.read(version or store.latest_version()).config)


@functools.cache
def _staged(entry: str, basepath: str, version: str, mode: str, sample: str) -> Replayer:
    handler = Replayer(JsonFileStore(basepath=basepath), entry, mode, sample_request=sample)
    handler.stage(version)
    handler.activate()
    return handler


FRAUD = Flow(
    "fraud", "fraud_interdiction.pipeline:build", "action_code", "decline_reason_codes", "primary_reason_code",
    "fraud_interdiction.reasons:REGISTRY", ("fraud_interdiction.overlays:REGISTER",),
    ("fired_rule_ids", "unevaluable_rule_ids", "fired_on_overlay_ids", "counterfactual_fired_rule_ids",
     "counterfactual_action_code", "action_source_rule_id", "hard_block_code", "degraded_mode_code"),
    _fraud_rules, lambda r: set(r["fired_rule_ids"]))

LOAN = Flow(
    "loan", "loan_granting.pipeline:build", "outcome_code", "decline_reason_codes", "primary_reason_code",
    "loan_granting.reasons:REGISTRY", ("loan_granting.scoring:SCORE_ADJUSTMENTS",),
    ("eligibility_gate_verdicts", "amount_cap", "amount_cap_binding_rule_id", "amount_cap_chain", "score",
     "risk_grade", "max_affordable_instalment", "suppressed_terms"),
    _loan_rules, _loan_fired)

BUSINESS = Flow(
    "business", "business_nested.pipeline:build", "outcome_code", "decline_reason_codes", "primary_reason_code",
    "business_nested.outcome:REASON_REGISTRY",
    ("business_nested.events:THRESHOLD_OVERLAYS", "business_nested.scoring:ENTITY_OVERLAYS",
     "business_nested.grade:BUSINESS_OVERLAYS"),
    ("attributing_entity_id", "attributing_event_ids", "attributing_rule_id", "risk_grade", "resolved_entity_count"),
    _business_rules, lambda r: {e["adverse_binding_rule"] for e in r["assessed_entities"]})

FLOWS = {f.code: f for f in (FRAUD, LOAN, BUSINESS)}
