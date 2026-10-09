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

from credit_core.adjustments import Overlay
from credit_core.reason_codes import Code
from decider import ConfigurableStep
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
    overlay_documents: tuple[str, ...]  # the config documents that hold its overlays, in the order they are listed
    detail: tuple[str, ...]           # the fields an analyst reads to see how it was decided
    rules: Callable[[dict], set[str]]  # every rule id, from the config documents
    fired: Callable[[dict], set[str]]  # the rule ids one response says bound

    def project(self) -> Path:
        return Path(pkgutil.resolve_name(self.entry.partition(":")[0]).__file__).parents[1]

    def store(self) -> JsonFileStore:
        return JsonFileStore(basepath=str(self.project() / "configs"))

    def documents(self, version: str | None = None) -> dict:
        """The config documents of `version` (default: latest) as stored; treat them as read-only.

            LOAN.documents()["reason_codes"]["version"]
        """
        store = self.store()
        return _documents(store.basepath, str(version or store.latest_version()))

    def overlays(self, version: str | None = None) -> list[Overlay]:
        """Every overlay in the overlay documents of `version`, in document order.

            [o.adjustment_id for o in FRAUD.overlays()]
        """
        loaded = (ConfigurableStep.load(self.documents(version)[name]) for name in self.overlay_documents)
        return [o for doc in loaded for o in getattr(doc, "stack", doc).adjustments]

    def reason_codes(self, version: str | None = None) -> dict[int, Code]:
        """The reason codes of `version` (default: latest) by code.

            LOAN.reason_codes()[5402].description
        """
        store = self.store()
        return _reason_codes(store.basepath, str(version or store.latest_version()))

    def replayer(self, version: str | None = None, mode: str = "fused", store: JsonFileStore | None = None,
                 params: dict | None = None) -> Replayer:
        """The flow staged at `version` (default: latest) in `store` (default: its own configs).

            fraud.replayer().served(request)
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
        return self.replayer(version).pipeline_fn(self.documents(version))


@functools.cache
def _documents(basepath: str, version: str) -> dict:
    return JsonFileStore(basepath=basepath).read(version).config


@functools.cache
def _reason_codes(basepath: str, version: str) -> dict[int, Code]:
    return {c.code: c for c in ConfigurableStep.load(_documents(basepath, version)["reason_codes"]).codes}


@functools.cache
def _staged(entry: str, basepath: str, version: str, mode: str, sample: str) -> Replayer:
    handler = Replayer(JsonFileStore(basepath=basepath), entry, mode, sample_request=sample)
    handler.stage(version)
    handler.activate()
    return handler


FRAUD = Flow(
    "fraud", "fraud_interdiction.pipeline:build", "action_code", "decline_reason_codes", "primary_reason_code",
    ("amount_overlays", "action_escalation"),
    ("fired_rule_ids", "unevaluable_rule_ids", "fired_on_overlay_ids", "counterfactual_fired_rule_ids",
     "counterfactual_action_code", "action_source_rule_id", "hard_block_code", "degraded_mode_code"),
    _fraud_rules, lambda r: set(r["fired_rule_ids"]))

LOAN = Flow(
    "loan", "loan_granting.pipeline:build", "outcome_code", "decline_reason_codes", "primary_reason_code",
    ("score_adjustments", "calibration_scale_adjustments", "probability_of_default_adjustments",
     "probability_of_default_for_grading_adjustments"),
    ("eligibility_gate_verdicts", "amount_cap", "amount_cap_binding_rule_id", "amount_cap_chain", "score",
     "risk_grade", "max_affordable_instalment", "suppressed_terms"),
    _loan_rules, _loan_fired)

BUSINESS = Flow(
    "business", "business_nested.pipeline:build", "outcome_code", "decline_reason_codes", "primary_reason_code",
    ("event_threshold_overlays", "entity_score_adjustments", "business_pd_adjustments"),
    ("attributing_entity_id", "attributing_event_ids", "attributing_rule_id", "risk_grade", "resolved_entity_count"),
    _business_rules, lambda r: {e["adverse_binding_rule"] for e in r["assessed_entities"]})

FLOWS = {f.code: f for f in (FRAUD, LOAN, BUSINESS)}
