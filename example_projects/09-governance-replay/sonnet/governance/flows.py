"""Adapters over the three flows this harness runs against (SCOPE.md: 01, 03, 05 --
"real-time rules", "solve", "nested" -- three different evidence shapes).

Each `FlowAdapter` knows how to build a bound `Executable` for a declared
config version, which raw fields are dates (JSON has no date type), which
output field carries the outcome and the reasons, and which
`credit_core.adjustments.AdjustmentRegister`s it owns (for the overlay
register, §5.14). This is the one place a ninth flow would be added
(acceptance §10 item 16: "with no change to the harness" -- true here only
in the sense that adding a flow is one more `FlowAdapter`, not a change to
`replay.py`/`explain.py`/etc., which are all written against this
interface, never against a specific flow).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from decider import Engine
from decider.steps.tables import DecisionTableConfig
from decider.steps.trees import TreeConfig

from governance import paths


@dataclass(frozen=True)
class BuiltFlow:
    """One flow, bound at one config version: ready to `.score()` or `.run()`.

    `executable` is whatever `Engine().bind(...)` returns (`decider.engine.Engine.bind`'s
    own docstring, not `decider`'s internals, is the contract this relies on: `.score()`,
    `.run()`, `.plan.versions`)."""

    code: str
    executable: object
    params: dict
    config_version: str


@dataclass(frozen=True)
class FlowAdapter:
    code: str
    name: str
    project: str                      # example_projects/<project> directory name
    module_name: str                  # distinct name to load this flow's pipeline.py under
    extra_projects: tuple[str, ...]   # sibling projects that must also be on sys.path
    config_files: tuple[str, ...]     # ConfigurableStep documents build() takes positionally, in order
    date_fields: tuple[str, ...]      # top-level fields that are dates, not strings, once loaded
    outcome_field: str                # the field a swap-set/replay compares as "the decision"
    reason_field: str = "decline_reason_codes"
    primary_reason_field: str = "primary_reason_code"
    registry_version_field: str = "reason_registry_version"
    typed: Callable[[dict], dict] | None = None   # extra, flow-specific JSON->Python typing (nested dates)

    def project_dir(self) -> Path:
        return paths.sibling_project(self.project)

    def build(self, config_version: str = "0.1.0", *, mode: str = "interpreted") -> BuiltFlow:
        project_dir = self.project_dir()
        for extra in self.extra_projects:
            paths.ensure_on_path(paths.sibling_project(extra))
        paths.ensure_on_path(project_dir)
        module = paths.load_pipeline(project_dir, self.module_name)

        config_dir = project_dir / "configs" / config_version
        configs = [_load_config_document(config_dir / f"{name}.json") for name in self.config_files]
        pipeline_step = module.build(*configs)
        executable = Engine().bind(pipeline_step, mode=mode)
        params = pipeline_step.parameters().defaults()
        return BuiltFlow(code=self.code, executable=executable, params=params, config_version=config_version)

    def type_record(self, record: dict) -> dict:
        """JSON has no date type; convert this flow's declared date fields (and any
        flow-specific nested ones) from ISO strings to `datetime.date` before scoring."""
        record = dict(record)
        for field_name in self.date_fields:
            if record.get(field_name) is not None:
                record[field_name] = datetime.date.fromisoformat(record[field_name])
        if self.typed is not None:
            record = self.typed(record)
        return record


def _load_config_document(path: Path):
    """A `ConfigurableStep` document's own `type` tag says which subclass to load with --
    `TreeConfig.load`/`DecisionTableConfig.load` both work from a path directly (the same
    call every one of 01/03/05's own tests make), so this only needs to pick between them."""
    import json

    doc_type = json.loads(path.read_text())["type"]
    if doc_type == "tree":
        return TreeConfig.load(str(path))
    if doc_type == "decision_table":
        return DecisionTableConfig.load(str(path))
    raise ValueError(f"{path}: unknown ConfigurableStep type {doc_type!r}")


def _type_business_nested(record: dict) -> dict:
    """05's own nested date fields (`entities[*].adverse_events[*].event_date`) -- the
    same conversion `05`'s own `tests/test_pipeline.py::_typed` does."""
    record = dict(record)
    entities = [dict(e) for e in record.get("entities", [])]
    for entity in entities:
        events = [dict(ev) for ev in entity.get("adverse_events", [])]
        for ev in events:
            if ev.get("event_date") is not None:
                ev["event_date"] = datetime.date.fromisoformat(ev["event_date"])
        entity["adverse_events"] = events
    record["entities"] = entities
    return record


def _type_loan_granting(record: dict) -> dict:
    """03's own nested date fields (bureau/internal account lists' `opened_date`)."""
    record = dict(record)
    for list_field in ("bureau_accounts", "internal_accounts", "applicant2_bureau_accounts",
                        "applicant2_internal_accounts"):
        accounts = [dict(a) for a in record.get(list_field, []) or []]
        for a in accounts:
            if a.get("opened_date") is not None:
                a["opened_date"] = datetime.date.fromisoformat(a["opened_date"])
        record[list_field] = accounts
    return record


FRAUD = FlowAdapter(
    code="01", name="Transaction fraud interdiction", project="01-transaction-fraud",
    module_name="flow01_pipeline", extra_projects=("00-shared-credit-core",),
    config_files=("live_rules", "shadow_rules", "overlay_base_rules"),
    date_fields=("decision_date",), outcome_field="action_code",
)

GRANTING = FlowAdapter(
    code="03", name="Unsecured granting and pricing (Flex Loan)", project="03-loan-granting-pricing",
    module_name="flow03_pipeline", extra_projects=("00-shared-credit-core", "02-affordability"),
    config_files=("rate_card_flex_loan",),
    date_fields=("decision_date", "bureau_as_of_date"), outcome_field="outcome_code",
    typed=_type_loan_granting,
)

BUSINESS = FlowAdapter(
    code="05", name="Business nested-entity credit", project="05-business-nested",
    module_name="flow05_pipeline", extra_projects=("00-shared-credit-core", "02-affordability"),
    config_files=(), date_fields=("decision_date",), outcome_field="outcome_code",
    typed=_type_business_nested,
)

ALL: dict[str, FlowAdapter] = {a.code: a for a in (FRAUD, GRANTING, BUSINESS)}


def get(flow_code: str) -> FlowAdapter:
    try:
        return ALL[flow_code]
    except KeyError:
        raise KeyError(f"unknown flow_code {flow_code!r}; known: {sorted(ALL)}") from None
