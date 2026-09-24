"""Shared intermediates (spec 10 §5.21): values produced in one phase, read in another.

Declared registry (documentation, checked by `tests/test_shared_intermediates.py`
against what the pipeline actually wires) for the twelve values with five or
more consumers (10 §5.21's table). This is the "connective tissue no isolated
specification can show" -- SCOPE.md asks for this registry "including one
concept with two live versions", which is `existing_obligations` (§5.21.1),
implemented for real in the loop L1 path: see `retail_credit.consolidation`
and `pipeline.py`'s `_l1_loop_body` for the actual and hypothetical columns
(`existing_obligations` vs `existing_obligations_hypothetical`, each carrying
its own `value_basis_code`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SharedIntermediate:
    name: str
    produced_in: str        # phase code, e.g. "P06"
    consumed_in: tuple[str, ...]
    has_two_live_versions: bool = False  # §5.21.1: actual + hypothetical, simultaneously


REGISTRY: tuple[SharedIntermediate, ...] = (
    SharedIntermediate("decision_date", "P01",
                        ("P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11",
                         "P12", "P13", "P14", "P15", "P16", "P17", "P18")),
    SharedIntermediate("net_monthly_income", "P06",
                        ("P07", "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18")),
    SharedIntermediate("risk_grade", "P08",
                        ("P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17")),
    SharedIntermediate("adjustment_set_id", "P08",
                        ("P09", "P10", "P12", "P13", "P14", "P15", "P16", "P17", "P18")),
    SharedIntermediate("existing_obligations", "P06",
                        ("P07", "P09", "P10", "P13", "P14", "P15", "P17"), has_two_live_versions=True),
    SharedIntermediate("segment_code", "P06",
                        ("P07", "P08", "P09", "P10", "P11", "P12", "P16")),
    SharedIntermediate("max_affordable_instalment", "P10",
                        ("P12", "P13", "P14", "P15", "P16", "P17")),
    SharedIntermediate("amount_cap", "P09", ("P11", "P12", "P13", "P14", "P16", "P17")),
    SharedIntermediate("fraud_verdict_code", "P05", ("P09", "P11", "P14", "P16", "P17", "P18")),
    SharedIntermediate("instalment", "P12", ("P10", "P13", "P14", "P16", "P17", "P18")),
    SharedIntermediate("probability_of_default", "P08", ("P09", "P12", "P15", "P16", "P17")),
    SharedIntermediate("nominal_annual_rate", "P12", ("P13", "P14", "P16", "P17", "P18")),
)


def consumers_of(name: str) -> tuple[str, ...]:
    for si in REGISTRY:
        if si.name == name:
            return si.consumed_in
    raise KeyError(name)


def two_version_concepts() -> tuple[str, ...]:
    return tuple(si.name for si in REGISTRY if si.has_two_live_versions)
