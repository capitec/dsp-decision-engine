"""The final `decline_reason_codes` waterfall: `outcome.business_outcome_step` writes it
first, `core.reason_codes`'s registry re-writes it ranked -- a same-name waterfall, so it
must be a `flow`, not a `dag` (`pricing.py`'s `_instalment_unit()` docstring explains the
same rule for `instalment`)."""
from __future__ import annotations

from decider import flow

from business_nested import outcome


def build_reasons_unit():
    return flow(outcome.business_outcome_step, outcome.REASON_REGISTRY.resolve_step(), name="reasons").emit(
        "outcome_code", "attributing_entity_id", "attributing_event_ids", "business_decline_binding_rule_actual",
    )
