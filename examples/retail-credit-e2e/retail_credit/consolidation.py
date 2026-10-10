"""Consolidation: when the client cannot afford what they asked for, test whether settling their other debts would help.

The affordability steps run a second time on `existing_obligations_hypothetical`; both results are kept, and
`offer_basis` picks the one the offer is solved and validated against.
"""
from __future__ import annotations

from decikit import missing_as, step

from credit_core.affordability import MARGINAL, PASS
from credit_core.vocab import FraudVerdictCode
from retail_credit.affordability import FAIL, capacity_unit
from retail_credit.vocab import BASIS_ACTUAL, BASIS_HYPOTHETICAL

HYPOTHETICAL_OUTPUTS = ("discretionary_income", "capacity", "pre_buffer", "affordability_binding_constraint_code",
                        "max_affordable_instalment", "instalment_cap", "affordability_verdict_code")


# ponytail: one scenario (settle every external debt, keep the internal ones). A search over subsets of debts proposes a
# new scenario per pass and is where `loop` belongs.
def consolidation_scenario(
    affordability_verdict_code: int, fraud_verdict_code: int, existing_obligations: float, obligations_internal: float,
    is_consolidation_eligible: bool = missing_as(False),
) -> tuple[bool, float]:
    """(consolidation_proposed, existing_obligations_hypothetical): only an eligible client who fails affordability,
    and whom the fraud check has not stopped, is offered a scenario. Otherwise the hypothetical is the actual figure."""
    proposed = (affordability_verdict_code == FAIL and is_consolidation_eligible
                and fraud_verdict_code == FraudVerdictCode.APPROVE)
    return proposed, obligations_internal if proposed else existing_obligations


def offer_basis(
    consolidation_proposed: bool, affordability_verdict_code: int, instalment_cap: float,
    affordability_verdict_code_hypothetical: int, instalment_cap_hypothetical: float,
) -> tuple[int, int, float]:
    """(offer_basis_code, offer_affordability_verdict_code, instalment_ceiling): the hypothetical basis only where a
    scenario was proposed and it clears affordability."""
    if consolidation_proposed and affordability_verdict_code_hypothetical in (PASS, MARGINAL):
        return BASIS_HYPOTHETICAL, affordability_verdict_code_hypothetical, instalment_cap_hypothetical
    return BASIS_ACTUAL, affordability_verdict_code, instalment_cap


def hypothetical_capacity_unit():
    """`capacity_unit` on `existing_obligations_hypothetical`, every output suffixed `_hypothetical`."""
    return capacity_unit("capacity_hypothetical").relabel(
        reads={"existing_obligations": "existing_obligations_hypothetical"},
        writes={name: f"{name}_hypothetical" for name in HYPOTHETICAL_OUTPUTS})


consolidation_scenario_step = step(consolidation_scenario,
                                   outputs=("consolidation_proposed", "existing_obligations_hypothetical"))
offer_basis_step = step(offer_basis, outputs=("offer_basis_code", "offer_affordability_verdict_code",
                                               "instalment_ceiling"))
