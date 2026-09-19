"""Stage 5.6 -- affordability, consumed whole from project 02.

This project does not compute affordability.  It consumes project 02's
published assessment.  The interesting requirement is the last clause of §5.6:

    "Those version identifiers travel into this project's decision record
     UNCHANGED -- a replay of a granting decision must pin the affordability
     tables as tightly as its own."

A consumed capability's table versions are not this project's to re-resolve,
and they are not optional to carry.  `consumes=` declares the contract and
`carries_versions=` makes the version identifiers part of THIS module's
declared outputs -- so they are in the wiring, in the schema, in the frozen
contract and in the audit record automatically.  A consumer that forgets to
carry them fails the build, rather than failing Internal Audit's next sample
with "a table version was not recorded", which core-library §11.9 records as
how the requirement came about in the first place.
See FRAMEWORK-DEMANDS #23.
"""

from __future__ import annotations

from decider2 import consumes, module
from decider2.money import Money

Consume = consumes(
    capability="project02.affordability_assessment",
    version=">=3.1,<4.0",
    name="affordability",
    reads=["risk_grade", "product_code", "credit_life_substitution_declared"],
    writes=["max_affordable_instalment", "discretionary_income",
            "affordability_verdict_code", "living_expenses", "existing_obligations"],
    # Verbatim, unchanged, into this project's record.
    carries_versions=["expense_norm_table_version", "buffer_table_version",
                      "obligation_treatment_matrix_version", "tax_table_version",
                      "affordability_adjustment_set_id"],
    # Three verdicts, three different treatments.  None of them is "continue".
    #   fail          -> decline, reason 1310
    #   marginal      -> approve with conditions, offer set limited to terms <= 24
    #   indeterminate -> refer, queue 5
    verdict_treatments={
        "fail": ("decline", 1310),
        "marginal": ("approve_with_conditions", {"max_term_months": 24}),
        "indeterminate": ("refer", 5),
    },
    # The affordability test is performed against the LOWER instalment where the
    # client substituted their own credit life policy, and the record states
    # that it was.  So substitution is an input to this contract, not a
    # post-hoc adjustment -- otherwise the search would maximise against the
    # wrong ceiling.
)
