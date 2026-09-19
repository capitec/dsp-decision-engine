"""Champion/challenger — deterministic, storage-free, and not an overlay.

Spec §5.12 is unusually prescriptive here and each sentence is a constraint:

  "by a deterministic function of account_id and a per-experiment salt"
      -> `stable_hash64(account_id, salt)`. See the note on hashing below.

  "is the same every day for the life of the experiment"
      -> nothing in the function varies with decision_date.

  "is recomputable without storing it"
      -> the assignment is a step, run in the batch and in the live-call path,
         and no table holds it.

  "is uncorrelated across concurrent experiments"
      -> different salts, and a registered test asserts pairwise chi-square
         independence over a 500k reference sample at registration time.

  "Cohort is an ACCOUNT-LEVEL property and does not change when the account
   rolls — reassigning on roll would select challengers on outcome."
      -> the function takes `account_id` and nothing else. Not bucket, not
         score, not balance. A framework that made it easy to add another
         argument here would be dangerous; the signature is the safeguard.
"""

from decider2 import Experiment, ExperimentRegistry, module, param, stable_hash64, step
from decider2.types import Date, i1, i2, i8, u8


@step(output="cohort_code")
def assign_cohort(account_id: i8, registry=None) -> i2:
    """Deterministic assignment across up to 6 concurrent experiments.

    WHY NOT `hash()`. CPython's built-in hash is salted by PYTHONHASHSEED. An
    experiment assigned with it is stable within one process and changes on every
    restart — so a unit test passes, a batch run is self-consistent, and the
    champion/challenger split silently re-randomises between the 05:00 batch and
    the 13:30 re-run. `stable_hash64` is a framework primitive with a fixed,
    documented, version-pinned algorithm for exactly this reason.
    FRAMEWORK-DEMANDS #13.
    """
    pass


REGISTRY = ExperimentRegistry(
    max_concurrent=6,
    max_per_account=2,
    assignment_order="registration_order",   # deterministic when an account
                                             # qualifies for three
    interaction_recorded=True,
    experiments=[
        Experiment(
            id=41, name="bucket3_negotiator_early",
            salt="c1f9a2e0",
            arms={"champion": 90, "challenger": 10},
            eligible={"arrears_bucket_code": [3], "product_family_code": [1, 2]},
            opens="2026-08-01", closes="2026-11-30",
            measures=["roll_rate_30d", "cure_rate_30d", "cost_per_rand_collected"],
            approval_reference="CS-EXP-2026-07-24",
        ),
        Experiment(
            id=42, name="sms_cadence_3v4_day",
            salt="7b3d10c4",
            arms={"champion": 50, "challenger": 50},
            eligible={"arrears_bucket_code": [1, 2]},
            opens="2026-09-01", closes="2026-12-15",
            measures=["promise_rate", "complaint_rate", "cost_per_rand_collected"],
            approval_reference="CS-EXP-2026-08-19",
        ),
        Experiment(
            id=43, name="settlement_offer_bucket6",
            salt="9e02f7aa",
            arms={"champion": 80, "challenger": 20},
            eligible={"arrears_bucket_code": [6], "balance_band_code": [1, 2, 3]},
            opens="2026-06-15", closes="2026-10-15",
            measures=["net_recovery_24m", "settlement_take_up"],
            approval_reference="CS-EXP-2026-06-02",
        ),
    ],
    holdout=Experiment(
        id=99, name="standing_holdout",
        salt="fixed_do_not_rotate",
        arms={"holdout": 0.5},
        eligible={"arrears_bucket_code": [1, 2, 3, 4, 5]},   # excluded above bucket 5
        opens="2026-01-01", closes="2027-06-30",             # an expiry date, mandatory
        treatment_override="minimum_contact",
        approval_reference="CCF-2025-12-08",
    ),
)


@step(output="cohort_interaction_code")
def record_interaction(cohort_code: i2) -> i2:
    """An account in two experiments records the pair, because a challenger
    result computed without knowing the other arm the account was in is an
    average over an uncontrolled factor."""
    pass


# ---------------------------------------------------------------------------
# COHORTS ARE NOT OVERLAYS, and the design must make conflating them hard.
#
# Spec §5.12: "A design that expresses a challenger as 'an overlay scoped to some
# accounts' will produce results nobody can interpret."
#
# Three separations, all structural rather than documentary:
#
#  1. Different artefacts. An Experiment lives here; an overlay lives in
#     overlays/register.json. Neither schema can express the other: an
#     Experiment has `arms` and `measures` and no `magnitude`; an overlay has
#     `magnitude` and `stack_position` and no `arms`.
#
#  2. Different reach. A challenger selects a different MATRIX VERSION (Grid is
#     `cohort_scoped=True`). An overlay modifies a VALUE the matrix produced.
#     A challenger cannot dial intensity and an overlay cannot swap a matrix.
#
#  3. Different neutrality obligations. `cohort_scoped: false` on an overlay
#     triggers the demonstrable-equality check in overlays/guard.py. An overlay
#     that IS cohort-scoped is legal but marks every experiment it touches as
#     `arms_not_comparable`, which propagates to the experiment result as
#     "uninterpretable" rather than as a small difference (spec §5.12).
# ---------------------------------------------------------------------------

Cohorts = module(assign_cohort, record_interaction, name="cohorts",
                 taps=["cohort_code", "cohort_interaction_code"])
