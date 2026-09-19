"""Ranking bases — two of them, switchable by a parameter, both registered.

Spec §5.9: "Ranking basis is a parameter, not a constant. Two bases must both be
expressible and switchable by Collections Strategy without a release."

The obvious implementation is a config field holding a function pointer, and doc
08 §1.1 forbids exactly that. So the basis is a `Choice` over REGISTERED steps,
selected by an enum param: the ids are checked at validation with a did-you-mean,
both bases appear in `lineage()` and `render()` whichever is selected, and the
one that ran is recorded. Switching is a value change — free, no compile.
"""

from decider2 import Choice, module, param, step
from decider2.types import cents, f8, i1, i2, i4

from ..settlement.justification import discounted_net_recovery


@step(description="Marginal expected rand from applying this treatment to this "
                  "account today, net of its cost.")
def marginal_value_rank_key(
    recovery_estimate: cents,
    cost_to_collect: cents,
    treatment_unit_cost: cents,
    uplift_probability: f8,
    discount_rate_pa: f8 = param(0.12, ge=0.0, le=0.5, owner="credit_risk_policy"),
) -> f8:
    """The value basis. Accumulates in float64, never int64 — a cents
    accumulator over 2.3M balances wraps (doc 03 §1.2)."""
    pass


@step(description="Policy priority ordering. Pre-prescription legal candidates "
                  "and first-time bucket-3 entrants outrank everything regardless of value.")
def policy_priority_rank_key(
    pre_prescription_flag: bool,
    days_to_prescription: i4,
    is_first_entry_to_bucket_3: bool,
    untouched_days: i4,
    arrears_bucket_code: i1,
    outstanding_balance: cents,
) -> f8:
    """The policy basis. A lexicographic ordering flattened into one f8 by
    declared tier weights, so that ties within a tier still fall through to the
    stable tie-break below rather than to input order."""
    pass


RANK_KEY = Choice(
    param="ranking_basis",
    options={
        1: marginal_value_rank_key,
        2: policy_priority_rank_key,
    },
    default=1,
    owner="collections_strategy",
    emits="ranking_basis_code",     # part of the record (spec §5.9, §5.10)
)


@step(output="rank_tie_break")
def rank_tie_break(account_id: i8) -> i8:
    """Ties are broken on a stable key, never on input order and never on
    anything derived from processing order (spec §5.9 Determinism).

    `stable_hash64` and NOT Python's `hash()`. CPython's hash is PYTHONHASHSEED-
    salted per process, so a tie-break built on it would produce a different
    allocation on every restart while looking perfectly deterministic in a test
    run inside one process. This is the single most plausible way to fail
    acceptance criterion 5 and it fails silently. See FRAMEWORK-DEMANDS #13.

    Sorting on `account_id` directly would be deterministic but would
    systematically favour older accounts on every tie, every day, forever. A
    stable hash is deterministic AND uncorrelated with anything.
    """
    pass  # stable_hash64(account_id, domain="collections.alloc")


Ranking = module(RANK_KEY, rank_tie_break, name="ranking",
                 taps=["ranking_basis_code", "rank_key"])
