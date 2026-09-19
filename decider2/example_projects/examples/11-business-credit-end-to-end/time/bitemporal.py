"""Bi-temporal entity structure. Spec 11 H3 / 5.13 / 13-Q4.

The problem in one line: both temporal views are plausible, neither errors, and
they disagree on about 9% of assessments (spec 5.3 O2).

    replay needs        known_from <= knowledge_date
    an ownership-change
    covenant needs      effective_from <= as_at < effective_to

Doc 03 5.2's answer to "the same module against two sources" is `.at()`, an
instance relabel. That is exactly wrong here, because `.at()` produces two
modules reading two names and a reviewer cannot see from the *use site* which
temporal semantics they got -- the relabel lives in the composition file.

What this project does instead: the two views live in **different value
namespaces**, and there is no bare name. A step asking for `entities` does not
get a default; it gets an unbound-input error naming both candidates:

    step 'ownership_change_test' input 'entities' is not produced by any step
    in scope. `entity_facts` is bi-temporal and has no unqualified view.
    Did you mean 'known.entities' (replay view, known_from <= knowledge_date)
                or 'actual.entities' (world view, effective_from <= as_at)?
    Spec 5.13.2 says which you want.

That turns spec 10's acceptance 17 -- "every phase that reads structure declares
which view it uses; undeclared reads are caught at build time" -- from a check
somebody has to write into a property of name resolution.
"""

from decider2 import bitemporal_source, view

KNOWN = view(
    "known",
    predicate="known_from <= :knowledge_date and effective_from <= :decision_date",
    doc="What the Bank knew when it decided. The replay view (09 5.15 item 4).",
)

ACTUAL = view(
    "actual",
    predicate="effective_from <= :as_at < effective_to",
    doc="What was true in the world, regardless of when the Bank learned it.",
)

EntityFacts = bitemporal_source(
    "entity_facts",
    effective=("effective_from", "effective_to"),
    knowledge="known_from",
    provenance=["source_code", "fact_kind"],
    views={"known": KNOWN, "actual": ACTUAL},
    unqualified=None,        # <- there is no default view. This line is the design.
    volumes={"attachments": 1_116_000, "changes_per_month": 34_000,
             "median_lag_days": 47, "p95_lag_days": 310},
)


def structure_delta(known_entities: list, prior_decision_entities: list) -> list:
    """Entities added, removed, ownership moved, roles and criticality changed.

    New in this project (spec 5.3 O2 "Records"): O2 in an origination emits a
    structure; O2 inside a lifecycle emits a structure AND its delta against the
    previous decision of record. The delta is an input to L1's cause
    decomposition and to L3's signals, so it is a first-class output, not a
    diff somebody computes in a report.
    """
    pass  # pair on entity_key across the two structures; classify each pairing


def late_fact_retests(facts: list, tests_already_run: list) -> list:
    """Facts arriving with an effective date behind a decision already made.

    4% of facts land here (spec 4.4). Spec 5.13.3(3): this produces a **re-test**
    with a new determination date, never a silent correction. The original test
    stands as the test that was performed; the difference between the two is the
    Bank's knowledge, and that difference is itself the finding.
    """
    pass  # emit (covenant_test_id, reason=LATE_FACT) for each affected past test
