"""Version pinning per ASSESSMENT, not per deployment. Spec 5.17.4, 13-Q16.

    "A 2027 covenant test must replay against the component versions in force
     in 2027, while today's origination uses today's."

Doc 08 4 has one active generation and at most one staging, selected by a
pointer swap. That is right for a config change and wrong for this: here three
different component versions must be live *simultaneously inside one process*,
selected per row --

    EP-1 today            -> credit.affordability major 3
    EP-3 cohort in flight -> credit.affordability major 2 (pinned at batch start)
    2027 replay           -> credit.affordability major 2, patch 2.4.1

---------------------------------------------------------------------------
The mechanism, stated plainly because it is ugly
---------------------------------------------------------------------------
A pinned component is a **Branch over the manifest column**, with one arm per
live major, and every arm compiled into the image at build.

    Affordability = Branch(pinned_major("p02"), [P02_v2, P02_v3], modifies=[...])

Consequences, all deliberate:

  + It renders. Both majors appear in the reviewable artefact, and the diff at
    re-approval shows a major being added or retired as a structural change,
    which is what it is.
  + It replays. The manifest is an INPUT column resolved from the decision
    record, so a replay needs no deployment archaeology and no old container.
  + It is bounded. `majors=` is a declared list; a third major is a code change
    with a review, not a config edit, and spec 10 acceptance 26 is demonstrable
    in one process.
  - It doubles compiled code for every pinned component, and the fused-kernel
    non-monotonicity of doc 02 1.1 means the arms must NOT be fused together.
    `parallel=False, fuse=False` on the pin branch is not an optimisation, it is
    a correctness-of-budget requirement. FRAMEWORK-DEMANDS D6.
  - Retiring a major is a seven-year problem, not a release. Spec 5.17.4's last
    line: a component version later overwritten in place is not a version.

---------------------------------------------------------------------------
Batch pinning
---------------------------------------------------------------------------
Spec 5.15.2 requirement 2 and 8.2 "batch version pinning": a batch in flight
pins every table, component and flow version at its start and holds them to
completion. `manifest.freeze(at=batch_start)` returns an immutable manifest that
becomes a constant column for the whole cohort. A version change mid-batch
either does not apply or aborts and restarts -- it never applies to part of it.
"""

from decider2 import component_manifest, pinned_major

MANIFEST = component_manifest(
    "component_versions",
    source="decision_of_record.component_versions",   # an input, not a deployment fact
    components={
        "core":  {"majors": [4, 5], "cadence": "release train", "owner": "credit_systems"},
        "p02":   {"majors": [2, 3], "cadence": "on gazette",    "owner": "compliance"},
        "p05":   {"majors": [3, 4], "cadence": "quarterly",     "owner": "credit_committee"},
        "p06":   {"majors": [2],    "cadence": "semi-annual",   "owner": "credit_committee"},
        "p07":   {"majors": [1],    "cadence": "monthly",       "owner": "portfolio_mgmt"},
    },
    # Spec 8.2: retrieval of any one within 5 seconds, for seven years.
    retention="7 years after facility closure",
    # The resolution rule for components is the POLICY rule (by determination
    # date). The covenant definition inside the same replay resolves by the
    # CONTRACT rule. Two mechanisms in one replay (spec 5.17.4 scenario 3),
    # and time/dating.py is why they cannot be swapped by accident.
    resolution="determination_date",
)

p02_major = pinned_major(MANIFEST, "p02")
p05_major = pinned_major(MANIFEST, "p05")
core_major = pinned_major(MANIFEST, "core")


def freeze_for_batch(batch_start_date) -> dict:
    """Pin every component, table and flow version for an 8-hour review cohort.

    Spec 5.15.1 collision 1: Treasury reissues the asset finance rate card on
    the 1st and Policy changes the DSCR thresholds on the 1st, both landing in
    the same monthly batch, and the batch's results are then attributable to
    neither. Freezing does not fix that -- the change calendar does (spec 5.15.2
    requirement 5) -- but it makes the collision *visible*, because the frozen
    manifest names both versions and the swap-set can be run against them.
    """
    pass  # snapshot every version id; return an immutable manifest constant
