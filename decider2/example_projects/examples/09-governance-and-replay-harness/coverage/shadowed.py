"""Shadowed logic: rules and nodes that can never fire regardless of data.

Spec §5.8: "This is a structural property and should be answerable without
running anything." That sentence is a demand on the framework, and decider2 can
almost meet it.

WHY IT IS ALMOST POSSIBLE
  doc 08 §3's interior vocabulary is CLOSED: 13 operators, three `cases`
  variants, composite and/or/not trees, and - critically - §3.2 removes the
  open expression language, so a rule's leaves are declared features or
  registered feature ids, never arbitrary arithmetic. A closed vocabulary over
  declared, typed features is a decidable fragment for exactly the cases that
  matter: interval constraints on numerics and set constraints on categoricals.

  So `shadowed()` builds, per ruleset, the accumulated feasible region after
  each rule in priority order, and reports any rule whose condition has empty
  intersection with what remains. No data, no execution, no sampling. It runs
  in the static plane, over the manifest, for all eight flows in one pass.

WHERE IT STOPS, AND IT SAYS SO RATHER THAN GUESSING
  Three constructs leave the fragment:
    - a condition comparing two features to each other (flow 04's node 7:
      `settlement_ratio <= estimated_instalment_to_income`). Linear relations
      between variables need an LP, not intervals. Decidable, but not cheaply
      at 9 200 nodes; currently reported `unknown`.
    - a condition over a registered derived feature whose body is a step
      (doc 08 §3.2). The step is opaque Python; its range is not declared.
      THIS IS A FRAMEWORK GAP: if `@step` could carry a declared output range
      (`-> float @ range(0, 1)`) the fragment would extend to most of them.
      FRAMEWORK-DEMANDS D14.
    - string matching and large `is_in` sets over codes whose universe is not
      declared. Fixed by the input inventory declaring the category set, which
      spec §4.6 requires anyway.

  `unknown` is reported as `unknown`, never as `not shadowed`, following doc 04
  §3's convention for lineage queries crossing a `@breaks_lineage` boundary.
  A governance tool that guesses in the safe-looking direction is worse than
  one that admits a gap, because the gap is then invisible.
"""

from __future__ import annotations

from manifest.model import GovernanceManifest


class Shadow:
    element: str                   # "CAP-0361" or "flow04:tree23:node17"
    shadowed_by: tuple[str, ...]   # the earlier elements whose conditions subsume it
    verdict: "Literal['unreachable', 'partially_shadowed', 'unknown']"
    why: str                       # a sentence: "CAP-0240 reduces to R30 000 for enquiries >= 6;
                                   #  CAP-0455 reduces to R45 000 for enquiries >= 8, which is a
                                   #  subset. CAP-0455 can never bind."
    fragment_exit: str | None      # named when verdict is `unknown`


def shadowed(m: GovernanceManifest) -> tuple[Shadow, ...]:
    """Static. Runs over every ruleset, every tree and every branch in one pass."""
    pass  # accumulate feasible regions in priority order; intersect; report empties and unknowns


def would_shadow(m: GovernanceManifest, candidate: "RuleDraft") -> Shadow | None:
    """The version a rule author actually wants: before you add CAP-0455, does
    anything already subsume it? Runs at authoring time against the interior
    document, in milliseconds, with no population. This is the only part of the
    harness a flow team touches daily, and it is the reason they tolerate the
    rest of it."""
    pass
