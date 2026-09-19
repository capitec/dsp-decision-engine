"""The reuse surface. Everything this project consumes rather than writes.

Spec 4.9 inventories it; this directory IS that inventory, as code. The single
most useful property of the layout is negative:

    grep -rn "^from consumed" ..     ->  the complete list of consumption sites
    grep -rn "^from modules"  ..     ->  the complete list of local logic

A consumed component never appears at a use site under its own import path. It
appears as a binding declared here, so that the answer to "what do we depend on
and how badly" is one directory listing rather than a dependency graph nobody
runs. 1 280 of this project's 1 900 decision points live behind these files.

---------------------------------------------------------------------------
What a binding carries, and why all five parts are mandatory
---------------------------------------------------------------------------
  component   the published id, e.g. "credit.entity_assessment@05"
  majors      which major versions this project keeps compiled. Plural, always.
              Spec 5.17.4 scenario 1: EP-1 moves to a new major on release while
              an in-flight EP-3 cohort and every replay stay on the old one.
              That is not a deployment property -- it is a per-assessment one --
              so both majors are in the graph and the manifest branches.
  roles       the role bindings that apply at THIS consumption site (roles.py)
  gaps        the declared resolutions for what the component does not produce.
              A local step writing a name a consumed component declares, with no
              `gap()` covering it, is a BUILD ERROR. That is the mechanism spec
              13-Q14 asks for: an undeclared gap cannot exist quietly.
  pin         where the version comes from per assessment (consumed/manifest.py)

---------------------------------------------------------------------------
How a consumed component looks at its use site
---------------------------------------------------------------------------
Deliberately different from local logic, in three visible ways:

  1. It is `Consumed`, not `Module`. It has no interior document, cannot be
     given params by this project's config, and `decider export --interiors`
     skips it. A policy analyst editing this project's rules cannot reach inside
     project 05's roll-up, which is correct: that roll-up is approved by a
     different Credit Committee paper.
  2. It renders in the reviewable artefact with its OWNER and its APPROVAL
     REFERENCE in the header, and its body rendered from the owner's published
     description rather than from this project's. Spec 9.4 requirement 2 --
     "sliceable by owner" -- is then free, because ownership is on the node.
  3. Its pinned version is printed at the use site in the rendered view:
     `p05.entity_assessment @ 4.2.1 (pinned by manifest)`. A reviewer reading
     the 2031 render of a 2027 decision sees 3.9.0 there, not 4.2.1.
"""

from decider2 import consumed_surface

SURFACE = consumed_surface(
    library=("credit-core@00", 22),          # all 22 capabilities consumed (spec 4.9)
    projects={
        "02": ("credit.affordability", 1),
        "05": ("credit.business_origination", 13),
        "06": ("credit.restructure", 3),
        "07": ("credit.limit_management", 1),
        "09": ("governance.evidence_contract", 23),   # an obligation, not a component
    },
    decision_points_consumed=1_280,
    decision_points_total=1_900,
    # Spec 5.17.7's six indicators, collected rather than argued about.
    thresholds={
        "identity_relabels": 40,
        "gaps_resolved_by_compose": 0.5,      # of the eight in spec 5.17.2
        "forks": 2,
        "releases_blocked_on_another_cadence": 4,
        "locally_retested_share": 0.15,
        "consumer_correctness_defects_per_year": 6,
    },
)
