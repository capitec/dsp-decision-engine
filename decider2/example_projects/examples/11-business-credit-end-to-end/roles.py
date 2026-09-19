"""Roles. The mapping declared ONCE, not at every use site. Spec 5.17.1, 13-Q13.

This is the project's central invention and the thing it most wants the
framework to take. Read this file before any pipeline.

---------------------------------------------------------------------------
The problem, as the spec states it
---------------------------------------------------------------------------
The library publishes `applicant_age_years`, `gross_monthly_income`,
`existing_obligations`, `max_affordable_instalment` and 27 more (spec 5.17.1:
31 names consumed in a role-dependent sense, 12 of which are materially misread
if the role is dropped). Every one of them was named for a retail applicant.

Here each belongs to somebody with a role, and the role changes the meaning:

  applicant_age_years   director's age | surety's age AT FINAL INSTALMENT DATE
                        (E-AROD-10) | undefined for a corporate guarantor
  gross_monthly_income  a sole proprietor's drawings -- which are THE SAME MONEY
                        as the business's net profit and must be counted once
  existing_obligations  a surety's personal obligations, EXCLUDING the facility
                        they stand surety for, which is not their obligation
                        until it is

`core.income` is called for up to 40 entities. Rewrapping its five outputs per
entity per role is 200 wrappers on one phase (spec 5.17.1). Doc 01 5.1 counted
79 of those in one real project and named it the framework's worst ergonomic
failure. This project would multiply it.

---------------------------------------------------------------------------
What doc 03 offers, and why each one fails here
---------------------------------------------------------------------------
  Name matching (layer 1)   -- fails: one library name, several meanings.
  Vocabulary   (layer 2)    -- fails: a function name->name has one right-hand
                               side; this needs one per role.
  `.at()`      (layer 3)    -- fails three ways. It is per instance, so 40
                               entities x 3 roles is 120 declarations; it lives
                               in the composition file, so a reviewer reading
                               the use site cannot see which semantics applied;
                               and it renames only, so it cannot carry the
                               parameterisation (criticality class) or the
                               null policy (`months_employed` is NOT APPLICABLE
                               to a retired surety, which is doc 00 7.4's
                               missing fourth situation).

---------------------------------------------------------------------------
What a role is
---------------------------------------------------------------------------
A `role(...)` is a **scope** in doc 03 3's sense -- a sixth one, alongside step,
module, branch arm, loop body and pipeline -- and it is declared data like every
other combinator, so it renders, diffs and serialises.

It carries four things that travel together and are useless apart:

  maps=     the vocabulary, for this role only, declared once
  params=   the parameterisation: what about the role changes the capability's
            answer. `core.adverse_events` does not know what a criticality class
            is; the class parameterises its thresholds (05 13-Q3)
  nulls=    role-inapplicable values, which is doc 00 7.4's FOURTH null
            situation (spec 5.17.3) -- distinct from not-collected
  discloses= whose record this is, which bounds what may be said about it
            (05 9.2, spec 9.7)

And it gives values a structural owner. Inside a role scope the value produced
by `core.income` for entity 7 is addressed `surety[7].gross_monthly_income`.
Two applications of one capability to two entities produce two answers that
cannot be confused and need no distinct names -- spec 5.17.1 property 1,
satisfied by the scope rule rather than by a naming convention.

Crossing the boundary requires a declared roll-up. You cannot accidentally leak
one entity's income into the business's namespace, which is the defect class
5.17.3's sole-proprietor case is made of.

---------------------------------------------------------------------------
The measurement
---------------------------------------------------------------------------
`decider reuse report` counts identity relabels attributable to consumed
components (spec 10 acceptance 24; 5.17.7's first indicator, threshold 40).
Roles do not count as relabels -- they are declarations, and there are six of
them below. A rise in the relabel count is a design finding.
"""

from decider2 import role, rollup, NOT_APPLICABLE, LOG_ODDS
from consumed.core_library import core

# --------------------------------------------------------------------------
# ENTITY-LEVEL ROLES. Spec 05 4.3's twelve relationship types collapse to five
# roles, because five is the number that changes what a library name MEANS.
# --------------------------------------------------------------------------

DIRECTOR = role(
    "director",
    over="known.entities",                          # bi-temporal view, named
    where="relationship_type_code in (1, 2, 3, 10)",  # director, member, trustee, partner
    maps={
        core.applicant_age_years: "entity.age_years_at_decision_date",
        core.gross_monthly_income: "entity.salary_from_applicant_cents",
        core.existing_obligations: "entity.personal_obligations_cents",
        core.months_employed: "entity.months_in_office",
    },
    params={
        # Spec 5.17.1 property 2: role is a property of the APPLICATION of the
        # capability, not of the capability or of the value. `core.adverse_events`
        # stays ignorant of criticality; criticality selects its threshold row.
        "criticality_class": "entity.criticality_class",
        "threshold_table": "event_amount_thresholds",
    },
    discloses="entity",
)

SURETY = role(
    "surety",
    over="known.entities",
    where="is_required_surety",
    maps={
        # The sharp one. E-AROD-10 tests the surety's age at the FINAL
        # INSTALMENT DATE, not at decision_date. Declaring the difference here,
        # once, is the alternative to a passthrough step per surety per use site.
        core.applicant_age_years: "entity.age_years_at_final_instalment",
        core.gross_monthly_income: "entity.personal_income_cents",
        # Excluding the facility they stand surety for. It is not their
        # obligation until it is called, and counting it makes every surety
        # unaffordable at the moment they are most needed.
        core.existing_obligations: "entity.obligations_excl_this_suretyship_cents",
    },
    nulls={
        core.months_employed: NOT_APPLICABLE.when("entity.is_retired"),
        core.employment_type_code: NOT_APPLICABLE.when("entity.is_retired"),
    },
    params={"criticality_class": "entity.criticality_class"},
    discloses="entity",
)

CORPORATE_GUARANTOR = role(
    "corporate_guarantor",
    over="known.entities",
    where="not is_natural_person and relationship_type_code == 8",
    maps={
        core.gross_monthly_income: NOT_APPLICABLE.always,
        core.applicant_age_years: NOT_APPLICABLE.always,   # a company has no age in this sense
        core.total_exposure: "guarantor.own_group_exposure_cents",
    },
    params={"scorecard_family": "BUS-COMM"},
    discloses="entity",
)

SOLE_PROPRIETOR = role(
    "sole_proprietor",
    over="known.entities",
    where="relationship_type_code == 11",
    maps={
        # Spec 5.17.3: the business's net profit after drawings IS the personal
        # income. `once=` is the declaration that makes double-counting a build
        # error rather than an affordability answer that is wrong in the
        # client's favour by exactly the drawings.
        core.gross_monthly_income: "business.net_profit_after_drawings_cents",
    },
    once={"business.drawings_cents": ["O14.debt_service", "p02.income"]},
    params={"criticality_class": "entity.criticality_class"},
    discloses="entity+business",     # the one role where the two coincide
)

PERIPHERAL = role(
    "peripheral",
    over="known.entities",
    where="criticality_class == 3",
    maps={},                     # nothing renamed: peripheral entities are scored,
                                 # never blended, and never asked for income
    params={"criticality_class": "entity.criticality_class"},
    discloses="entity",
)

# --------------------------------------------------------------------------
# SUBJECT-LEVEL ROLES. A group is not an entity, and `total_exposure` means
# something different against it. Spec 4.1: a group is a derived set whose
# identity changes when its composition does, so the role carries the
# composition date and every group-level output names it (13-Q11).
# --------------------------------------------------------------------------

GROUP = role(
    "group",
    over="group_members",
    maps={core.total_exposure: "group.aggregate_exposure_cents"},
    params={
        "composition_date": "group.composition_date",
        "contingent_conversion": "group.conversion_convention",  # 100% / 50%, spec 5.3 O10
    },
    identity=("group_members", "composition_date"),   # NOT a stable group_id
    discloses="group",
)

# --------------------------------------------------------------------------
# ROLL-UPS. Leaving a role scope requires naming how. Three consumers, three
# roll-ups, one component (spec 5.17.3 / 00 13-Q6) -- and here all three live in
# one artefact, so "why does this event show as material here and disqualifying
# there" is answered by reading two records, not two code paths.
# --------------------------------------------------------------------------

ENTITY_VERDICT = rollup("entity_adverse_verdict", by="05.AE-R-01..12",
                        over=[DIRECTOR, SURETY, CORPORATE_GUARANTOR, PERIPHERAL],
                        attribution="binding_rule + contributing_event_set")

PEOPLE_BLEND = rollup("people_pd", by=LOG_ODDS, weights="05.PP-03..04",
                      over=[DIRECTOR, SOLE_PROPRIETOR],
                      excludes=[SURETY, CORPORATE_GUARANTOR],   # 05 PP-01
                      partial=True)   # see consumed/GAPS.toml: `partial_reblend`

SIGNAL_SCORE = rollup("watchlist_score", by="weighted_decay",
                      over=[DIRECTOR, SURETY, CORPORATE_GUARANTOR, PERIPHERAL],
                      scope_fanout="entity -> business -> facility")   # spec 5.6.2 rule 3
