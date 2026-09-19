"""Two version-resolution rules, expressed as two *types*, not two conventions.

Spec 11 H2 / 5.5.1 / 6.3 / 13-Q3. This is the single most consequential
departure from doc 03 in the whole project, so it is the first file to read.

The estate has exactly two kinds of versioned artefact and they resolve
differently:

    POLICY    artefacts resolve by `decision_date`      (doc 00 7.3)
    CONTRACT  artefacts resolve by the version BOUND to an instance when that
              instance was created, and NEVER move -- not by decision_date, not
              by test date, not by "current standard".

Doc 03 has one effective-dating story: `decision_date` governs every versioned
artefact. Applied to a covenant definition it produces a number, without error,
that breaches a covenant the client did not breach (spec 9.5). The failure is
silent, plausible, and legally consequential, so the design answer cannot be a
convention or a code review rule. It has to be that the wrong call does not
*exist*.

So: two constructors, two types, and no accessor connects them.

    dated_table(...)   ->  Dated[T]   .  Resolves only against `decision_date`.
                                         Has no `.bound_to()`.
    bound_table(...)   ->  Bound[T]   .  Resolves only against a binding key
                                         carried on the subject row.
                                         Has NO `.in_force_at(date)` at all --
                                         there is no way to ask a Bound artefact
                                         what today's version says without first
                                         naming an instance. The question that
                                         produces the wrong answer is unaskable.

Build-time checks the two types make possible (all static, no execution):

  R1  A step that reads a `Bound[T]` must have the binding key in scope, from
      the same row. Otherwise: build error naming the step and the key.
  R2  A module tagged `contractual=True` may not read any `Dated[T]`. The
      covenant test module is so tagged; see modules/covenants/test.py.
  R3  A `Bound[T]` version may not be deleted, edited or superseded while any
      live instance binds it -- including every member of its `closure`.
"""

from decider2 import resolution, dated_table, bound_table

# --------------------------------------------------------------------------
# The two rules. Declared once, here, and imported by name everywhere else so
# that `grep -rn "CONTRACT" .` is the complete list of contractual reads.
# --------------------------------------------------------------------------
POLICY = resolution(
    "policy",
    by="decision_date",
    doc="Doc 00 7.3. The version in force on the date whose rules govern.",
)

CONTRACT = resolution(
    "contract",
    by="instance_binding",
    doc=(
        "Spec 11 H2. The version agreed when the instance was created. Immune "
        "to policy change, to the passage of time, and to its own owner."
    ),
    immutable_once_bound=True,
)

# --------------------------------------------------------------------------
# Policy artefacts. 62 of the 63 tables (spec 6.1) are these.
# --------------------------------------------------------------------------
sector_risk = dated_table("sector_risk", resolution=POLICY, owner="sector_analytics")
appetite_grid = dated_table("appetite_grid", resolution=POLICY, owner="credit_committee")
business_rate_card = dated_table("rate_card_business", resolution=POLICY, owner="treasury")
statement_line_map = dated_table("statement_line_mapping", resolution=POLICY, owner="policy")
breach_bands = dated_table("breach_classification", resolution=POLICY, owner="policy+legal")

# --------------------------------------------------------------------------
# The one contractual artefact. Spec 6.3 / table 39.
#
# `closure=` is spec 6.3 requirement 2 made mechanical: a covenant definition
# depends on the statement mapping version, the accounting-basis rules and the
# annualisation conventions that were in force when it was written. Retiring
# any of them retires the definition. Declaring the closure lets the framework
# refuse the retirement rather than discovering it as a 2031 replay failure.
# --------------------------------------------------------------------------
covenant_definitions = bound_table(
    "covenant_definitions",
    resolution=CONTRACT,
    bind_on="covenant_definition_version",     # a column on the covenant instance
    owner="legal",
    levels_owner="business_credit_risk_policy",  # two owners, one artefact (spec 5.5.3)
    closure=[statement_line_map, "accounting_basis_rules", "annualisation_conventions"],
    live_versions=1_900,   # not a typo: ~1 900 simultaneously live (spec 6.3)
)

# Contractual, but not in the definition library: these are per-instance values
# fixed at documentation and owned by nobody thereafter (spec 6.4, class
# "contractual"). They are params with `frozen_at="instance"`, which removes
# them from every params document -- a policy owner literally cannot express a
# change to them, which is the point.
contractual_margin = bound_table("facility_margin", resolution=CONTRACT, bind_on="facility_id")
security_terms = bound_table("security_terms", resolution=CONTRACT, bind_on="facility_id")
