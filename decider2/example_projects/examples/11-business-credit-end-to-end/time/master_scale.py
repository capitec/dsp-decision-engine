"""The master scale registry and its restatement maps. Spec 5.10.4, table 56/57.

Small file, large consequence: it is the artefact that lets the Bank say a 2026
grade 6 and a 2031 grade 7 are or are not the same statement.

Why this is not just another `dated_table`: a master scale version is not
superseded. All four versions stay simultaneously *readable* forever, because
every decision of record ever written names one of them and none of them is
rewritten. `scale_registry` is therefore a `bound_table` in spirit -- it is
bound by the `master_scale_version` recorded on the decision -- but it is
resolved by a *policy* rule when a NEW grade is being assigned. It is the one
artefact in the project read under both resolution rules, which is why it gets
its own file and an explicit note rather than being buried.
"""

from decider2 import dated_table, bound_table, module
from time.dating import POLICY, CONTRACT

# Which scale a NEW grade is expressed on: policy, by decision_date.
master_scale_current = dated_table(
    "master_scale_registry", resolution=POLICY, owner="credit_risk_modelling",
)

# Which scale an EXISTING grade was expressed on: bound, by the decision record.
master_scale_registry = bound_table(
    "master_scale_registry", resolution=CONTRACT,
    bind_on="master_scale_version", owner="credit_risk_modelling",
    retention="life of any decision of record that names it, plus 7 years",
)

# 6 pairwise maps across 4 scale versions, each 12 x 12 (spec table 57).
# `partial=True` is the important flag: a map may legitimately cover only part
# of its domain, and a cell with no entry means *not comparable*, not zero.
restatement_maps = dated_table(
    "grade_restatement_maps", resolution=POLICY,
    owner="credit_risk_modelling", co_owner="portfolio_management",
    partial=True,
    approval_required=True,
)


def scale_of(risk_grade: int, master_scale_version: str) -> tuple:
    """A grade never travels without its scale. Spec 5.10.4 requirement 1."""
    pass  # pair the grade with its scale entry; fail loudly on an unknown version


def map_exists(from_scale: str, onto_scale: str, grade: int) -> bool:
    """Whether a restatement is possible for THIS grade, not for the scale pair.

    Requirement 5's sharp edge: invertibility is per-cell, not per-map. The
    v3 -> v4 collapse maps 11 of 15 grades cleanly and leaves 4 with no
    pre-image. A design that asks "does a map exist" at map granularity will
    restate four grades wrongly and never notice.
    """
    pass  # restatement_maps[(from_scale, onto_scale)].has(grade)


MasterScale = module(scale_of, map_exists, name="master_scale",
                     contract="contracts/master_scale.json")
