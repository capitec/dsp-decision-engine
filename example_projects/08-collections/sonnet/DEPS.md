# What this project consumes, and what it does not

Per the specs' own `example_projects/specs/DEPS.md`: **08: hard 00, 02 | soft 01, 04, 07, 09.**

## Hard: project 00 (`credit_core`)

Consumed read-only via `PYTHONPATH`. Used directly: `credit_core.adjustments`
(`Adjustment`, `AdjustmentEffect`, `AdjustmentRegister` -- both overlay
registers in this project, `scoring.SCORE_ADJUSTMENTS` and
`matrix.MATRIX_ADJUSTMENTS`), `credit_core.calibration.probability_of_default`
(reused unmodified as the roll-probability curve), `credit_core.evidence.cell_id`
(every table's cell attribution).

## Hard: project 02 (affordability assessment)

Consumed read-only via `PYTHONPATH`. `collections_treatment/arrangements.py`
loads 02's own `pipeline.py` (its `evidence_unit()`/`capacity_unit()`, in
ARRANGEMENT mode) by file path under a private module name, specifically
*not* a bare `import pipeline`, because this project's own build entry
point is also named `pipeline.py`, and the two collide on `sys.path`. See
that module's docstring and NOTES.md "Framework friction" for the full
writeup.

## Soft, not built: project 07 (credit limit management)

**Project 07 depends on project 08, not the other way round.** The specs'
own DEPS.md is explicit: "07 | 00, 02 | 03, 08 (treatment-state feed), 09"
-- the arrow points from 07 to 08. 07 §4.1 names "Project 08 treatment
state | ... | 0.31M accounts | Daily" as one of *its* inputs, used only by
its exclusion rule X16. Spec 08 itself mentions project 07 only as a
comparison ("Project 07 has the same shape at a portfolio budget level",
§2; §13 Q5 asks whether 08's capacity allocation and 07's portfolio budget
are "the same shape, or a different one" -- a design question, not a data
dependency).

So there is nothing to stub *from* 07 here. What this project does instead,
to make the reverse direction usable by a future 07 implementer, is publish
a stable, documented output shape that is exactly what 07 §4.1's feed
needs: `account_id`, `treatment_code`, `treatment_intensity`,
`path_position`, `episode_open`, `allocated`, `non_selection_reason_code`,
`decision_date` -- every field a daily treatment-state feed would carry.
No code in this project reads anything from 07.

## Soft, not built: projects 01, 04, 09

01 (fraud) and 04 (campaign trees) are mentioned only in each other's §13
questions, not as data dependencies of 08. 09 (governance/replay) is
adopted as the evidence-contract checklist (spec 09 §5.14-§5.15), not as
code this project calls -- see NOTES.md "What I built" for how each of the
09-C minimum items is satisfied.
