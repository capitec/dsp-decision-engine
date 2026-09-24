"""assessment: the affordability and obligations assessment (spec 02).

Composes `credit_core`'s five affordability units (project 00: `income`,
`deductions`, `expense_norms`, `obligations`, `affordability`) into the
household-level assessment, the four modes and the three answer shapes
DEPS.md's "Cycles" §2 assigns to this project. 00 owns the units'
arithmetic, tables and interfaces; this package never re-derives them, it
composes them.

Modules:
    household        stage 1 (framing) plus the household-level combination
                      of the per-applicant income and deductions units.
    modes             the four `assessment_mode_code` values and their
                      evidence-rule parameters (minimum income tier).
    evidence_gates    the checks that decide `evidence_sufficiency_code`:
                      an applicant whose income can't be established, a
                      stale bureau view, a REFER account type.
    capacity          stage 6: the buffer/residual-floor race and the
                      tighten-only overlay on top.
    verdict           stage 7: the verdict and the three answer shapes.
"""
