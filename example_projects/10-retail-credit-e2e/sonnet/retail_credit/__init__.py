"""retail_credit: project 10, the retail credit flow, end to end (spec 10).

Standalone by design (10 §1.1): the only thing this project takes from
outside itself is `credit_core` (project 00's published capabilities,
project 00 §6) and its canonical vocabulary (project 00 §4). It does not
import project 02, 03, 06 or 07, even though it covers the same ground —
the overlap is intentional (10 §1.1), and this project supplies its own
thresholds, tables, band edges and budgets.

This package is organised the way the spec is organised: one module per
phase (P01..P18), plus the cross-cutting registries a flow of this size
needs and no isolated flow does -- `entry_points` (the phase-set matrix,
§5.20), `phases` (the 18-phase registry and the ownership map, §5.1,
§5.26.1), `shared_intermediates` (§5.21), `ordering` (§5.22) and
`decision_record` (§5.31). See NOTES.md "How I organised a large project"
for why one module per phase, not one module per team or one giant file.
"""
