"""The decision record: one schema, eight fillings. Spec 5.31, 5.31.1.
Referenced by every entry point's `record_shape=` (entrypoints/manifest.py).

`record_shape="record_shapes:FullAssessment"` is a STRING today because the
skeleton predates this file; it resolves to the classes below. There is no
`FullAssessmentRecord`, `CampaignRowRecord`, `QuotationRecordRecord` as eight
unrelated schemas — there is one sparse record and eight FILTERED VIEWS over
it, because eight schemas guarantee the fourth one is missing the table
versions when a finding lands (spec 5.31.1's opening line).

The four absences of spec 4.5/5.31.1 are four DIFFERENT recorded facts and
this file is where they stop being collapsible into one another: a phase
that did not run on this entry point (`record_completeness_code` 2) cannot be
represented the same way as a phase abandoned on budget (3), a phase that ran
degraded (4), or a value that could not be established from evidence (a
credit fact, not a record fact, carried on the value itself).
"""

from __future__ import annotations

from decider2.record import RecordShape, sparse_field

DecisionRecordBase = RecordShape(
    fields=[
        "assessment_id", "entry_point_code", "phase_set_id", "decision_date",
        "blast_radius_id", "build_id",
        "inputs_as_received",                       # sparse_field: pre-normalisation
        "table_and_parameter_versions",              # cell-level, doc 07 §6.1
        "adjustment_set_id", "overlay_effects",
        "evaluated_decision_points",                 # not just fired -- spec 5.31.2(7)
        "cap_chains",                                # all five, full chains
        "characteristic_contributions",
        "solve_evaluations",                         # every candidate the solve tried
        "loop_passes",                                # every pass, trigger, discarded offers
        "value_basis_code", "scenario_ref",           # per multi-basis value
        "final_validation_assertions",
        "degraded_mode_code", "source_degradation_codes",
        "reason_set", "primary_reason_code",
        "phase_timings", "phase_budget_overrun_codes",
        "record_completeness_code",
    ],
    idempotent=True,
    emission="at_least_once",
)

FullAssessment = DecisionRecordBase.view(
    "FullAssessment", entry_points=(1,), mean_bytes=58_000,
    note="71% bureau payload -- content-addressed de-duplication saves 710 GB/yr.",
)

LimitChangeRecord = DecisionRecordBase.view("LimitChangeRecord", entry_points=(2,))

LimitProgrammeRow = DecisionRecordBase.view(
    "LimitProgrammeRow", entry_points=(3,), mean_bytes=1_700,
    note="4.1 M rows/cycle, including excluded accounts -- 'we did consider "
         "your account' is itself an answer.",
)

CampaignRow = DecisionRecordBase.view(
    "CampaignRow", entry_points=(4,), mean_bytes=2_300,
    extra_fields=["campaign_assignment", "tree_path_taken", "holdout_flag"],
)

ConsolidationRecord = DecisionRecordBase.view(
    "ConsolidationRecord", entry_points=(5,), mean_bytes=296_000,
    note="250 scenarios x ~58 values. Full detail for the top 12 rejected "
         "scenarios, summary for the rest -- 'we did not keep it' is not "
         "available for a rejected scenario (spec 5.31.3).",
)

RepriceRecord = DecisionRecordBase.view("RepriceRecord", entry_points=(6,))

QuotationRecord = DecisionRecordBase.view(
    "QuotationRecord", entry_points=(7,), mean_bytes=1_100,
    excludes=["evaluated_decision_points_for_absent_phases", "cap_chains_beyond_regulatory",
             "loop_passes"],
    note="emits_decision_record=False, emits_quotation_record=True -- structurally "
         "a different contract, not a filtered FullAssessment.",
)

WhatIfComparison = DecisionRecordBase.view(
    "WhatIfComparison", entry_points=(8,), mean_bytes=164_000,
    non_production=True, writable_to_decision_store=False, issuable_to_client=False,
)

RECORD_SHAPES = (FullAssessment, LimitChangeRecord, LimitProgrammeRow, CampaignRow,
                 ConsolidationRecord, RepriceRecord, QuotationRecord, WhatIfComparison)

# Spec 5.31.1's table, as an assertion checked against every emitted record:
COMPLETENESS_CODES = {
    1: "complete",
    2: "phase_absent_by_entry_point",     # structural, normal
    3: "phase_abandoned_on_budget",       # operational, monitored
    4: "phase_ran_degraded",              # degraded, re-assessable
    # a fourth code space -- "value could not be established" -- lives on the
    # VALUE (spec 4.5), never collapsed into this field.
}
