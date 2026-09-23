# Old test coverage map (for T7.1)

Before `decider_old/`, `decider2/` and `tests/_legacy/` were deleted (T7.1):
does every old test's behaviour have a test under `tests/`? Old tests are every
test function in `decider2/tests/*.py` and `tests/_legacy/**` (decider_old's
tests; `decider_old/` itself had none). Matched by behaviour, not name. The
paths on the left no longer exist; the history is in git.

**Summary: 639 old tests: 543 PORTED, 96 DROPPED, 0 PENDING, 0 GAP** (the 24 GAPs were closed by G1: 14 ported, 10 dropped as triaged in `notes/progress.md`; the 35 decision-table rows pending T4.3 were resolved in T7.1: 32 covered by T4.3's `tests/tables/`, 2 wide-table tests ported by T7.1, 1 dropped).

| Source | Tests | Ported | Dropped | Gap |
|---|---|---|---|---|
| decider2 boundary + shim (7 files) | 104 | 87 | 17 | 0 |
| decider2 compile/runtime/serving/cli/arity (9 files) | 99 | 70 | 29 | 0 |
| decider2 graph/params/registry/testing (8 files) | 110 | 89 | 21 | 0 |
| decider2 control flow/flagship/spec/review (5 files) | 73 | 64 | 9 | 0 |
| decider2 trees/expr/tables (11 files) | 155 | 148 | 7 | 0 |
| tests/_legacy (decider_old, 10 files) | 98 | 85 | 13 | 0 |

Classes: PORTED = a new test covers the same behaviour (possibly adapted to a
design change, noted in the line); DROPPED = behaviour removed, reason quoted
from notes/progress.md, notes/spec-amendments.md, notes/*.md or a task commit;
GAP = no new test and no recorded drop.

## GAPs (closed by G1; kept for the record)

| Old test | What it checks | Still applies? | Suggested new test |
|---|---|---|---|
| `decider2/tests/test_review_findings.py::test_assert_equivalent_fails_when_all_three_modes_crash_identically` | `assert_equivalent` raises, not passes vacuously, when every mode raises | Yes. `decider/testing` re-raises today but nothing pins it | `tests/testing/test_testing.py` |
| `decider2/tests/test_params_namespacing.py::test_a_bound_value_changes_the_answer` | a `bind()` value is what `run()` computes with (was a real decider2 bug) | Yes. Only IR and direct call are tested, never `run()`/`score()` in compiled modes | `tests/steps/test_function_step.py` (all modes) |
| `decider2/tests/test_params_namespacing.py::test_a_bound_value_is_still_validated_by_the_model` | `bind(cap=999)` on `le=60` is rejected | Yes. Models use `validate_default=True`, untested | `tests/steps/test_function_step.py` |
| `decider2/tests/test_runtime_serve.py::test_params_only_activation_does_not_recompile`, `test_score_plan.py::test_serve_handle_generation_swap_takes_effect_and_compiles_nothing` | staging/activating a params-only version compiles nothing | Yes. `RequestHandler.stage` rebinds per version; the answer swap is tested, no-compile is not | `tests/serving/test_serving.py` (stage v2 inside `decider.testing.no_recompile`) |
| `decider2/tests/test_trees_strings.py::test_precompile_leaves_nothing_for_the_first_string_request_to_compile` | after warm-up the first `run()`/`score()` on a string tree compiles nothing | Yes. Single-record latency is the priority; `stage()` warms with made-up values | `tests/serving/test_serving.py` |
| `decider2/tests/test_trees_strings.py::test_editing_adding_and_removing_patterns_never_recompiles`, `test_a_pattern_count_change_and_a_threshold_change_share_one_kernel_signature` | editing/adding/removing string patterns changes data only, no recompile | Yes. T4.2 claims it; tested for numeric thresholds only | `tests/trees/test_strings.py` |
| `decider2/tests/test_typed_features.py::test_a_typed_tree_path_fn_is_a_genuine_cache_hit_in_a_fresh_process` | the tree walker loads from numba's disk cache in a fresh process | Yes. `walker.walk` is `cache=True` with intrinsics/overloads; only a plain row kernel is tested | `tests/trees/test_tree_config.py` (subprocess pattern from `tests/testing/test_shared_bundle_cache.py`) |
| `decider2/tests/test_no_arity_ceiling.py::test_a_400_feature_tree_builds_and_answers_correctly` | a tree reading 400 distinct columns answers correctly in every mode | Yes. Fixtures use ~7 features; a slot bug in `encode.py` would go unseen | `tests/trees/test_trees.py` |
| `decider2/tests/test_no_arity_ceiling.py::test_more_than_six_computed_features_in_one_tree` | ~10 computed features, each gating its own node | Yes. Every computed-feature fixture has one feature | `tests/trees/test_computed_features.py` |
| `decider2/tests/test_trees.py::test_a_tree_reports_which_leaf_it_reached`, `test_the_path_column_survives_an_explicit_emit` | a tree writes `<name>_path` (reached leaf) usable/emittable like any value | Decide. Visits only reach debug sessions, so `run()` output can't say which leaf answered. Port or record a drop | `tests/trees/test_tree_config.py` |
| `tests/_legacy/rules/test_tree_end_to_end.py::test_prioritize_results_empty_list_returns_default` | a prioritised rule set with no rules answers the default | Yes. `PrioritizedFlatRuleDocument.rules` accepts `[]`, result unpinned; test it or reject it like empty composites | `tests/trees/test_legacy_end_to_end.py` |
| `tests/_legacy/rules/test_tree_migration.py::test_can_create_default_tree` | a new tree starts as a valid one-leaf v3 document | Decide. Only matters for an editor's starter tree; drop recorded only in decider2's own port file | `tests/trees/test_tree_documents.py`, or record in `notes/tree-documents.md` |
| `decider2/tests/test_graph_module.py::test_contract_true_snapshots_to_the_derived_path`, `test_contract_catches_a_breaking_change` | `contract=` snapshots a step's interface and fails on drift | Decide. Not built in decider; likely a drop to record | record in `notes/spec-amendments.md` |
| `decider2/tests/test_params_scratch.py::test_parse_docstring_with_implements_line`, `test_parse_docstring_without_implements_line`, `test_parse_docstring_none`; `decider2/tests/test_flagship.py::test_the_docstring_is_the_description` | step description from docstring, `Implements: <policy ref>` recorded | Decide. Not built in decider (no `Step.doc`/implements) | `tests/steps/test_function_step.py`, or record the drop |
| `decider2/tests/test_review_findings.py::test_6a_a_floating_def_in_a_pipeline_file_is_flagged` | lint flags a def in a pipeline file no pipeline uses | Decide. No lint module in decider | record the drop |
| `decider2/tests/test_review_findings.py::test_6b_an_unread_params_model_field_is_a_build_error` | a hand-written params model field the body never reads is a build error | Probably not (no `module(params=Model)`); record the drop | `notes/progress.md` |
| `decider2/tests/test_review_findings.py::test_6c_a_step_parameter_never_referenced_in_the_body_is_a_build_error` | a declared step argument the body never reads is a build error | Decide. Meaningful for function steps, not implemented | `tests/steps/test_function_step.py`, or record the drop |

Rows marked Yes (12 tests) are real missing tests for behaviour decider has.
Rows marked Decide/Probably not (12 tests) are decider2/decider_old features
decider never built (or a choice not yet made) with no recorded drop: each needs
a one-line drop in the notes or a port.

### Drops with no recorded reason

Counted as DROPPED because the behaviour no longer exists; the Old-test coverage
triage in `notes/progress.md` now records each:

- `decider2/tests/test_score_plan.py::test_plan_holds_exactly_the_schema_invariant_pieces` (decider2 `ScorePlan` internals)
- `decider2/tests/test_score_plan.py::test_a_re_entrant_call_on_one_thread_never_shares_the_busy_pool` (no row pool now)
- `decider2/tests/test_score_plan.py::test_a_hostile_in_place_field_swap_is_detected_not_served_stale` (steps are frozen; the IR cache is keyed by `id(step)`, so an `object.__setattr__` swap would serve a stale IR)
- `decider2/tests/test_testing.py::test_assert_equivalent_rejects_a_mode_kwarg` (superseded by `modes=`)
- `tests/_legacy/rules/test_parameters.py::test_computed_feature_uses_parameter` (`p.bonus` in a computed feature; the refusal is pinned by `tests/trees/test_tree_conditions.py::test_computed_feature_p_dot_attribute_syntax_is_refused`)

Weak ports worth tightening (not gaps): the run()-level "a cast clones, the caller's
frame is untouched" half of `test_boundary_integration.py::test_the_callers_frame_is_rechunked_in_place_by_apply`;
batch `run()` under 16 threads; `test_flagship.py::test_the_bare_default_error_names_the_parameter`
(the new error does not assert the argument name); the categorical-cast and
numeric-declared-as-string tree tests are covered at boundary-plan level only.

## References to decider2 / decider_old outside the old packages

- **Imports:** none in `decider/`, `tests/` (excluding `_legacy`) or `pyproject.toml`.
  `pyproject.toml` has `testpaths = ["tests"]`, `--ignore=tests/_legacy`, `packages = ["decider"]`.
- **Benchmarks (import decider2 via `PYTHONPATH=<decider2 src with _nashim built>`):**
  `benchmarks/control_flow_vs_decider2.py`, `benchmarks/modes_vs_decider2.py`
  (imports `decider2.examples.flagship`), `benchmarks/trees_vs_decider2.py`. They stop
  working once `decider2/src` is gone: T7.1 must delete them or strip the decider2 half.
- **Text only:** `decider/steps/trees/__init__.py` docstring ("decider_old flat
  rules"); test names/docstrings in `tests/trees/test_tree_migration.py`,
  `test_tree_documents.py`, `test_legacy_conditions.py`, `test_legacy_parameters.py`,
  `test_ported_extra.py` (inline fixtures, no file reads from the old trees);
  `CLAUDE.md` lines 43-51 describe the old packages.
- **Out of this map's scope:** `decider2/evaluation/**` and
  `decider2/example_projects/**` contain their own tests that import decider2;
  deleting only `decider2/src` leaves them broken, so T7.1 should delete all of `decider2/`.
- `tests/_legacy/` itself is deleted with decider_old (its 5 decision-table tests pending T4.3).

## Full mapping

One line per old test: `old test | class | new test or reason`.

### `decider2/tests/test_boundary_dtypes.py` (16)

- test_kind_for_mirrors_numpy_dtype | PORTED | tests/boundary/test_boundary_dtypes.py::test_an_annotation_lands_in_its_feature_kind_and_optional_int_is_f64 + tests/compile/test_compile_kernel.py::test_numpy_dtype_mirrors_feature_kind (deferred numpy_dtype half landed in T3.2)
- test_every_row_of_the_table_that_nanoarrow_reads_natively_gets_no_cast | PORTED | tests/boundary/test_boundary_dtypes.py::test_a_dtype_nanoarrow_reads_natively_gets_no_cast
- test_a_castable_mismatch_gets_exactly_one_reported_frame_tier_cast | PORTED | tests/boundary/test_boundary_dtypes.py::test_a_castable_mismatch_gets_exactly_one_frame_tier_cast
- test_a_column_with_no_flat_form_is_reported_for_the_kernel_split_by_name | PORTED | tests/boundary/test_boundary_dtypes.py::test_a_column_with_no_flat_form_needs_a_kernel_split_by_name
- test_needs_kernel_split_is_an_arrow_kind_error | PORTED | tests/boundary/test_boundary_dtypes.py::test_needs_kernel_split_is_an_arrow_kind_error
- test_a_pair_with_no_cast_is_left_for_the_import_to_refuse_by_arrow_type | PORTED | tests/boundary/test_boundary_dtypes.py::test_a_pair_with_no_cast_is_left_for_the_import_to_refuse
- test_decimal_casts_to_money_scaled_int64_cents | PORTED | tests/boundary/test_boundary_dtypes.py::test_decimal_casts_to_money_scaled_int64_cents
- test_decimal_rescales_from_a_different_source_scale_to_money_scale_2 | PORTED | tests/boundary/test_boundary_dtypes.py::test_decimal_rescales_from_any_source_scale_to_cents
- test_decimal_declared_float_arrives_as_cents_too | PORTED | tests/boundary/test_boundary_dtypes.py::test_decimal_declared_float_arrives_as_cents_too
- test_a_decimal_that_overflows_int64_is_the_kernel_split_case_not_a_crash | PORTED | tests/boundary/test_boundary_dtypes.py::test_a_decimal_that_overflows_int64_needs_a_kernel_split_not_a_crash
- test_a_failing_cast_is_caught_as_baseexception_not_just_exception | PORTED | tests/boundary/test_boundary_dtypes.py::test_a_rust_panic_during_a_cast_is_caught_even_though_it_is_not_an_exception
- test_temporal_columns_declared_float_cast_through_their_storage_integer | PORTED | tests/boundary/test_boundary_dtypes.py::test_temporal_columns_declared_float_cast_through_their_storage_integer
- test_cast_frame_returns_a_new_frame_and_never_retypes_the_callers_columns | PORTED | tests/boundary/test_boundary_dtypes.py::test_cast_frame_returns_a_new_frame_and_never_retypes_the_callers_columns
- test_explain_boundary_reports_arrow_type_kind_and_cast_per_declared_input | PORTED | tests/boundary/test_boundary_dtypes.py::test_explain_boundary_reports_arrow_type_kind_and_cast_per_declared_input
- test_explain_boundary_without_inputs_reads_every_column_as_its_natural_kind | PORTED | tests/boundary/test_boundary_dtypes.py::test_explain_boundary_without_inputs_reads_every_column_as_its_natural_kind
- test_explain_boundary_reports_a_refused_column_instead_of_raising | PORTED | tests/boundary/test_boundary_dtypes.py::test_explain_boundary_reports_a_refused_column_instead_of_raising

### `decider2/tests/test_boundary_extract.py` (28)

- test_every_dtype_of_the_table_round_trips_exactly_as_python_sees_it | PORTED | tests/boundary/test_boundary_extract.py::test_every_dtype_of_the_table_round_trips_exactly_as_python_sees_it
- test_an_int64_above_2_53_survives_exactly_in_int64_and_uint64 | PORTED | tests/boundary/test_boundary_extract.py::test_an_int64_above_2_53_survives_exactly_in_int64_and_uint64
- test_floats_stay_floats_ints_stay_ints_bools_stay_bools | PORTED | tests/boundary/test_boundary_extract.py::test_floats_stay_floats_ints_stay_ints_bools_stay_bools
- test_a_string_column_declared_bytes_arrives_as_spans_into_polars_memory | PORTED | tests/boundary/test_boundary_extract.py::test_a_string_column_declared_bytes_arrives_as_spans_into_polars_memory
- test_a_str_input_arrives_as_dictionary_codes_with_the_exported_dictionary | PORTED | tests/boundary/test_boundary_extract.py::test_a_str_input_arrives_as_dictionary_codes_with_the_exported_dictionary
- test_categorical_and_enum_inputs_read_their_dictionary_indices_and_dictionaries | PORTED | tests/boundary/test_boundary_extract.py::test_categorical_and_enum_inputs_read_their_dictionary_indices_and_dictionaries
- test_missing_as_fills_in_the_gather_and_reports_what_it_filled | PORTED | tests/boundary/test_boundary_extract.py::test_missing_as_fills_nulls_in_the_gather_and_needs_no_mask (fill half; reporting half dropped: T3.1 "routing, NOT_APPLICABLE_AS and fill reporting removed")
- test_not_applicable_as_fills_identically_but_tags_a_different_reason | DROPPED | T3.1: "routing, NOT_APPLICABLE_AS and fill reporting removed"
- test_a_clean_missing_as_column_reports_nothing_filled | PORTED | tests/boundary/test_boundary_extract.py::test_a_clean_missing_as_column_passes_through_unchanged
- test_optional_keeps_a_validity_mask_and_never_fills | PORTED | tests/boundary/test_boundary_extract.py::test_optional_keeps_a_validity_mask_and_never_fills
- test_optional_clean_column_still_gets_an_all_true_mask | PORTED | tests/boundary/test_boundary_extract.py::test_optional_clean_column_still_gets_an_all_true_mask
- test_the_garbage_slot_under_a_null_is_never_read | PORTED | tests/boundary/test_boundary_extract.py::test_the_garbage_slot_under_a_null_is_never_read
- test_a_left_join_null_is_filled_too | PORTED | tests/boundary/test_boundary_extract.py::test_a_left_join_null_is_filled_too
- test_an_all_null_null_dtype_column_fills_or_masks_like_any_other | PORTED | tests/boundary/test_boundary_extract.py::test_an_all_null_null_dtype_column_fills_or_masks_like_any_other
- test_fresh_sliced_and_multi_chunk_frames_extract_identically | PORTED | tests/boundary/test_boundary_extract.py::test_fresh_sliced_and_multi_chunk_frames_extract_identically
- test_the_null_on_row_zero_and_on_the_last_row_survives_every_slice | PORTED | tests/boundary/test_boundary_extract.py::test_the_null_on_row_zero_and_on_the_last_row_survives_every_slice
- test_a_zero_row_frame_extracts_zero_length_columns_of_the_right_dtype | PORTED | tests/boundary/test_boundary_extract.py::test_a_zero_row_frame_extracts_zero_length_columns_of_the_right_dtype
- test_extract_frame_removes_routed_rows_from_the_kernel_frame | DROPPED | T3.1: "REQUIRED nulls raise MissingInputError...; routing ... removed"
- test_extract_frame_raise_for_fails_before_any_export | PORTED | tests/boundary/test_boundary_extract.py::test_a_null_in_a_required_input_fails_before_any_export_naming_input_path_and_count
- test_a_required_column_entirely_absent_routes_every_row_and_synthesizes_an_empty_column | PORTED | tests/boundary/test_boundary_extract.py::test_a_required_column_entirely_absent_fails_by_name (adapted: raise replaces routing, T3.1)
- test_absent_columns_are_synthesized_per_tier_in_the_kinds_own_dtype | PORTED | tests/boundary/test_boundary_extract.py::test_absent_columns_are_filled_or_masked_in_the_kinds_own_dtype
- test_a_str_column_with_a_text_fill_is_refused_by_name | PORTED | tests/boundary/test_boundary_extract.py::test_a_str_column_with_a_text_fill_is_refused_by_name
- test_a_list_or_struct_column_is_reported_for_the_kernel_split_before_any_export | PORTED | tests/boundary/test_boundary_extract.py::test_a_list_or_struct_column_is_reported_for_the_kernel_split_before_any_export
- test_a_pair_the_table_has_no_cast_for_is_refused_by_column_arrow_type_and_kind | PORTED | tests/boundary/test_boundary_extract.py::test_a_pair_the_table_has_no_cast_for_is_refused_by_column_arrow_type_and_kind
- test_a_decimal_column_crosses_as_scaled_int64_cents_as_before | PORTED | tests/boundary/test_boundary_extract.py::test_a_decimal_column_crosses_as_scaled_int64_cents_as_before
- test_a_decimal_that_overflows_int64_fails_by_name_not_by_crash | PORTED | tests/boundary/test_boundary_extract.py::test_a_decimal_that_overflows_int64_fails_by_name_not_by_crash
- test_the_schema_plan_is_decided_once_per_inputs_and_frame_schema | PORTED | tests/boundary/test_boundary_extract.py::test_the_schema_plan_is_decided_once_per_inputs_and_frame_schema
- test_frame_views_are_pooled_per_thread_not_per_process | PORTED | tests/boundary/test_boundary_extract.py::test_frame_views_are_pooled_per_thread_not_per_process

### `decider2/tests/test_boundary_integration.py` (8)

- test_extracted_readonly_arrays_feed_a_real_njit_kernel_and_write_back_matches_polars | PORTED | tests/boundary/test_boundary_integration.py::test_extracted_readonly_arrays_feed_a_real_njit_kernel_matching_polars (write-back half dropped with writeback.py, C1)
- test_a_null_dense_batch_still_produces_a_full_result_via_refer_routing | DROPPED | T3.1: "refer/decline routing, NOT_APPLICABLE_AS and fill-reason reporting are dropped"
- test_the_frame_shape_corpus_through_apply_is_byte_identical | PORTED | tests/testing/test_testing.py::test_assert_equivalent_passes_over_every_corpus_frame (boundary/single/chunked/empty in every mode) + tests/boundary/test_boundary_extract.py::test_fresh_sliced_and_multi_chunk_frames_extract_identically (slices at extract level only)
- test_the_callers_frame_is_rechunked_in_place_by_apply | PORTED | partial: tests/boundary/test_shim.py::test_a_multi_chunk_frame_imports_as_one_struct_and_polars_rechunks_the_caller pins the in-place rechunk at the shim; the run()-level "cast clones, caller untouched" half has no test (minor, documented in extract.py docstring)
- test_every_rung_of_the_equivalence_ladder_agrees_on_the_boundary_corpus | PORTED | tests/testing/test_testing.py::test_assert_equivalent_passes_over_every_corpus_frame (T3.4 assert_equivalent incl. score per row)
- test_sixteen_threads_on_one_pipeline_agree_with_a_single_threaded_oracle | PORTED | tests/boundary/test_boundary_integration.py::test_sixteen_threads_extracting_at_once_agree_with_a_single_threaded_oracle + tests/run/test_score.py::test_sixteen_threads_score_concurrently_with_no_cross_talk (batch run() under threads not exercised)
- test_a_column_the_kernel_cannot_read_fails_by_name_through_apply | PORTED | tests/boundary/test_boundary_integration.py::test_a_column_the_kernel_cannot_read_fails_by_name (via extract_frame, not run())
- test_a_nullable_int64_at_100k_rows_stays_int64_with_its_fill_and_no_float_round_trip | PORTED | tests/boundary/test_boundary_integration.py::test_a_nullable_int64_at_100k_rows_stays_int64_with_its_fill_and_no_float_round_trip

### `decider2/tests/test_boundary_nulls.py` (11)

- test_only_the_two_fill_tiers_have_a_reason_and_they_differ | DROPPED | T3.1: "routing, NOT_APPLICABLE_AS and fill reporting removed"
- test_default_policy_refers_rather_than_raising | DROPPED | T3.1: "refer/decline routing ... dropped"; spec: missing is an error by default
- test_raise_for_hard_fails_the_whole_batch_by_name | PORTED | tests/boundary/test_boundary_nulls.py::test_a_null_in_a_required_input_names_the_input_the_step_path_and_the_row_count
- test_raise_for_message_matches_doc_05_2s_shape | PORTED | tests/boundary/test_boundary_nulls.py::test_a_null_in_a_required_input_names_the_input_the_step_path_and_the_row_count + ::test_a_missing_input_error_is_a_value_error
- test_a_non_raise_for_required_null_never_reaches_the_kernel_frame_untreated | DROPPED | T3.1: "routing ... removed" (a required null now raises; covered by tests/run/test_inputs_and_score.py::test_a_required_null_raises_naming_the_input_the_step_and_the_null_count)
- test_first_match_wins_when_two_required_columns_are_both_null_on_one_row | PORTED | tests/boundary/test_boundary_nulls.py::test_the_first_offending_required_input_in_declaration_order_is_reported
- test_a_required_column_absent_from_the_frame_routes_every_row | PORTED | tests/boundary/test_boundary_nulls.py::test_a_required_column_absent_from_the_frame_raises_by_name (adapted: raise replaces routing, T3.1)
- test_a_required_raise_for_column_absent_from_the_frame_raises_by_name | PORTED | tests/boundary/test_boundary_nulls.py::test_a_required_column_absent_from_the_frame_raises_by_name + tests/run/test_inputs_and_score.py::test_run_raises_naming_a_required_column_absent_from_the_frame
- test_a_clean_required_column_routes_nothing | PORTED | tests/boundary/test_boundary_nulls.py::test_a_clean_required_column_passes
- test_only_required_inputs_route_and_a_null_dtype_column_counts_as_all_null | PORTED | tests/boundary/test_boundary_nulls.py::test_only_required_inputs_are_checked_and_a_null_dtype_column_counts_as_all_null
- test_routing_a_zero_row_frame_is_empty_not_an_error | PORTED | tests/boundary/test_boundary_nulls.py::test_a_zero_row_frame_is_not_an_error

### `decider2/tests/test_boundary_plans.py` (7)

- test_missing_as_and_not_applicable_as_fill_identically_and_differ_only_in_reason | PORTED | tests/boundary/test_boundary_plans.py::test_missing_as_fills_the_null_with_the_declared_value (NOT_APPLICABLE_AS half dropped, T3.1)
- test_optional_masks_and_never_fills | PORTED | tests/boundary/test_boundary_plans.py::test_optional_masks_and_never_fills
- test_required_routes_the_row_before_the_kernel_sees_it | PORTED | tests/boundary/test_boundary_plans.py::test_required_fails_on_a_null_for_every_kind (adapted: raise replaces routing, T3.1)
- test_a_bytes_input_has_no_fill_and_reports_null_spans | PORTED | tests/boundary/test_boundary_plans.py::test_a_bytes_input_has_no_fill_and_reports_null_spans
- test_the_two_fill_reasons_stay_distinct_when_the_column_is_absent | DROPPED | T3.1: "NOT_APPLICABLE_AS and fill-reason reporting are dropped"
- test_an_absent_column_and_an_all_null_column_produce_the_same_column_per_tier | PORTED | tests/boundary/test_boundary_plans.py::test_an_absent_column_and_an_all_null_column_produce_the_same_column_per_policy
- test_extract_frame_keeps_all_four_tiers_correct_at_once | PORTED | tests/boundary/test_boundary_plans.py::test_every_policy_is_correct_at_once_in_one_frame

### `decider2/tests/test_boundary_writeback.py` (14)

- test_dtype_group_rejects_a_1d_array | DROPPED | C1 cleanup removed `writeback.py` (audit: boundary API only tests use)
- test_dtype_group_rejects_a_name_count_mismatch_column_major | DROPPED | C1 cleanup removed `writeback.py`
- test_dtype_group_rejects_a_name_count_mismatch_row_major | DROPPED | C1 cleanup removed `writeback.py`
- test_column_major_to_series_is_zero_copy_per_column | DROPPED | C1 cleanup removed `writeback.py`
- test_row_major_to_series_is_correct_even_though_strided | DROPPED | C1 cleanup removed `writeback.py`
- test_write_back_appends_computed_columns_to_every_input_column | PORTED | behaviour: tests/run/test_score.py::test_score_returns_the_record_plus_every_output; tests/run/test_flagship.py (run() output = inputs + outputs); writeback.py itself removed in C1
- test_write_back_with_keep_drops_a_declared_column | PORTED | behaviour: tests/wiring/test_emit_drop.py::test_drop_of_an_input_column_removes_it
- test_write_back_with_no_computed_outputs_returns_the_kept_frame_unchanged | DROPPED | C1 cleanup removed `writeback.py`
- test_write_back_combines_all_three_dtype_groups | DROPPED | C1 cleanup removed `writeback.py` (dtype groups no longer exist; mixed-dtype outputs covered by tests/testing/test_testing.py pipeline)
- test_resolve_kept_input_columns_excludes_dropped_and_overwritten | PORTED | behaviour: tests/wiring/test_emit_drop.py::test_drop_of_an_input_column_removes_it + tests/wiring/test_versions.py (waterfall overwrite); helper removed in C1
- test_resolve_kept_input_columns_keeps_everything_by_default | DROPPED | C1 cleanup removed `writeback.py`
- test_row_to_dict_merges_all_groups_for_one_record | PORTED | behaviour: tests/run/test_score.py::test_score_returns_the_record_plus_every_output
- test_row_to_dict_rejects_more_than_one_row | DROPPED | C1 cleanup removed `writeback.py`
- test_row_to_dict_rejects_a_column_major_group | DROPPED | C1 cleanup removed `writeback.py`

### `decider2/tests/test_shim.py` (20)

- test_strings_round_trip_including_the_inline_boundary_empty_null_and_multibyte | PORTED | tests/boundary/test_shim.py::test_strings_round_trip_including_the_inline_boundary_empty_null_and_multibyte
- test_get_string_through_the_accessor_pointer_matches_lengths_and_nulls | PORTED | tests/boundary/test_shim.py::test_get_string_through_the_accessor_pointer_matches_lengths_and_nulls
- test_a_sliced_frame_carries_its_offset_on_each_child_and_reads_correctly | PORTED | tests/boundary/test_shim.py::test_a_sliced_frame_carries_its_offset_on_each_child_and_reads_correctly
- test_a_multi_chunk_frame_imports_as_one_struct_and_polars_rechunks_the_caller | PORTED | tests/boundary/test_shim.py::test_a_multi_chunk_frame_imports_as_one_struct_and_polars_rechunks_the_caller
- test_a_zero_row_frame_binds_with_n_zero | PORTED | tests/boundary/test_shim.py::test_a_zero_row_frame_binds_with_n_zero
- test_a_million_long_strings_span_more_than_one_variadic_buffer | PORTED | tests/boundary/test_shim.py::test_a_million_long_strings_span_more_than_one_variadic_buffer
- test_every_dtype_of_the_ladder_lands_in_its_kind_without_a_cast | PORTED | tests/boundary/test_shim.py::test_every_dtype_of_the_ladder_lands_in_its_kind_without_a_cast
- test_nulls_per_kind_take_the_default_or_the_declared_fill | PORTED | tests/boundary/test_shim.py::test_nulls_per_kind_take_the_default_or_the_declared_fill
- test_categorical_and_enum_expose_a_dictionary_view_of_the_right_length_and_index_type | PORTED | tests/boundary/test_shim.py::test_categorical_and_enum_expose_a_dictionary_view_of_the_right_length_and_index_type
- test_a_decimal_column_is_refused_with_the_frame_tier_cast_hint | PORTED | tests/boundary/test_shim.py::test_a_decimal_column_is_refused_with_the_frame_tier_cast_hint
- test_nested_columns_are_reported_not_crashed | PORTED | tests/boundary/test_shim.py::test_nested_columns_are_reported_not_crashed
- test_a_kind_mismatch_names_the_column_the_arrow_type_and_the_kind | PORTED | tests/boundary/test_shim.py::test_a_kind_mismatch_names_the_column_the_arrow_type_and_the_kind
- test_frame_view_misuse_is_an_error_not_a_wrong_answer | PORTED | tests/boundary/test_shim.py::test_frame_view_misuse_is_an_error_not_a_wrong_answer
- test_numba_loads_from_the_resolved_addresses_agree_with_polars | PORTED | tests/boundary/test_shim.py::test_numba_loads_from_the_resolved_addresses_agree_with_polars
- test_a_kernel_calling_the_shim_through_a_pointer_argument_is_a_genuine_cache_hit | PORTED | tests/boundary/test_shim.py::test_a_kernel_calling_the_shim_through_a_pointer_argument_is_a_genuine_cache_hit
- test_diagnose_reports_the_loaded_shim_and_the_vendor_pin | PORTED | tests/boundary/test_shim.py::test_diagnose_reports_the_loaded_shim_and_the_vendor_pin
- test_the_extension_exports_exactly_one_symbol | PORTED | tests/boundary/test_shim.py::test_the_shim_exports_only_its_own_functions_and_namespaced_nanoarrow (adapted: ctypes .so, not a CPython extension, T3.1)
- test_the_vendored_nanoarrow_matches_its_manifest_and_the_header_matches_the_pin | PORTED | tests/boundary/test_shim.py::test_the_vendored_nanoarrow_matches_its_manifest_and_the_header_matches_the_pin
- test_a_missing_extension_is_one_clear_import_error_naming_platform_interpreter_and_cause | PORTED | tests/boundary/test_shim.py::test_no_c_compiler_is_one_clear_import_error_naming_platform_interpreter_and_cause (adapted: build-on-first-use, T3.1)
- test_a_truncated_extension_file_reports_the_loader_error_with_platform_and_interpreter | PORTED | tests/boundary/test_shim.py::test_a_corrupt_cached_shim_reports_the_loader_error_rather_than_rebuilding

### `decider2/tests/test_compile_driver.py` (9)

- test_one_kernel_per_module_by_default | PORTED | tests/compile/test_compile_units.py::test_each_innermost_sequence_is_its_own_kernel (+ test_without_fusion_every_call_is_its_own_unit)
- test_fuse_merges_a_contiguous_group_into_one_segment | PORTED | tests/compile/test_compile_units.py::test_consecutive_scalar_calls_in_one_sequence_share_a_kernel
- test_intermediate_crossing_a_kernel_boundary_is_a_required_output | PORTED | tests/compile/test_compile_units.py::test_a_value_read_by_another_kernel_is_kept
- test_an_intermediate_local_to_a_fused_segment_is_not_a_required_output | PORTED | tests/compile/test_compile_units.py::test_a_value_only_read_inside_its_kernel_is_not_kept
- test_a_terminal_value_is_always_a_required_output | PORTED | tests/compile/test_compile_units.py::test_a_plan_output_is_always_kept
- test_retuning_never_grows_driver_signatures | PORTED | tests/compile/test_compile_units.py::test_retuning_never_grows_the_kernel_signatures
- test_an_unnjitable_step_splits_the_kernel_around_it | PORTED | tests/compile/test_compile_units.py::test_a_step_numba_cant_compile_splits_the_kernel_around_it
- test_fallback_step_fn_in_step_fns_is_the_plain_python_function | PORTED | tests/compile/test_compile_units.py::test_a_step_numba_cant_compile_splits_the_kernel_around_it (asserts units[1].fn is bad)
- test_a_genuine_runtime_bug_is_not_mistaken_for_a_fallback | PORTED | tests/compile/test_compile_units.py::test_a_runtime_error_in_a_compiled_step_propagates (+ test_an_error_that_is_not_a_compile_failure_is_not_a_fallback)

### `decider2/tests/test_compile_kernel.py` (17)

- test_topological_steps_respects_dependencies | PORTED | tests/steps/test_composition.py::test_dag_sorts_members_by_dependency
- test_topological_steps_is_a_pure_function_of_its_input | PORTED | tests/steps/test_composition.py::test_dag_keeps_written_order_for_independent_members (stable sort)
- test_topological_steps_detects_a_cycle | PORTED | tests/steps/test_composition.py::test_dag_cycle_raises
- test_kernel_signature_orders_arrays_then_params_then_outputs | DROPPED | spec-amendments "No code generation ... fixed `kernel(n, cols, valids, params_all, outs)` signature" (no per-kernel generated signature to order)
- test_kernel_signature_adds_a_validity_array_for_optional_inputs | DROPPED | same amendment (fixed signature; valids always passed); behaviour covered by tests/compile/test_compile_kernel.py::test_an_optional_input_reaches_the_step_as_none_when_invalid
- test_two_steps_sharing_a_param_name_get_distinct_slots | PORTED | tests/compile/test_compile_kernel.py::test_two_steps_sharing_a_param_name_get_their_own_values
- test_fused_kernel_is_bit_identical_to_stepped_on_the_flagship | PORTED | tests/compile/test_compile_kernel.py::test_fused_output_is_bit_identical_to_one_kernel_per_call
- test_no_param_value_is_baked_into_the_kernel | PORTED | tests/compile/test_compile_units.py::test_retuning_never_grows_the_kernel_signatures (b == [36.0]*4, signatures == 1)
- test_an_optional_input_reaches_the_step_as_none_when_invalid | PORTED | tests/compile/test_compile_kernel.py::test_an_optional_input_reaches_the_step_as_none_when_invalid
- test_mixed_dtypes_cross_between_fused_steps_as_their_declared_dtype | PORTED | tests/compile/test_compile_kernel.py::test_values_cross_between_fused_steps_as_their_declared_dtype (+ test_a_mixed_int_float_bool_pipeline_fused_equals_per_call_bit_for_bit)
- test_steps_defined_inside_a_function_now_fuse_instead_of_falling_back | PORTED | tests/compile/test_compile_kernel.py::test_a_step_defined_inside_a_function_fuses
- test_a_step_reading_an_unknown_name_is_refused_at_build_time | PORTED | tests/compile/test_compile_kernel.py::test_an_argument_that_is_no_input_const_or_param_is_refused_at_build_time
- test_a_fuse_group_of_120_steps_120_params_and_120_outputs | PORTED | tests/compile/test_compile_kernel.py::test_a_kernel_of_120_steps_120_params_and_120_outputs
- test_fuse_group_through_the_public_api_passes_the_full_ladder | PORTED | tests/compile/test_compile_kernel.py::test_a_chain_through_the_public_api_uses_param_defaults (+ tests/run/test_compiled_modes.py::test_the_three_modes_agree_exactly_on_the_flagship for the assert_equivalent ladder)
- test_parallel_group_with_scalar_params_matches_the_serial_kernel | DROPPED | Design.md 4.3 "An explicit `fuse()` / `parallel()` may come later"; Prompt.md lists `parallel()` among decider2 features not to port
- test_parallel_group_reading_a_bundle_is_refused_with_a_reason | DROPPED | same (no parallel kernels)
- test_nogil_is_only_released_when_every_step_asked | DROPPED | progress T0.4 "decider2 shipped `nogil` opt-in, the new rule is unconditional for serving"; replaced by tests/compile/test_compile_kernel.py::test_kernels_release_the_gil_and_keep_fastmath_off

### `decider2/tests/test_runtime_invoke.py` (15)

- test_every_mode_produces_the_same_answer | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer
- test_the_three_modes_agree_exactly | PORTED | tests/run/test_compiled_modes.py::test_the_three_modes_agree_exactly_on_the_flagship
- test_output_is_additive | PORTED | tests/run/test_flagship.py::test_output_is_additive
- test_untapped_intermediate_is_not_materialised | PORTED | tests/run/test_flagship.py::test_untapped_intermediates_are_not_materialised
- test_emit_materialises_an_intermediate | PORTED | tests/run/test_flagship.py::test_emit_materialises_an_intermediate
- test_drop_removes_a_declared_column | PORTED | tests/run/test_flagship.py::test_drop_removes_an_input_column
- test_retuning_changes_the_answer_without_editing_code | PORTED | tests/run/test_flagship.py::test_retuning_changes_the_answer_without_editing_code
- test_a_param_outside_its_bounds_is_rejected_by_pydantic | PORTED | tests/run/test_flagship.py::test_a_param_outside_its_bounds_is_rejected_at_run_time_naming_node_param_and_rows
- test_score_takes_a_dict_and_returns_a_dict | PORTED | tests/run/test_flagship.py::test_score_takes_a_dict_and_returns_a_dict
- test_score_agrees_with_apply_row_for_row | PORTED | tests/run/test_flagship.py::test_score_agrees_with_run_row_for_row
- test_score_and_apply_use_the_same_compiled_kernel | PORTED | tests/run/test_compiled_modes.py::test_score_and_run_use_the_same_compiled_kernel
- test_required_null_raises_naming_the_column_count_and_example_rows | PORTED | tests/run/test_inputs_and_score.py::test_a_required_null_raises_naming_the_input_the_step_and_the_null_count (names input, step, count; no example rows)
- test_required_null_routes_the_row_instead_of_raising_by_default | DROPPED | T3.1 "REQUIRED nulls raise MissingInputError(input, path, null_count); routing, NOT_APPLICABLE_AS and fill reporting removed"
- test_missing_as_fill_is_applied_before_the_step_sees_it | PORTED | tests/run/test_inputs_and_score.py::test_a_missing_as_null_is_filled_before_the_step_sees_it
- test_boundary_shim_handles_a_nullable_float64_column_round_trip | PORTED | tests/run/test_inputs_and_score.py::test_a_missing_as_null_is_filled_before_the_step_sees_it (nullable Float64 through every mode's boundary; same [6100,-1500,9000,...] shape)

### `decider2/tests/test_runtime_modes.py` (8)

- test_interpreted_matches_the_flagship_answer | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer[interpreted]
- test_stepped_matches_the_flagship_answer | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer[stepped]
- test_fused_matches_the_flagship_answer | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer[fused]
- test_the_three_modes_agree_exactly | PORTED | tests/run/test_compiled_modes.py::test_the_three_modes_agree_exactly_on_the_flagship
- test_the_three_modes_agree_exactly_when_fused_into_one_kernel | PORTED | tests/run/test_compiled_modes.py::test_fused_runs_the_flagship_as_one_kernel_and_stepped_as_one_per_step (+ test_the_three_modes_agree_exactly_on_the_flagship)
- test_shared_is_passed_by_reference_to_steps_that_ask_for_it | DROPPED | spec-amendments "its reserved `shared` *bundle* is dropped; the top-level \"shared\" key in the params document stays" (shared keys: tests/run/test_params_namespacing.py::test_a_shared_param_is_read_from_the_shared_entry)
- test_reads_params_bundle_is_passed_as_a_namedtuple | PORTED | tests/compile/test_compile_kernel.py::test_a_row_node_is_its_own_unit_called_with_row_params_and_consts (adapted: bare `params` bundle replaced by the row call convention fn(row, params, consts), spec-amendments T1.1b)
- test_optional_input_reaches_the_step_as_a_real_none | PORTED | tests/run/test_compiled_modes.py::test_an_optional_input_reaches_every_mode_as_a_real_none

### `decider2/tests/test_runtime_serve.py` (13)

- test_pipeline_serve_returns_a_serve_handle | DROPPED | T6.2 "Dropped: decider2 params-play routes, sealed/live modes" (ServeHandle replaced by RequestHandler, tested in tests/serving/test_serving.py)
- test_serve_rejects_an_unknown_mode | DROPPED | T6.2 "sealed/live modes" dropped (engine mode check: tests/run/test_flagship.py::test_an_unknown_mode_is_an_error_listing_the_modes)
- test_params_only_activation_does_not_recompile | PORTED | tests/serving/test_serving.py::test_staging_and_activating_a_params_only_version_compiles_nothing
- test_stage_validates_before_returning_a_plan | PORTED | tests/serving/test_serving.py::test_a_failing_stage_raises_and_the_active_version_keeps_serving[invalid value]
- test_stage_rejects_an_unknown_module | PORTED | tests/serving/test_serving.py::test_a_failing_stage_raises_and_the_active_version_keeps_serving[unknown namespace]
- test_activate_requires_a_prior_stage | PORTED | tests/serving/test_serving.py::test_activate_requires_a_prior_stage
- test_rollback_requires_a_prior_activate | PORTED | tests/serving/test_serving.py::test_rollback_requires_a_previous_activation
- test_resolved_params_round_trips_through_stage_and_activate | DROPPED | T6.2 "Dropped: decider2 params-play routes" (partial params staging replaced by whole config versions)
- test_unmentioned_modules_survive_a_partial_stage | DROPPED | T6.2 "Dropped: decider2 params-play routes" (no partial stage)
- test_preview_does_not_mutate_state | DROPPED | T6.2 "Dropped: decider2 params-play routes" (no preview)
- test_gil_report_flags_every_flagship_kernel_as_holding_the_gil | DROPPED | T0.4 "decider2 shipped `nogil` opt-in, the new rule is unconditional for serving" (no GIL report; tests/compile/test_compile_kernel.py::test_kernels_release_the_gil_and_keep_fastmath_off)
- test_structure_fingerprint_is_stable_and_ignores_param_values | DROPPED | T6.2 "sealed/live modes" dropped (fingerprint existed for the sealed guard); compile-cache fingerprint covered by tests/compile/test_compile_cache.py::test_the_fingerprint_ignores_names_and_sees_constants
- test_sealed_mode_refuses_a_plan_that_would_recompile | DROPPED | T6.2 "Dropped: ... sealed/live modes"

### `decider2/tests/test_score_plan.py` (13)

- test_plan_interface_and_flat_shape_are_built_once_and_held | PORTED | tests/run/test_compiled_modes.py::test_repeated_calls_do_not_rebuild_the_driver (no_recompile over repeated run/score)
- test_plan_holds_exactly_the_schema_invariant_pieces | DROPPED | unrecorded: asserts decider2 ScorePlan internals (slots, terminal_names, spaces) with no counterpart; behaviour covered by tests/run/test_score.py::test_score_returns_the_record_plus_every_output
- test_direct_invoke_score_and_pipeline_score_run_the_same_plan_code | PORTED | tests/run/test_score.py::test_score_returns_the_record_plus_every_output
- test_every_public_mutator_yields_a_pipeline_with_its_own_plan | PORTED | tests/run/test_score.py::test_an_emit_is_seen_only_by_the_pipeline_that_asked_for_it (emit; steps are frozen dataclasses, drop/relabel return new steps)
- test_a_hostile_in_place_field_swap_is_detected_not_served_stale | DROPPED | unrecorded: new design relies on frozen steps (decider/steps/configurable.py "frozen: changing one means a new object, whose IR is built afresh"; IR cache keyed by id(step), progress T1.1); object.__setattr__ would serve stale IR
- test_sixteen_threads_score_concurrently_with_no_cross_talk | PORTED | tests/run/test_score.py::test_sixteen_threads_score_concurrently_with_no_cross_talk
- test_a_re_entrant_call_on_one_thread_never_shares_the_busy_pool | DROPPED | unrecorded: decider2 ScorePlan row pool has no counterpart (new score path allocates per call; decider/engine/run has no pool)
- test_pooled_validity_and_fill_state_never_leaks_between_calls | PORTED | tests/run/test_score.py::test_validity_and_fills_never_leak_between_calls
- test_retune_takes_effect_and_compiles_nothing | PORTED | tests/run/test_compiled_modes.py::test_retuning_never_recompiles
- test_serve_handle_generation_swap_takes_effect_and_compiles_nothing | PORTED | tests/serving/test_serving.py::test_staging_and_activating_a_params_only_version_compiles_nothing (answer swap: test_activate_switches_answers_and_rollback_restores_them)
- test_a_bad_retune_is_still_a_hard_error | PORTED | tests/run/test_score.py::test_a_bad_retune_is_a_hard_error
- test_score_still_agrees_with_apply_on_the_flagship_and_an_optional_input | PORTED | tests/run/test_compiled_modes.py::test_score_equals_run_row_for_row_through_a_branch_and_nulls (+ assert_equivalent's score()==run() check in test_the_three_modes_agree_exactly_on_the_flagship)
- test_flagship_score_p50_is_within_the_60us_spec | PORTED | tests/run/test_score.py::test_fused_flagship_score_p50_is_within_60_microseconds

### `decider2/tests/test_serving.py` (12)

- test_ping_is_200_when_the_model_is_ready | PORTED | tests/serving/test_serving.py::test_ping_is_503_until_a_version_is_active (asserts 200 once active)
- test_invocations_returns_a_real_decision | PORTED | tests/serving/test_serving.py::test_invocations_scores_a_record_and_follows_activation (+ test_record_fields_survive_in_the_answer)
- test_params_round_trip | DROPPED | T6.2 "Dropped: decider2 params-play routes"
- test_params_schema_is_pydantics_own_json_schema | DROPPED | T6.2 "Dropped: decider2 params-play routes" (schema itself: tests/ir/test_worked_example.py::test_json_schema_nests_params_by_path)
- test_a_params_change_actually_changes_the_decision | PORTED | tests/serving/test_serving.py::test_activate_switches_answers_and_rollback_restores_them (params change arrives as a new config version)
- test_params_preview_shows_both_sides_without_activating | DROPPED | T6.2 "Dropped: decider2 params-play routes"
- test_a_validation_error_is_400_not_500 | DROPPED | T6.2 "Dropped: decider2 params-play routes" (PUT /params); handler-level: tests/serving/test_serving.py::test_a_failing_stage_raises_and_the_active_version_keeps_serving
- test_an_unknown_module_is_also_400_not_500 | DROPPED | T6.2 "Dropped: decider2 params-play routes"
- test_rollback_restores_the_previous_answer | PORTED | tests/serving/test_serving.py::test_activate_switches_answers_and_rollback_restores_them
- test_rollback_with_nothing_to_roll_back_to_is_400_not_500 | PORTED | tests/serving/test_serving.py::test_rollback_requires_a_previous_activation (handler-level; no HTTP rollback route: T6.2 "cross-process swaps need an admin route")
- test_health_reports_mode_fingerprint_generations_and_gil | DROPPED | T6.2 "sealed/live modes" dropped + T0.4 nogil unconditional (no /health route)
- test_health_fingerprint_is_unchanged_by_a_params_change | DROPPED | T6.2 "sealed/live modes" dropped (no /health route)

### `decider2/tests/test_cli.py` (8)

- test_loads_the_conventional_module_level_pipeline_name | DROPPED | T6.3 "Dropped: decider2 pipeline discovery (settings `module:attr` instead)"
- test_an_explicit_attr_name_is_honoured | PORTED | tests/cli/test_cli.py::test_build_honours_an_explicit_pipeline_attribute
- test_no_pipeline_found_is_a_clean_cli_error | DROPPED | T6.3 "Dropped: decider2 pipeline discovery"
- test_a_missing_file_is_a_clean_cli_error | PORTED | tests/cli/test_cli.py::test_build_fails_cleanly_when_the_pipeline_is_not_importable
- test_a_dotted_module_path_also_works | PORTED | tests/cli/test_cli.py::test_build_honours_an_explicit_pipeline_attribute (module:attr import path)
- test_build_precompiles_and_reports | PORTED | tests/cli/test_cli.py::test_build_stages_warms_and_records_the_cpu_target
- test_build_verify_is_the_zero_compilation_release_gate | DROPPED | T6.3 "Dropped: ... `build --verify`"
- test_build_verify_fails_cleanly_when_the_shim_is_unavailable | DROPPED | T6.3 "Dropped: ... `build --verify`"

### `decider2/tests/test_no_arity_ceiling.py` (4)

- test_a_400_feature_tree_builds_and_answers_correctly | PORTED | tests/trees/test_trees.py::test_a_400_feature_tree_builds_and_answers_correctly
- test_more_than_six_computed_features_in_one_tree | PORTED | tests/trees/test_trees.py::test_more_than_six_computed_features_in_one_tree
- test_a_table_of_comparable_width_builds_and_answers_correctly | PORTED | tests/tables/test_tables.py::test_a_400_condition_table_builds_and_answers_correctly (every mode; ported by T7.1)
- test_more_than_eight_eq_and_in_conditions_in_one_table | PORTED | tests/tables/test_tables.py::test_twenty_eq_and_twenty_in_conditions_in_one_table (every mode; ported by T7.1)

### `decider2/tests/test_graph_module.py` (17)

- test_single_step_module_derives_its_name_from_the_step | PORTED | tests/steps/test_function_step.py::test_every_spelling_of_step_produces_identical_fields (name == fn name)
- test_multi_step_module_requires_name | DROPPED | IR.md §11 "dag() replaces module()"; progress T1.1 "anonymous flows/dags inline"
- test_restating_the_derived_name_is_a_lint_error | DROPPED | module()-only lint; IR.md §11 "dag() replaces module()"
- test_step_output_can_be_overridden | PORTED | tests/steps/test_function_step.py::test_output_overrides_the_output_name_but_not_the_step_name
- test_duplicate_output_inside_one_module_is_a_build_error | PORTED | tests/steps/test_composition.py::test_dag_with_two_writers_of_one_name_raises_suggesting_flow
- test_near_miss_unbound_input_is_a_build_error_with_a_suggestion | PORTED | tests/wiring/test_wiring_errors.py::test_a_near_miss_inside_a_dag_names_the_node_path
- test_genuinely_unrelated_unbound_input_is_not_an_error | PORTED | tests/wiring/test_wiring_errors.py::test_a_genuinely_unrelated_unknown_name_is_an_input_column
- test_all_four_null_tiers_are_classified | PORTED | tests/steps/test_function_step.py::test_inputs_carry_their_null_policies_into_the_ir (3 tiers; NOT_APPLICABLE_AS dropped: Plan.md T1.2 "drop not_applicable_as")
- test_bind_removes_a_param_from_the_caller_facing_schema | PORTED | tests/steps/test_function_step.py::test_bind_sets_a_params_default_without_touching_the_original (semantics changed: progress T1.1 "bind() changes the default only")
- test_bind_rejects_an_unknown_param | PORTED | tests/steps/test_function_step.py::test_bind_rejects_an_unknown_param
- test_relabel_reads_renames_an_input_without_touching_the_original | PORTED | tests/steps/test_composition.py::test_relabel_reads_renames_an_input_without_touching_the_original
- test_relabel_writes_renames_an_output | PORTED | tests/steps/test_composition.py::test_relabel_writes_renames_an_output
- test_calling_a_module_renames_it_for_reuse | PORTED | tests/steps/test_function_step.py::test_named_renames_a_copy (.named replaces __call__)
- test_reusing_the_same_module_instance_twice_in_a_pipeline_is_an_error | PORTED | tests/ir/test_to_ir.py::test_a_path_clash_raises_suggesting_a_name (flow(ratio, ratio) raises; same step under two parents is legal per IR.md §10 test 6)
- test_renamed_reuse_is_allowed | PORTED | tests/ir/test_to_ir.py::test_a_path_clash_raises_suggesting_a_name (.named("ratio_2") passes) + test_the_same_step_placed_twice_gets_two_paths
- test_contract_true_snapshots_to_the_derived_path | DROPPED | Old-test coverage triage (progress.md): `contract=` snapshots dropped
- test_contract_catches_a_breaking_change | DROPPED | Old-test coverage triage (progress.md): `contract=` snapshots dropped

### `decider2/tests/test_graph_pipeline.py` (12)

- test_waterfall_overwrite_across_modules_is_not_an_error | PORTED | tests/wiring/test_versions.py::test_a_waterfall_keeps_every_write_as_a_version
- test_final_version_is_the_default_terminal | PORTED | tests/wiring/test_versions.py::test_the_final_version_is_the_output
- test_near_miss_across_module_boundary_is_a_build_error | PORTED | tests/wiring/test_wiring_errors.py::test_a_near_miss_across_a_named_flow_boundary_is_a_typo
- test_consumed_intermediate_is_excluded_a_terminal_is_included | PORTED | tests/wiring/test_versions.py::test_a_consumed_intermediate_is_not_an_output_a_value_nothing_reads_is
- test_emit_makes_a_consumed_intermediate_addressable | PORTED | tests/wiring/test_emit_drop.py::test_emit_makes_a_consumed_intermediate_an_output + tests/steps/test_composition.py::test_emit_and_drop_return_new_flows_and_reach_the_ir
- test_emit_of_an_unproducible_name_is_a_build_error | PORTED | tests/wiring/test_emit_drop.py::test_emit_of_an_unknown_name_is_an_error_with_a_suggestion
- test_emit_qualified_by_a_producer_that_never_produced_it_is_an_error | PORTED | tests/wiring/test_emit_drop.py::test_emit_qualified_by_a_step_that_never_wrote_it_lists_the_producers
- test_drop_of_a_leaf_input_is_allowed | PORTED | tests/wiring/test_emit_drop.py::test_drop_of_an_input_column_removes_it
- test_drop_of_an_internal_wire_is_a_build_error | PORTED | tests/wiring/test_emit_drop.py::test_drop_of_an_internal_value_is_an_error
- test_flow_and_pipe_agree | PORTED | tests/steps/test_composition.py::test_flow_and_pipe_agree
- test_flow_flattens_nested_pipelines | PORTED | tests/steps/test_composition.py::test_flow_merges_anonymous_flows (named flows now stay a unit: test_a_named_flow_stays_a_unit)
- test_pipe_mixes_module_and_bare_function | PORTED | tests/steps/test_composition.py::test_function_pipe_step_works

### `decider2/tests/test_params_namespacing.py` (11)

- test_a_multi_step_module_is_tuned_through_its_module_name | PORTED | tests/run/test_params_namespacing.py::test_a_named_flow_is_tuned_through_its_path (namespace is now the nested step path)
- test_a_knob_reaches_the_step_that_declared_it_not_its_sibling | PORTED | tests/params/test_params_validate.py::test_each_node_gets_only_its_own_params_not_its_siblings
- test_score_is_namespaced_the_same_way_as_apply | PORTED | tests/run/test_params_namespacing.py::test_score_is_namespaced_the_same_way_as_run
- test_an_unknown_params_namespace_is_a_hard_error_with_a_suggestion | PORTED | tests/run/test_params_namespacing.py::test_an_unknown_params_namespace_is_a_hard_error_with_a_suggestion
- test_a_misspelled_field_inside_a_known_namespace_is_a_hard_error | PORTED | tests/run/test_spec_conformance.py::test_a_misspelled_param_is_rejected + tests/params/test_params_validate.py::test_a_misspelled_param_is_invalid_with_a_suggestion
- test_a_bound_value_changes_the_answer | PORTED | tests/steps/test_function_step.py::test_a_bound_value_changes_the_answer_of_run_and_score (every mode)
- test_a_bound_value_leaves_the_caller_facing_interface | DROPPED | progress T1.1 "bind() changes the default only" (bound param stays tunable)
- test_overriding_a_bound_value_is_an_error_not_a_silent_win | DROPPED | progress T1.1 "bind() changes the default only" (overriding is now legal)
- test_a_bound_value_is_still_validated_by_the_model | PORTED | tests/steps/test_function_step.py::test_a_bound_value_outside_the_params_bounds_is_rejected (every mode)
- test_binding_one_knob_leaves_its_siblings_tunable | DROPPED | progress T1.1 "bind() changes the default only" (every knob stays tunable)
- test_binding_does_not_change_the_compiled_signature_count | PORTED | tests/run/test_compiled_modes.py::test_retuning_never_recompiles (a bound value is just a param default, passed as an argument)

### `decider2/tests/test_params_scratch.py` (28)

- test_param_returns_the_default_value | PORTED | tests/params/test_params_declare.py::test_param_returns_the_default_value
- test_param_is_directly_usable_where_a_float_is_expected | PORTED | tests/params/test_params_declare.py::test_param_is_directly_usable_where_a_float_is_expected
- test_param_carries_field_info_with_forwarded_kwargs | PORTED | tests/params/test_params_declare.py::test_param_carries_field_info_with_forwarded_kwargs
- test_param_is_an_instance_of_paramspec | PORTED | tests/params/test_params_declare.py::test_param_is_an_instance_of_paramspec_not_missing_as
- test_param_supports_every_documented_carrier_type | PORTED | tests/params/test_params_declare.py::test_param_carries_every_subclassable_default_type
- test_param_rejects_bool_naming_the_alternative | DROPPED | spec-amendments "param(default, ...) accepts any default, including None and bool" (inverse: test_params_declare::test_bool_and_none_defaults_are_accepted_as_plain_markers)
- test_param_rejects_none | DROPPED | spec-amendments "param(default, ...) accepts any default, including None and bool"
- test_missing_as_returns_the_fill_value_and_is_directly_callable | PORTED | tests/params/test_params_declare.py::test_missing_as_returns_the_fill_value_and_is_directly_callable
- test_not_applicable_as_returns_the_fill_value | DROPPED | Plan.md T1.2 "drop not_applicable_as"
- test_missing_as_and_not_applicable_as_are_distinguishable | PORTED | tests/params/test_params_declare.py::test_missing_as_and_param_are_distinguishable (NA half dropped: Plan.md T1.2)
- test_missing_as_rejects_bool_and_none | PORTED | tests/params/test_params_declare.py::test_missing_as_rejects_none_naming_the_optional_spelling (bool now accepted: test_missing_as_accepts_bool_as_a_plain_marker)
- test_not_applicable_as_rejects_bool_and_none | DROPPED | Plan.md T1.2 "drop not_applicable_as"
- test_harvest_required_tier | PORTED | tests/params/test_params_declare.py::test_harvest_required_inputs
- test_harvest_missing_as_tier | PORTED | tests/params/test_params_declare.py::test_harvest_missing_as_input
- test_harvest_optional_tier_from_union_annotation | PORTED | tests/params/test_params_declare.py::test_harvest_optional_input_from_union_annotation
- test_harvest_not_applicable_as_tier | DROPPED | Plan.md T1.2 "drop not_applicable_as"
- test_harvest_separates_params_from_inputs | PORTED | tests/params/test_params_declare.py::test_harvest_separates_params_from_inputs
- test_harvest_bare_params_and_shared_are_flagged_not_inputs | DROPPED | Plan.md T1.2 "drop ... the reserved shared bundle"; spec-amendments "Renames" (inverse: test_params_declare::test_params_and_shared_are_ordinary_input_names)
- test_harvest_rejects_var_args | PORTED | tests/params/test_params_declare.py::test_harvest_rejects_var_args
- test_build_params_model_is_namespaced_and_matches_hand_written | PORTED | tests/params/test_params_declare.py::test_node_model_is_named_by_path_and_enforces_bounds
- test_build_params_model_returns_none_when_no_params | PORTED | tests/params/test_params_declare.py::test_a_node_with_no_params_gets_an_empty_bundle
- test_a_hand_written_model_and_a_harvested_one_agree_on_values | PORTED | tests/params/test_params_declare.py::test_a_hand_written_model_and_a_harvested_one_agree
- test_a_misspelled_param_is_a_hard_error | PORTED | tests/params/test_params_declare.py::test_a_misspelled_param_is_a_hard_error_in_the_model
- test_parse_docstring_with_implements_line | DROPPED | Old-test coverage triage (progress.md): docstring-as-description/`Implements:` dropped
- test_parse_docstring_without_implements_line | DROPPED | Old-test coverage triage (progress.md): docstring-as-description/`Implements:` dropped
- test_parse_docstring_none | DROPPED | Old-test coverage triage (progress.md): docstring-as-description/`Implements:` dropped
- test_harvest_step_end_to_end | PORTED | tests/steps/test_function_step.py::test_every_spelling_of_step_produces_identical_fields + test_name_and_nogil_reach_the_ir (name/fn/inputs/params); doc/implements half dropped (see parse_docstring)
- test_harvest_step_name_override | PORTED | tests/steps/test_function_step.py::test_name_and_nogil_reach_the_ir

### `decider2/tests/test_registry.py` (8)

- test_base_types_register_the_same_way_a_third_party_would | PORTED | tests/trees/test_tree_config.py::test_a_tree_config_loads_by_alias_and_round_trips_with_node_positions (tree loads through ConfigurableStep.load by alias like any subclass)
- test_from_config_decision_tree_builds_a_working_module | PORTED | tests/trees/test_tree_config.py::test_a_tree_config_loads_by_alias_and_round_trips_with_node_positions (+ tree runs in tests/trees/*)
- test_from_config_decision_table_builds_a_structurally_equivalent_module | PORTED | tests/tables/test_tables.py::test_a_table_round_trips_through_json (loads by the `decision_table` alias; same answers)
- test_third_party_extension_needs_one_class_in_one_file | PORTED | tests/steps/test_configurable_integration.py::test_loads_through_a_step_ref_by_alias_or_import_path (ThresholdRule user subclass)
- test_register_rejects_a_mismatched_type_literal | DROPPED | progress T2.1 "Import-path tags plus Literal aliases" (the alias is the `type` Literal; no separate id to mismatch)
- test_register_rejects_reusing_an_id_for_a_different_class | PORTED | tests/registry/test_registry.py::test_two_classes_claiming_one_alias_is_an_error_naming_both
- test_from_config_unknown_type_suggests_the_closest_registered_id | PORTED | tests/registry/test_registry.py::test_unknown_tag_suggests_the_closest_registered_one
- test_from_config_validates_the_resolved_class_normally | PORTED | tests/registry/test_registry.py::test_resolved_class_validates_its_fields_normally

### `decider2/tests/test_string_params.py` (5)

- test_a_string_literal_param_never_grows_kernel_signatures | PORTED | tests/run/test_compiled_modes.py::test_retuning_a_string_literal_never_recompiles
- test_str_param_resolves_through_the_columns_categories | PORTED | tests/run/test_spec_conformance.py::test_changing_a_string_literal_is_a_value_change (end to end; resolve_params unit seam gone, codes from a per-runner table, T3.3A)
- test_a_literal_absent_from_categories_resolves_to_the_sentinel_not_an_error | PORTED | tests/run/test_spec_conformance.py::test_a_literal_absent_from_the_data_never_matches
- test_a_step_reading_two_str_inputs_is_a_clear_error_not_a_guess | PORTED | tests/run/test_compiled_modes.py::test_a_step_reading_two_string_inputs_is_a_clear_error_when_compiled
- test_a_str_input_with_no_str_param_is_a_named_loud_error | PORTED | tests/run/test_spec_conformance.py::test_a_string_compared_with_a_literal_in_the_body_is_a_loud_error_when_compiled

### `decider2/tests/test_shared_bundle_cache.py` (13)

- test_same_prefix_and_fields_is_the_same_class_object | PORTED | tests/testing/test_shared_bundle_cache.py::test_field_order_is_part_of_the_class_identity
- test_qualname_round_trips_prefix_and_every_field | PORTED | tests/testing/test_shared_bundle_cache.py::test_the_qualname_round_trips_awkward_field_names
- test_class_is_registered_on_the_module_under_its_qualname | PORTED | tests/testing/test_shared_bundle_cache.py::test_the_class_is_registered_on_the_module_and_pickles_by_reference_with_numbas_pickler
- test_class_pickles_by_reference_and_unpickles_to_the_same_object | PORTED | tests/testing/test_shared_bundle_cache.py::test_the_class_is_registered_on_the_module_and_pickles_by_reference_with_numbas_pickler
- test_a_process_that_never_built_the_class_resolves_the_reference_to_its_own | PORTED | tests/testing/test_shared_bundle_cache.py::test_a_process_that_never_built_the_class_rebuilds_it_from_the_reference
- test_unknown_module_attributes_still_raise_attribute_error | PORTED | tests/testing/test_shared_bundle_cache.py::test_other_module_attributes_still_raise_attribute_error
- test_resolve_params_hands_the_kernel_a_registered_class | DROPPED | reserved `shared` bundle: Plan.md T1.2 "drop ... the reserved shared bundle"; T3.4 "what still applies of test_shared_bundle_cache"
- test_shared_bundle_values_are_what_the_step_sees | DROPPED | reserved `shared` bundle: Plan.md T1.2; shared_key values covered by test_shared_bundle_cache.py::test_two_nodes_sharing_a_key_agree_across_modes
- test_a_table_step_gets_only_the_fields_it_declares_and_the_same_class_alone_or_composed | PORTED | tests/testing/test_shared_bundle_cache.py::test_a_nodes_bundle_holds_only_its_own_fields_whatever_it_is_composed_with (scalar steps; table variant rides on T4.3)
- test_a_missing_declared_shared_field_is_a_clear_error_not_a_numba_typing_error | DROPPED | table rows via reserved shared bundle: Plan.md T1.2 "drop ... the reserved shared bundle" (tables take rows as TableRef params, T4.3)
- test_two_composed_tables_agree_across_all_modes | PORTED | tests/testing/test_shared_bundle_cache.py::test_two_nodes_sharing_a_key_agree_across_modes (composed-table variant rides on T4.3)
- test_a_shared_reading_table_is_a_genuine_cache_hit_in_a_fresh_process | PORTED | tests/testing/test_shared_bundle_cache.py::test_a_params_reading_row_kernel_is_a_disk_cache_hit_in_a_fresh_process
- test_a_table_compiled_alone_is_still_a_cache_hit_when_composed_with_another | PORTED | tests/testing/test_shared_bundle_cache.py::test_a_nodes_bundle_holds_only_its_own_fields_whatever_it_is_composed_with (same bundle class alone or composed => same signature)

### `decider2/tests/test_testing.py` (16)

- test_assert_equivalent_passes_when_all_three_modes_agree | PORTED | tests/testing/test_testing.py::test_assert_equivalent_passes_and_returns_the_output_when_every_mode_agrees
- test_assert_equivalent_passes_over_the_generated_boundary_corpus | PORTED | tests/testing/test_testing.py::test_assert_equivalent_passes_over_every_corpus_frame
- test_assert_equivalent_passes_over_the_empty_corpus_frame | PORTED | tests/testing/test_testing.py::test_assert_equivalent_passes_over_every_corpus_frame[empty]
- test_assert_equivalent_rejects_a_mode_kwarg | DROPPED | no record; superseded by T3.4's `modes=` narrowing (test_modes_can_be_narrowed); a stray `mode=` is Python's own TypeError
- test_flaky_global_step_breaks_the_ladder_and_is_localised_to_the_right_rung | PORTED | tests/testing/test_testing.py::test_a_mode_that_computes_differently_fails_naming_the_mode_and_both_values
- test_corpus_returns_a_boundary_frame_and_an_empty_frame | PORTED | tests/testing/test_testing.py::test_corpus_frames_share_one_schema_and_the_expected_row_counts
- test_corpus_covers_every_declared_input | PORTED | tests/testing/test_testing.py::test_corpus_has_a_column_per_input_typed_by_its_declaration
- test_corpus_zero_and_negative_rows_hit_the_named_column | PORTED | tests/testing/test_testing.py::test_corpus_zero_and_negative_rows_move_only_the_named_column
- test_corpus_skips_negative_for_a_bool_input_but_not_for_an_int_one | PORTED | tests/testing/test_testing.py::test_corpus_skips_negative_for_a_bool_but_not_for_an_int
- test_corpus_bool_zero_case_is_false | PORTED | tests/testing/test_testing.py::test_corpus_skips_negative_for_a_bool_but_not_for_an_int (asserts zero:active is False)
- test_corpus_int_typed_input_gets_a_near_2_53_row | PORTED | tests/testing/test_testing.py::test_corpus_gives_an_int_input_a_value_past_exact_float64
- test_corpus_float_typed_input_has_no_near_2_53_row | PORTED | tests/testing/test_testing.py::test_corpus_gives_an_int_input_a_value_past_exact_float64 (asserts none for float `a`)
- test_corpus_null_edge_is_a_genuine_null | PORTED | tests/testing/test_testing.py::test_corpus_nulls_only_inputs_no_step_requires
- test_corpus_rejects_a_source_with_no_declared_inputs | PORTED | tests/testing/test_testing.py::test_corpus_needs_a_step_that_reads_something
- test_assert_no_recompile_passes_for_a_value_only_retune | PORTED | tests/testing/test_testing.py::test_no_recompile_passes_for_a_value_only_retune
- test_assert_no_recompile_catches_a_genuine_rebuild_between_calls | PORTED | tests/testing/test_testing.py::test_no_recompile_fails_when_something_compiles

### `decider2/tests/test_control_flow.py` (17)

- test_branch_both_arms_write_the_modifies_value | PORTED | tests/control_flow/test_branch.py::test_both_arms_write_the_modified_value
- test_branch_one_arm_stays_silent_and_passes_through | PORTED | tests/control_flow/test_branch.py::test_an_arm_silent_about_a_modified_name_passes_it_through
- test_branch_x_path_reports_the_arm_that_fired | PORTED | tests/control_flow/test_branch.py::test_emitting_the_condition_by_path_reports_the_arm_each_row_took
- test_branch_cross_arm_type_disagreement_is_a_build_error | PORTED | tests/control_flow/test_branch.py::test_arms_disagreeing_on_a_modified_type_is_an_error
- test_branch_routing_form_selects_by_index | PORTED | tests/control_flow/test_branch.py::test_an_int_condition_picks_the_arm_by_index (+ tests/run/test_runners.py::test_an_int_condition_picks_the_arm_by_index)
- test_branch_needs_name | PORTED | tests/control_flow/test_branch.py::test_a_branch_needs_a_name
- test_branch_needs_modifies | PORTED | tests/control_flow/test_branch.py::test_a_branch_needs_modifies
- test_loop_runs_to_its_condition | PORTED | tests/control_flow/test_loop.py::test_a_loop_runs_each_row_until_its_condition_fails
- test_loop_hits_max_iterations | PORTED | tests/control_flow/test_loop.py::test_a_loop_stops_at_max_iterations_without_an_error
- test_loop_early_exit_does_not_run_the_full_iteration_count | PORTED | tests/control_flow/test_loop.py::test_a_loop_runs_each_row_until_its_condition_fails (rows stop after 1 / 0 iterations under max 511)
- test_loop_body_must_produce_every_carries_value | PORTED | tests/control_flow/test_loop.py::test_a_carry_the_body_never_writes_is_an_error (+ tests/wiring/test_control_flow.py::test_a_carry_the_body_never_writes_is_an_error)
- test_loop_carries_must_be_a_body_leaf_too | DROPPED | T1.3: "Not ported: decider2 loop limits (body must read carry, loop_idx, single carry)"; inverted by tests/control_flow/test_loop.py::test_a_body_need_not_read_what_it_carries
- test_loop_should_continue_cannot_read_a_non_carry_body_output | PORTED | tests/wiring/test_control_flow.py::test_a_loop_condition_reading_what_only_the_body_produces_is_an_error
- test_loop_max_iterations_is_required | PORTED | tests/control_flow/test_loop.py::test_max_iterations_is_required_and_positive
- test_loop_max_iterations_must_be_a_positive_int | PORTED | tests/control_flow/test_loop.py::test_max_iterations_is_required_and_positive (+ tests/steps/test_composition.py::test_loop_needs_a_positive_int_bound)
- test_loop_two_mutually_needed_carries_is_a_clear_build_error | DROPPED | T1.3: "Not ported: decider2 loop limits (... single carry)"; T5.1: "several carries" — inverted by tests/control_flow/test_loop.py::test_two_carries_update_together_through_a_multi_step_body
- test_nested_loop_of_branch_assert_equivalent | PORTED | tests/control_flow/test_loop.py::test_a_loop_in_a_branch_in_a_loop (same fixture, same [9.0, 1.0])

### `decider2/tests/test_control_flow_arity.py` (4)

- test_branch_with_32_leaves_and_8_params_has_no_arity_ceiling | PORTED | tests/control_flow/test_arity.py::test_a_branch_with_32_inputs_and_9_params
- test_loop_body_with_32_leaves_and_8_params_has_no_arity_ceiling | PORTED | tests/control_flow/test_arity.py::test_a_loop_body_with_33_inputs_and_8_params
- test_callee_receives_its_own_declared_types | PORTED | tests/control_flow/test_branch.py::test_arms_receive_their_declared_types (tuple indexed by int band, bool flag)
- test_str_leaf_alongside_another_input_is_a_named_build_error | PORTED | limitation removed; tests/control_flow/test_arity.py::test_a_condition_reading_a_str_input_alongside_others_packs asserts the same shape now works and packs

### `decider2/tests/test_flagship.py` (16)

- test_step_is_directly_callable_with_its_declared_default | PORTED | tests/steps/test_function_step.py::test_a_step_is_callable_with_its_param_defaults
- test_step_default_is_overridable_in_a_direct_call | PORTED | tests/steps/test_function_step.py::test_a_step_is_callable_with_its_param_defaults
- test_step_above_the_floor_is_untouched | PORTED | tests/steps/test_function_step.py::test_a_step_is_callable_with_its_param_defaults
- test_bare_function_is_a_pipeline_element | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer (flow of bare functions writes cap_by_income_band)
- test_params_are_namespaced_by_module_instance | PORTED | tests/steps/test_function_step.py::test_params_are_namespaced_by_the_step_path (+ tests/ir/test_worked_example.py::test_defaults_document_matches)
- test_the_docstring_is_the_description | DROPPED | Old-test coverage triage (progress.md): docstring-as-description/`Implements:` dropped
- test_every_mode_produces_the_same_answer | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer
- test_the_three_modes_agree_exactly | PORTED | tests/run/test_compiled_modes.py::test_the_three_modes_agree_exactly_on_the_flagship
- test_output_is_additive | PORTED | tests/run/test_flagship.py::test_output_is_additive
- test_untapped_intermediates_are_not_materialised | PORTED | tests/run/test_flagship.py::test_untapped_intermediates_are_not_materialised
- test_emit_materialises_an_intermediate | PORTED | tests/run/test_flagship.py::test_emit_materialises_an_intermediate
- test_drop_removes_a_declared_column | PORTED | tests/run/test_flagship.py::test_drop_removes_an_input_column
- test_retuning_changes_the_answer_without_editing_code | PORTED | tests/run/test_flagship.py::test_retuning_changes_the_answer_without_editing_code
- test_a_param_outside_its_bounds_is_rejected_by_pydantic | PORTED | tests/run/test_flagship.py::test_a_param_outside_its_bounds_is_rejected_at_run_time_naming_node_param_and_rows
- test_score_takes_a_dict_and_returns_a_dict | PORTED | tests/run/test_flagship.py::test_score_takes_a_dict_and_returns_a_dict
- test_score_agrees_with_apply_row_for_row | PORTED | tests/run/test_flagship.py::test_score_agrees_with_run_row_for_row

### `decider2/tests/test_spec_conformance.py` (16)

- test_a_probe_never_executes_the_authors_step_body | PORTED | tests/run/test_spec_conformance.py::test_running_never_calls_a_step_on_values_the_caller_did_not_supply (all modes)
- test_repeated_calls_do_not_rebuild_the_driver | PORTED | tests/run/test_compiled_modes.py::test_repeated_calls_do_not_rebuild_the_driver
- test_a_step_may_narrow_a_value_it_also_reads | PORTED | tests/run/test_spec_conformance.py::test_a_step_may_narrow_a_value_it_also_reads (+ tests/wiring/test_versions.py::test_a_step_may_narrow_a_value_it_also_reads)
- test_the_waterfall_agrees_across_modes | PORTED | tests/run/test_spec_conformance.py::test_a_step_may_narrow_a_value_it_also_reads (bind fixture runs every mode)
- test_the_version_chain_matches_the_answer | PORTED | tests/run/test_spec_conformance.py::test_the_version_chain_matches_the_answer
- test_a_legitimate_name_resembling_its_output_is_not_a_build_error | PORTED | tests/run/test_spec_conformance.py::test_a_name_resembling_its_own_output_is_not_an_error (+ tests/wiring/test_versions.py::test_a_legitimate_name_resembling_its_own_output_is_no_typo)
- test_an_int_step_returns_an_integer_column | PORTED | tests/run/test_spec_conformance.py::test_an_int_step_returns_an_integer_column
- test_int64_above_2_to_the_53_is_not_degraded | PORTED | tests/run/test_spec_conformance.py::test_int64_above_2_to_the_53_is_not_degraded
- test_a_bool_step_returns_a_boolean_column | PORTED | tests/run/test_spec_conformance.py::test_a_bool_step_returns_a_boolean_column
- test_a_string_input_is_never_silently_zeroed | PORTED | tests/run/test_spec_conformance.py::test_a_string_input_reaches_an_interpreted_step_as_a_string + ::test_a_string_compared_with_a_literal_in_the_body_is_a_loud_error_when_compiled
- test_a_misspelled_param_is_rejected_for_a_hand_written_model | PORTED | tests/run/test_spec_conformance.py::test_a_misspelled_param_is_rejected + tests/params/test_params_declare.py::test_a_hand_written_model_and_a_harvested_one_agree (module(params=Model) itself not in new design)
- test_a_step_reading_shared_without_shared_supplied_fails_clearly | DROPPED | spec-amendments: "its reserved `shared` *bundle* is dropped"; `shared` is an ordinary input name (tests/params/test_params_declare.py::test_params_and_shared_are_ordinary_input_names)
- test_a_frame_column_shadowing_a_step_output_is_a_build_error | PORTED | tests/run/test_spec_conformance.py::test_a_frame_column_shadowing_a_step_output_is_an_error
- test_a_string_literal_declared_as_a_param_compares_correctly | PORTED | tests/run/test_spec_conformance.py::test_a_string_literal_declared_as_a_param_compares_correctly
- test_changing_a_string_literal_is_a_value_change | PORTED | tests/run/test_spec_conformance.py::test_changing_a_string_literal_is_a_value_change (+ tests/run/test_compiled_modes.py::test_retuning_a_string_literal_never_recompiles)
- test_a_literal_absent_from_the_data_never_matches | PORTED | tests/run/test_spec_conformance.py::test_a_literal_absent_from_the_data_never_matches

### `decider2/tests/test_review_findings.py` (20)

- test_a_forward_reference_is_a_build_error_not_a_silent_leaf | PORTED | tests/wiring/test_wiring_errors.py::test_reading_an_input_column_a_later_step_writes_is_an_error_naming_both
- test_the_forward_reference_error_names_both_modules | PORTED | tests/wiring/test_wiring_errors.py::test_reading_an_input_column_a_later_step_writes_is_an_error_naming_both
- test_the_cross_module_waterfall_self_read_still_works | PORTED | tests/run/test_spec_conformance.py::test_a_step_may_narrow_a_value_it_also_reads
- test_a_chain_of_self_read_waterfalls_narrowing_the_same_name_still_works | PORTED | tests/wiring/test_wiring_errors.py::test_a_chain_of_self_read_waterfalls_is_no_forward_reference
- test_a_standalone_module_self_read_seeded_from_the_frame_still_works | PORTED | tests/wiring/test_versions.py::test_a_single_self_read_is_seeded_from_the_input_column (+ tests/run/test_spec_conformance.py::test_a_self_read_waterfall_seeded_from_the_frame_overwrites_the_input_column)
- test_score_preserves_int_precision_above_2_53 | PORTED | tests/run/test_inputs_and_score.py::test_score_preserves_int_precision_above_2_53
- test_score_types_a_bool_input_as_bool_not_float | PORTED | tests/run/test_inputs_and_score.py::test_score_types_a_bool_input_as_bool
- test_score_can_index_a_tuple_by_an_int_annotated_input | PORTED | tests/run/test_inputs_and_score.py::test_score_can_index_a_tuple_by_an_int_input
- test_assert_equivalent_rejects_a_non_polars_frame | PORTED | tests/testing/test_testing.py::test_assert_equivalent_needs_a_dataframe
- test_assert_equivalent_fails_when_all_three_modes_crash_identically | PORTED | tests/testing/test_testing.py::test_assert_equivalent_fails_when_every_mode_raises_the_same_error
- test_assert_equivalent_checks_score_agrees_with_apply | PORTED | tests/testing/test_testing.py::test_a_score_disagreeing_with_run_fails_naming_the_row
- test_score_routes_a_required_absent_input_instead_of_keyerror | DROPPED | T3.1: "refer/decline routing, NOT_APPLICABLE_AS and fill-reason reporting are dropped"; replaced by tests/run/test_inputs_and_score.py::test_score_raises_naming_a_required_input_absent_from_the_record
- test_score_fills_a_missing_as_input_when_the_key_is_absent | PORTED | tests/run/test_inputs_and_score.py::test_score_fills_a_missing_as_input_absent_from_the_record
- test_apply_routes_a_required_column_entirely_absent_from_the_frame | DROPPED | T3.1: "refer/decline routing ... dropped" (REQUIRED absent raises MissingInputError); replaced by tests/run/test_inputs_and_score.py::test_run_raises_naming_a_required_column_absent_from_the_frame
- test_apply_fills_a_missing_as_column_entirely_absent_from_the_frame | PORTED | tests/run/test_inputs_and_score.py::test_run_fills_a_missing_as_column_absent_from_the_frame
- test_a_bare_python_default_is_rejected_at_harvest | PORTED | tests/params/test_params_declare.py::test_harvest_rejects_a_bare_default_naming_both_spellings
- test_the_bare_default_error_names_the_parameter | PORTED | tests/params/test_params_declare.py::test_harvest_rejects_a_bare_default_naming_both_spellings (message names param(1.0)/missing_as(1.0), not the argument name; weak)
- test_6a_a_floating_def_in_a_pipeline_file_is_flagged | DROPPED | Old-test coverage triage (progress.md): lints 6a/6b/6c dropped
- test_6b_an_unread_params_model_field_is_a_build_error | DROPPED | Old-test coverage triage (progress.md): lints 6a/6b/6c dropped
- test_6c_a_step_parameter_never_referenced_in_the_body_is_a_build_error | DROPPED | Old-test coverage triage (progress.md): lints 6a/6b/6c dropped

### `decider2/tests/test_trees.py` (18)

- test_no_threshold_is_ever_written_into_emitted_source | DROPPED | spec-amendments "No code generation. Nothing renders Python source"; thresholds-as-data covered by tests/trees/test_trees.py::test_retuning_a_param_threshold_never_recompiles
- test_retuning_a_threshold_never_recompiles | PORTED | tests/trees/test_trees.py::test_retuning_a_param_threshold_never_recompiles
- test_a_literal_and_an_inputref_threshold_become_the_same_thing | PORTED | tests/trees/test_trees.py::test_a_literal_threshold_and_a_param_threshold_give_the_same_answers
- test_one_inputref_named_twice_is_one_knob | PORTED | tests/trees/test_trees.py::test_one_param_named_twice_is_one_knob
- test_a_tree_reports_which_leaf_it_reached | PORTED | tests/trees/test_tree_config.py::test_a_tree_reports_the_leaf_that_answered_in_every_mode (opt-in `TreeConfig.path_output`, a String column naming the leaf id, not an int result_idx)
- test_the_path_column_survives_an_explicit_emit | PORTED | tests/trees/test_tree_config.py::test_the_path_column_survives_an_explicit_emit_and_a_drop
- test_an_unconnected_branch_reaches_the_default_leaf | PORTED | tests/trees/test_trees.py::test_an_unconnected_branch_reaches_the_default_row
- test_composite_and_between_and_isin_nodes | PORTED | tests/trees/test_trees.py::test_composite_between_and_isin_nodes
- test_two_same_shaped_sibling_thresholds_do_not_collide | PORTED | tests/trees/test_trees.py::test_two_same_shaped_sibling_thresholds_do_not_collide
- test_a_tree_composes_with_ordinary_modules | PORTED | tests/trees/test_trees.py::test_a_tree_composes_with_ordinary_steps
- test_a_very_large_tree_no_longer_hits_a_line_cap | PORTED | tests/trees/test_trees.py::test_a_very_large_tree_builds_and_runs
- test_a_then_chain_deeper_than_cpython_allows_now_just_builds | PORTED | tests/trees/test_trees.py::test_a_then_chain_thousands_deep_builds_and_runs
- test_a_computed_feature_now_compiles_instead_of_being_refused | PORTED | tests/trees/test_computed_features.py::test_a_computed_feature_used_as_a_tree_condition
- test_string_matching_a_kernel_cannot_do_is_refused_not_approximated | DROPPED | spec-amendments T4.2 "Regex, case-insensitive and whitespace-trimming matches run the tree's Python walker as a Fallback"; replaced by tests/trees/test_strings.py::test_regex_case_folding_and_trimming_run_the_python_walker_in_compiled_modes
- test_contains_is_matched_in_the_kernel_by_bytes | PORTED | tests/trees/test_strings.py::test_byte_matches_compile_into_the_kernel[contains]
- test_encoding_the_same_tree_twice_is_identical | PORTED | tests/trees/test_trees.py::test_the_same_document_encodes_to_the_same_arrays
- test_strict_range_validation_matches_decider1 | PORTED | tests/trees/test_trees.py::test_strict_ranges_must_be_contiguous (+ tests/trees/test_tree_conditions.py::test_strict_range_validation_*)
- test_a_trees_thresholds_are_retunable_over_the_serving_params_endpoint | DROPPED | T6.2 "Dropped: decider2 params-play routes"; retune via score covered by tests/trees/test_trees.py::test_retuning_changes_score_without_recompiling

### `decider2/tests/test_trees_strings.py` (19)

- test_every_match_type_agrees_with_python_over_the_string_corpus | PORTED | tests/trees/test_strings.py::test_every_match_type_agrees_with_python_over_the_string_corpus
- test_a_string_is_never_truncated_at_the_utf8view_inline_boundary | PORTED | tests/trees/test_strings.py::test_values_sharing_a_twelve_byte_prefix_are_told_apart
- test_frame_shapes_give_byte_identical_outputs | PORTED | tests/trees/test_strings.py::test_frame_shapes_give_identical_outputs
- test_a_zero_row_frame_and_a_frame_with_every_row_routed_still_run | PORTED | tests/trees/test_strings.py::test_an_empty_frame_an_all_null_column_and_an_absent_column_still_run (routed half: T3.1 "refer/decline routing ... dropped")
- test_assert_equivalent_drives_all_four_rungs_over_the_generated_corpus | PORTED | tests/trees/test_strings.py::test_every_match_type_agrees_with_python_over_the_string_corpus (3 modes via run_modes + fused score per value)
- test_a_mixed_tree_with_a_threshold_and_a_string_node_agrees_in_every_mode | PORTED | tests/trees/test_strings.py::test_a_mixed_tree_with_a_threshold_and_a_string_node_agrees_in_every_mode
- test_editing_adding_and_removing_patterns_never_recompiles | PORTED | tests/trees/test_strings.py::test_editing_adding_and_removing_patterns_never_recompiles (literal patterns are consts the walker reads as data, so edits are new documents; plus ParamRef pattern retunes)
- test_a_pattern_count_change_and_a_threshold_change_share_one_kernel_signature | PORTED | tests/trees/test_strings.py::test_a_pattern_count_change_and_a_threshold_change_share_one_kernel_signature (numba type of the consts)
- test_precompile_leaves_nothing_for_the_first_string_request_to_compile | PORTED | tests/serving/test_serving.py::test_after_staging_the_first_string_tree_requests_compile_nothing (found and fixed: warm-up fed a string input 1.0)
- test_injected_span_drift_fails_between_apply_and_score_and_not_between_modes | PORTED | tests/testing/test_testing.py::test_a_score_disagreeing_with_run_fails_naming_the_row (generic score-vs-run drift detection)
- test_regex_case_folding_and_trimming_are_still_refused | DROPPED | spec-amendments T4.2 "Regex, case-insensitive and whitespace-trimming matches run the tree's Python walker as a Fallback"; replaced by tests/trees/test_strings.py::test_regex_case_folding_and_trimming_run_the_python_walker_in_compiled_modes
- test_an_inputref_pattern_is_a_str_param_retuned_by_its_key | PORTED | tests/trees/test_strings.py::test_a_pattern_param_is_retuned_by_its_key
- test_two_nodes_on_one_column_and_two_trees_on_one_column_keep_their_own_patterns | PORTED | tests/trees/test_strings.py::test_two_nodes_on_one_column_and_two_trees_on_one_column_keep_their_own_patterns
- test_exact_over_many_patterns_is_an_isin | PORTED | tests/trees/test_strings.py::test_exact_over_many_patterns_is_set_membership
- test_a_non_str_record_value_for_a_string_feature_is_a_type_error_naming_the_column | PORTED | tests/trees/test_tree_config.py::test_a_string_feature_given_a_number_is_a_type_error_naming_it
- test_a_null_string_routes_like_any_required_null | PORTED | tests/trees/test_strings.py::test_a_null_string_never_matches_not_even_the_empty_pattern (adapted: T3.1 routing dropped; tree-walker.md "A null string never matches")
- test_a_categorical_column_crosses_as_a_frame_tier_cast_to_string_until_stage_6 | PORTED | tests/boundary/test_boundary_dtypes.py::test_a_castable_mismatch_gets_exactly_one_frame_tier_cast[Categorical->STR] (plan level only, not end-to-end through a tree)
- test_a_numeric_column_declared_as_a_string_feature_is_refused_not_misread | PORTED | tests/boundary/test_boundary_dtypes.py::test_a_pair_with_no_cast_is_left_for_the_import_to_refuse[Int64->STR] (plan level)
- test_explain_reports_patterns_instead_of_matcher_steps | PORTED | tests/trees/test_tree_config.py::test_the_tree_is_one_row_node_whose_params_are_its_param_refs (inputs/one node; decider2's explain() text report has no counterpart, tree is one row node per T4.2)

### `decider2/tests/test_typed_features.py` (14)

- test_int64_above_2_53_is_compared_exactly_in_every_mode | PORTED | tests/trees/test_typed_features.py::test_int64_above_2_53_is_compared_exactly_in_every_mode
- test_an_undeclared_int_feature_is_still_float_and_still_collapses | PORTED | tests/trees/test_typed_features.py::test_an_undeclared_int_feature_is_read_as_a_float_and_collapses
- test_a_bool_feature_stays_a_bool_end_to_end | PORTED | tests/trees/test_typed_features.py::test_a_bool_feature_stays_a_bool_end_to_end
- test_an_int_feature_stays_an_int_end_to_end | PORTED | tests/trees/test_typed_features.py::test_an_int_feature_stays_an_int_end_to_end
- test_a_mixed_tree_agrees_across_every_mode_and_explains_its_kinds | PORTED | tests/trees/test_typed_features.py::test_a_mixed_tree_agrees_across_every_mode (kinds() asserted)
- test_an_ordering_comparison_on_a_categorical_is_a_build_error | PORTED | tests/trees/test_typed_features.py::test_meaningless_uses_of_a_feature_are_errors_naming_it ("numeric comparison on the string feature")
- test_other_meaningless_uses_are_build_errors | PORTED | tests/trees/test_typed_features.py::test_meaningless_uses_of_a_feature_are_errors_naming_it
- test_polars_spellings_and_dtype_objects_are_accepted | PORTED | tests/trees/test_typed_features.py::test_polars_spellings_are_accepted + tests/trees/test_tree_config.py::test_feature_types_accept_polars_spellings_and_refuse_unknown_ones
- test_retuning_an_int_threshold_never_recompiles | PORTED | tests/trees/test_typed_features.py::test_retuning_an_int_threshold_never_recompiles
- test_encoding_is_deterministic_and_carries_feat_kind | PORTED | tests/trees/test_trees.py::test_the_same_document_encodes_to_the_same_arrays
- test_an_undeclared_tree_encodes_exactly_as_before | PORTED | tests/trees/test_typed_features.py::test_an_undeclared_tree_reads_every_feature_as_a_float
- test_the_typed_layout_rule_is_slot_within_kind_in_input_order | DROPPED | layout replaced: notes/tree-walker.md "Declaring inputs grouped by kind (then by name)"; exercised by tests/trees/test_typed_features.py::test_a_mixed_tree_agrees_across_every_mode
- test_the_string_slot_carries_polars_bytes_zero_copy_as_spans | PORTED | tests/trees/test_strings.py::test_a_long_string_column_is_read_in_place_and_an_override_replaces_it
- test_a_typed_tree_path_fn_is_a_genuine_cache_hit_in_a_fresh_process | PORTED | tests/trees/test_tree_config.py::test_the_tree_walker_is_a_disk_cache_hit_in_a_fresh_process

### `decider2/tests/test_expr.py` (28)

- test_bare_name_is_a_dependency_and_evaluates_to_the_bound_value | PORTED | tests/expr/test_expr.py::test_bare_name_is_a_dependency_and_evaluates_to_the_bound_value
- test_binary_operators | PORTED | tests/expr/test_expr.py::test_binary_operators
- test_unary_minus | PORTED | tests/expr/test_expr.py::test_unary_minus
- test_comparisons | PORTED | tests/expr/test_expr.py::test_comparisons
- test_boolean_operators_and_or_not | PORTED | tests/expr/test_expr.py::test_boolean_operators_and_or_not
- test_parentheses_change_evaluation_order | PORTED | tests/expr/test_expr.py::test_parentheses_change_evaluation_order
- test_whitelisted_function_calls | PORTED | tests/expr/test_expr.py::test_whitelisted_function_calls
- test_the_owners_own_example_x_minus_y | PORTED | tests/expr/test_expr.py::test_x_minus_y_greater_than_ten
- test_numeric_literals_admitted_int_and_float | PORTED | tests/expr/test_expr.py::test_numeric_literals_admitted_int_and_float
- test_dependencies_reports_every_name_read_and_only_names | PORTED | tests/expr/test_expr.py::test_dependencies_reports_every_name_read_and_only_names
- test_dependencies_excludes_whitelisted_call_targets | PORTED | tests/expr/test_expr.py::test_dependencies_excludes_whitelisted_call_targets
- test_dependencies_excludes_numeric_literals | PORTED | tests/expr/test_expr.py::test_dependencies_excludes_numeric_literals
- test_dependencies_repeated_name_counted_once | PORTED | tests/expr/test_expr.py::test_dependencies_repeated_name_counted_once
- test_rejected_construct_names_itself_at_parse | PORTED | tests/expr/test_expr.py::test_rejected_construct_names_itself_at_parse
- test_rejected_construct_reports_its_position | PORTED | tests/expr/test_expr.py::test_rejected_construct_reports_its_position
- test_import_is_rejected_a_statement_cannot_even_be_an_expression | PORTED | tests/expr/test_expr.py::test_import_is_rejected_because_a_statement_is_not_an_expression
- test_dunder_access_via_attribute_is_rejected_as_attribute_not_dunder | PORTED | tests/expr/test_expr.py::test_rejected_construct_names_itself_at_parse[x.__class__ -> "attribute access"]
- test_star_args_on_a_call_are_rejected_even_to_a_whitelisted_function | PORTED | tests/expr/test_expr.py::test_star_args_on_a_call_are_rejected_even_to_a_whitelisted_function
- test_a_non_string_or_empty_source_is_rejected | PORTED | tests/expr/test_expr.py::test_a_blank_source_is_rejected
- test_deeply_nested_but_admitted_expression_still_evaluates | PORTED | tests/expr/test_expr.py::test_deeply_nested_but_admitted_expression_still_evaluates
- test_expression_nested_past_the_cap_is_a_clean_error_not_a_recursionerror | PORTED | tests/expr/test_expr.py::test_expression_nested_past_the_cap_is_a_clean_error_not_a_recursion_error
- test_a_very_long_expression_reports_every_one_of_its_names | PORTED | tests/expr/test_expr.py::test_a_very_long_expression_reports_every_one_of_its_names
- test_a_very_wide_boolean_expression_short_circuits_the_same_as_python | PORTED | tests/expr/test_expr.py::test_a_very_wide_boolean_expression_evaluates_like_python
- test_a_computed_feature_used_as_a_tree_condition_end_to_end | PORTED | tests/trees/test_computed_features.py::test_a_computed_feature_used_as_a_tree_condition
- test_the_owners_full_condition_as_one_computed_boolean_end_to_end | PORTED | tests/trees/test_computed_features.py::test_a_whole_condition_as_one_computed_boolean
- test_computed_feature_dependencies_are_wired_like_any_other_feature | PORTED | tests/trees/test_tree_conditions.py::test_computed_feature_dependencies_are_wired_like_any_other_feature + tests/trees/test_computed_features.py::test_a_computed_features_columns_are_the_tree_inputs
- test_retuning_a_literal_inside_a_computed_expression_never_recompiles | PORTED | tests/trees/test_computed_features.py::test_editing_a_literal_inside_an_expression_never_recompiles
- test_len_driver_signatures_stays_one_across_eight_retunes_of_an_expr_literal | PORTED | tests/trees/test_computed_features.py::test_editing_a_literal_inside_an_expression_never_recompiles (walker.walk.signatures pinned)

### `decider2/tests/test_trees_migration.py` (10)

- test_v3_cases_ranges_bands_exactly_as_decider1 | PORTED | tests/trees/test_tree_migration.py::test_v3_range_cases_band_exactly_as_decider_old
- test_v1_string_match_subtree_exactly_as_decider1 | PORTED | tests/trees/test_tree_migration.py::test_a_string_match_subtree_exactly_as_decider_old (v1 logic as v3; v1 format deprecated)
- test_nested_unary_then_cases_ranges_exactly_as_decider1 | PORTED | tests/trees/test_tree_migration.py::test_a_unary_node_leading_to_range_cases_exactly_as_decider_old
- test_between_maps_ranges_to_output_labels | PORTED | tests/tables/test_legacy.py::test_between_maps_ranges_to_output_labels (every mode)
- test_between_default_when_outside_all_ranges | PORTED | tests/tables/test_legacy.py::test_between_default_when_outside_all_ranges (every mode)
- test_in_expression_categorical_lookup | PORTED | tests/tables/test_legacy.py::test_in_expression_categorical_lookup (every mode)
- test_and_expression_requires_all_conditions | PORTED | tests/tables/test_legacy.py::test_and_expression_requires_all_conditions (every mode)
- test_multiple_output_columns_all_populated | PORTED | tests/tables/test_legacy.py::test_multiple_output_columns_all_populated (every mode)
- test_a_migrated_tree_agrees_across_all_three_modes | PORTED | tests/trees/test_tree_migration.py::test_a_migrated_numeric_tree_agrees_in_every_mode
- test_a_migrated_table_agrees_across_all_three_modes | PORTED | tests/tables/test_legacy.py (every test runs through assert_equivalent: all modes, score, session)

### `decider2/tests/test_trees_ported_conditions.py` (17)

- test_all_numeric_comparison_operators | PORTED | tests/trees/test_legacy_conditions.py::test_numeric_comparisons_route_boundary_values
- test_between_variants | PORTED | tests/trees/test_legacy_conditions.py::test_between_is_closed_at_both_ends_and_takes_one_bound
- test_null_and_boolean_operators | PORTED | tests/trees/test_legacy_conditions.py::test_is_true_and_is_false_route_booleans (is_null half: tree-documents.md "Dropped: is_null/is_not_null (null handling is per input)")
- test_string_match_types | PORTED | tests/trees/test_legacy_conditions.py::test_every_string_match_type (now real answers, not UnsupportedInKernel)
- test_string_match_case_insensitive_and_trim | PORTED | tests/trees/test_legacy_conditions.py::test_string_match_case_insensitive_and_trimmed
- test_cases_ranges_lower_and_upper_inclusive | PORTED | tests/trees/test_legacy_conditions.py::test_range_cases_close_the_end_their_end_logic_names
- test_cases_isin | PORTED | tests/trees/test_legacy_conditions.py::test_isin_cases_route_codes
- test_cases_string_match | PORTED | tests/trees/test_legacy_conditions.py::test_string_match_cases_route_prefixes
- test_composite_and_or_not | PORTED | tests/trees/test_legacy_conditions.py::test_and_or_not_composites
- test_nested_composite | PORTED | tests/trees/test_legacy_conditions.py::test_a_composite_nests_inside_a_composite
- test_default_returned_when_no_match | PORTED | tests/trees/test_legacy_conditions.py::test_the_default_row_answers_when_nothing_matches
- test_correct_output_row_indexed | PORTED | tests/trees/test_legacy_conditions.py::test_result_idx_selects_its_output_row
- test_special_numeric_values_through_range_rules | PORTED | tests/trees/test_legacy_conditions.py::test_infinities_and_nan_fail_a_finite_range
- test_null_input_is_routed_away_not_evaluated | PORTED | tests/trees/test_legacy_conditions.py::test_a_null_number_is_a_missing_input_not_a_silent_miss (adapted: T3.1 routing dropped; T4.2 "numeric nulls raise")
- test_unary_is_in | PORTED | tests/trees/test_legacy_conditions.py::test_unary_isin_routes_listed_values
- test_cases_ranges_inputref_bounds | PORTED | tests/trees/test_legacy_conditions.py::test_range_case_bounds_come_from_params
- test_cases_string_match_inputref_pattern | PORTED | tests/trees/test_legacy_conditions.py::test_a_string_case_pattern_comes_from_a_param

### `decider2/tests/test_trees_ported_end_to_end.py` (18)

- test_nested_unary_then_cases_ranges | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_unary_rule_can_lead_to_a_cases_rule
- test_nested_cases_ranges_then_unary | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_branch_can_be_a_string_match_rule
- test_three_level_nested_tree | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_three_level_tree_routes_every_row
- test_composite_inside_nested_tree | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_composite_rule_works_inside_a_nested_tree
- test_path_unary_then_branch | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_nested_tree_visits_one_node_per_decision_and_the_leaf (adapted to reference-walker visits; the reached leaf is also `path_output`, see test_a_tree_reports_which_leaf_it_reached)
- test_path_otherwise_branch_index_is_branch_count | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_visits_the_branch_of_the_matched_condition_or_otherwise (visits)
- test_path_nested_tree_depth_reflects_decisions | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_nested_tree_visits_one_node_per_decision_and_the_leaf (visits)
- test_path_cases_ranges_records_correct_bucket_index | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_visits_the_branch_of_the_matched_condition_or_otherwise (isin cases, not ranges; visits)
- test_path_identical_inputs_always_produce_identical_paths | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_nested_tree_visits_one_node_per_decision_and_the_leaf (deterministic visits; path column: see test_a_tree_reports_which_leaf_it_reached)
- test_multi_column_output_all_fields_correct | PORTED | tests/trees/test_legacy_end_to_end.py::test_every_output_column_comes_from_the_reached_row
- test_multi_column_output_default_row_on_no_match | PORTED | tests/trees/test_legacy_end_to_end.py::test_every_output_column_comes_from_the_default_row_when_nothing_matches
- test_cases_ranges_empty_conditions_returns_otherwise | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_without_conditions_always_takes_otherwise[ranges]
- test_cases_string_match_empty_conditions_returns_otherwise | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_without_conditions_always_takes_otherwise[string_match]
- test_cases_isin_empty_conditions_returns_otherwise | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_without_conditions_always_takes_otherwise[isin]
- test_composite_rule_empty_conditions_evaluates_false | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_composite_rule_without_conditions_is_rejected (tree-documents.md "An empty composite is rejected, as decider2 did")
- test_cases_ranges_get_required_parameters_with_inputref_bounds | PORTED | tests/trees/test_tree_conditions.py::test_cases_ranges_required_params_include_param_bounds
- test_cases_string_match_get_required_parameters_with_inputref | PORTED | tests/trees/test_tree_conditions.py::test_cases_string_match_required_params_include_param_patterns
- test_cases_isin_get_required_parameters_with_inputref | PORTED | tests/trees/test_tree_conditions.py::test_cases_isin_required_params_include_a_param_value_set (+ test_legacy_end_to_end.py::test_params_referenced_inside_cases_rules_are_the_steps_params)

### `decider2/tests/test_trees_ported_migration.py` (4)

- test_v1_tree_as_basemodule_produces_correct_output | DROPPED | spec-amendments "v1/v2 documents raise a clear deprecation error naming the version"; logic kept in tests/trees/test_legacy_migration.py::test_the_v1_age_split_runs_the_same_as_a_v3_tree_and_as_flat_rules
- test_v3_numerical_and_range_nodes_execute_correctly | PORTED | tests/trees/test_legacy_migration.py::test_v3_range_cases_band_lower_inclusive
- test_tree_parse_rejects_missing_nodes | PORTED | tests/trees/test_tree_documents.py::test_v3_parse_rejects_missing_nodes
- test_tree_parse_rejects_unknown_node_type | PORTED | tests/trees/test_tree_documents.py::test_v3_parse_rejects_unknown_node_type

### `decider2/tests/test_trees_ported_parameters.py` (6)

- test_inputref_uses_default_when_no_runtime_column | PORTED | tests/trees/test_legacy_parameters.py::test_a_param_ref_uses_the_parameters_block_default
- test_inputref_uses_runtime_override | PORTED | tests/trees/test_legacy_parameters.py::test_the_params_document_overrides_the_default
- test_inputref_cannot_vary_per_row_only_per_call | PORTED | tests/trees/test_ported_extra.py::test_a_param_threshold_is_one_value_per_call_not_per_row (+ test_legacy_parameters.py::test_each_run_takes_its_own_params)
- test_inputref_between_with_two_parameters | PORTED | tests/trees/test_legacy_parameters.py::test_both_between_bounds_can_be_params
- test_computed_feature_two_column_expression_compiles | PORTED | tests/trees/test_legacy_parameters.py::test_a_computed_feature_combines_two_columns
- test_computed_feature_p_dot_attribute_syntax_is_refused | PORTED | tests/trees/test_tree_conditions.py::test_computed_feature_p_dot_attribute_syntax_is_refused

### `decider2/tests/test_tables.py` (16)

- test_editing_rows_never_recompiles | PORTED | tests/tables/test_table_params.py::test_editing_inline_rows_rebuilds_the_node_but_never_recompiles
- test_a_rebuilt_table_answers_differently_with_no_new_compile | PORTED | tests/tables/test_table_params.py::test_editing_inline_rows_rebuilds_the_node_but_never_recompiles, ::test_retuning_rows_and_their_count_never_recompiles
- test_a_table_reports_which_row_matched | DROPPED | notes/decision-tables.md "Decision tables have no matched-row output ... waits for someone to ask"; rows tried are visits: tests/tables/test_tables.py::test_the_python_matcher_visits_each_row_it_tries
- test_no_match_takes_the_default | PORTED | tests/tables/test_tables.py::test_no_match_takes_the_default
- test_upper_inclusive_moves_the_closed_end | PORTED | tests/tables/test_tables.py::test_upper_inclusive_moves_the_closed_end
- test_bound_mode_is_the_trees_range_end_logic | PORTED | tests/tables/test_tables.py::test_bound_mode_is_the_trees_range_end_logic
- test_or_of_two_bands | PORTED | tests/tables/test_tables.py::test_or_of_two_bands
- test_eq_on_a_string_column | PORTED | tests/tables/test_tables.py::test_eq_on_a_string_column
- test_score_takes_a_string_input_and_agrees_with_apply | PORTED | tests/tables/test_tables.py::test_score_takes_a_string_input (+ score==run per row via the `run` fixture)
- test_a_string_literal_in_a_table_is_a_retunable_param | PORTED | tests/tables/test_table_params.py::test_retuning_rows_and_their_count_never_recompiles (adapted: rows, strings included, retune through a table param, notes/decision-tables.md "Rows inline or from params"; no per-literal str params)
- test_in_on_numeric_sets | PORTED | tests/tables/test_tables.py::test_in_on_numeric_sets
- test_and_of_between_and_is_true | PORTED | tests/tables/test_tables.py::test_and_of_between_and_is_true
- test_non_contiguous_rows_are_rejected_unless_gaps_are_allowed | PORTED | tests/tables/test_tables.py::test_a_malformed_table_is_rejected_when_it_loads[ranges are not contiguous] (allow_gaps accepted: test_no_match_takes_the_default)
- test_an_output_column_absent_from_the_table_is_rejected | PORTED | tests/tables/test_tables.py::test_a_malformed_table_is_rejected_when_it_loads[not found in parameters columns]
- test_a_default_of_the_wrong_length_is_rejected | PORTED | tests/tables/test_tables.py::test_a_malformed_table_is_rejected_when_it_loads[must match outputs length]
- test_a_table_composes_with_ordinary_modules | PORTED | tests/tables/test_tables.py::test_a_table_composes_with_ordinary_steps

### `decider2/tests/test_tables_ported.py` (5)

- test_between_maps_ranges_to_output_labels | PORTED | tests/tables/test_legacy.py::test_between_maps_ranges_to_output_labels (every mode)
- test_between_default_when_outside_all_ranges | PORTED | tests/tables/test_legacy.py::test_between_default_when_outside_all_ranges (every mode)
- test_in_expression_categorical_lookup | PORTED | tests/tables/test_legacy.py::test_in_expression_categorical_lookup (every mode)
- test_and_expression_requires_all_conditions | PORTED | tests/tables/test_legacy.py::test_and_expression_requires_all_conditions (every mode)
- test_multiple_output_columns_all_populated | PORTED | tests/tables/test_legacy.py::test_multiple_output_columns_all_populated (every mode)

### `tests/_legacy/test_config_roundtrip.py` (5)

- test_save_creates_json_file | PORTED | tests/config/test_store.py::test_create_version_writes_one_json_file_per_key
- test_roundtrip_preserves_config_values | PORTED | tests/config/test_store.py::test_fresh_store_loads_what_another_wrote, ::test_configurable_step_and_params_round_trip (GraphModule vehicle dropped per T6.1 "legacy GraphModule round-trip tests")
- test_roundtrip_produces_correct_output | PORTED | tests/steps/test_configurable_integration.py::test_configs_and_params_from_the_config_store_run_the_same
- test_versioning_increments | PORTED | tests/config/test_store.py::test_versions_increment_and_latest_wins
- test_sync_save | PORTED | tests/config/test_store.py::test_create_version_writes_one_json_file_per_key (store is sync-only, T6.1 "Synchronous; no polling")

### `tests/_legacy/test_expression_module.py` (10)

- test_single_function | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer (any function step run)
- test_output_column_named_after_function | PORTED | tests/steps/test_function_step.py::test_a_named_lambda_writes_its_name_while_a_named_function_keeps_its_own
- test_input_columns_preserved | PORTED | tests/run/test_flagship.py::test_an_unread_frame_column_passes_through_unless_dropped, tests/wiring/test_versions.py::test_input_columns_pass_through_to_the_output
- test_sibling_dependency_wired_automatically | PORTED | tests/steps/test_composition.py::test_dag_sorts_members_by_dependency, tests/wiring/test_versions.py::test_a_dag_member_reads_its_sibling_by_dependency
- test_multi_level_dependency_chain | PORTED | tests/steps/test_composition.py::test_dag_sorts_members_by_dependency, tests/wiring/test_versions.py::test_each_node_reads_the_latest_version
- test_config_fields_on_module | PORTED | tests/steps/test_configurable_step.py::test_round_trips_through_json (config fields on a ConfigurableStep)
- test_config_values_affect_output | PORTED | tests/steps/test_configurable_integration.py::test_swapping_a_literal_for_a_param_ref_moves_the_value_not_the_answer, tests/run/test_flagship.py::test_retuning_changes_the_answer_without_editing_code
- test_config_default_used_when_not_overridden | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer (param defaults), tests/params/test_params_declare.py::test_param_returns_the_default_value
- test_union_merges_output_columns | PORTED | tests/steps/test_composition.py::test_dag_keeps_written_order_for_independent_members, tests/run/test_flagship.py::test_output_is_additive (`&` union replaced by dag)
- test_type_identifier_is_module_name | PORTED | tests/registry/test_registry.py::test_type_defaults_to_import_path_and_dumps_it, ::test_alias_loads_by_either_spelling_and_dumps_as_alias

### `tests/_legacy/test_graph_module_registry.py` (3)

- test_registered_module_validates_from_dict | PORTED | tests/registry/test_registry.py::test_alias_loads_by_either_spelling_and_dumps_as_alias, ::test_resolved_class_validates_its_fields_normally
- test_re_registration_replaces_not_duplicates | PORTED | tests/registry/test_registry.py::test_redefining_the_same_qualified_name_replaces_the_entry
- test_multiple_distinct_modules_coexist | PORTED | tests/registry/test_registry.py::test_distinct_classes_coexist

### `tests/_legacy/test_sequential_and_join.py` (9)

- test_pipe_creates_sequential | PORTED | tests/steps/test_composition.py::test_function_pipe_step_works, ::test_a_pipe_chain_is_one_flow
- test_sequential_output_is_last_step | PORTED | tests/run/test_flagship.py::test_output_is_additive, ::test_untapped_intermediates_are_not_materialised, ::test_emit_materialises_an_intermediate (consumed intermediates now need emit, by IR design)
- test_sequential_chaining | PORTED | tests/run/test_flagship.py::test_every_mode_gives_the_flagship_answer, tests/wiring/test_versions.py::test_each_node_reads_the_latest_version
- test_sequential_input_frame_keys_from_first_step | DROPPED | named input frames gone: IR.md `run(df)` takes one frame; Design.md D7 "polars expression tier is dropped"
- test_pipe_extends_existing_sequential | PORTED | tests/steps/test_composition.py::test_a_pipe_chain_is_one_flow, ::test_flow_merges_anonymous_flows
- test_join_basic | PORTED | tests/run/test_worked_example_runs.py (join_bureau left-join frame step); Design.md D7 "Polars remains for frame steps: joins"
- test_join_inner_drops_unmatched | DROPPED | T1.4 "frame steps can't change row count"; tests/run/test_runners.py::test_a_frame_step_changing_the_row_count_is_an_error
- test_join_input_frame_keys | DROPPED | named input frames gone (IR.md `run(df)`); JoinModule replaced by frame steps, Design.md D7
- test_join_then_expression | PORTED | tests/run/test_runners.py::test_names_read_after_an_unknown_lineage_frame_come_from_what_it_returns, tests/run/test_worked_example_runs.py

### `tests/_legacy/rules/test_conditions.py` (19)

- test_all_numeric_comparison_operators | PORTED | tests/trees/test_legacy_conditions.py::test_numeric_comparisons_route_boundary_values
- test_between_variants | PORTED | tests/trees/test_legacy_conditions.py::test_between_is_closed_at_both_ends_and_takes_one_bound
- test_null_and_boolean_operators | PORTED | tests/trees/test_legacy_conditions.py::test_is_true_and_is_false_route_booleans; is_null/is_not_null half DROPPED: notes/tree-documents.md "Dropped: is_null/is_not_null (null handling is per input)"
- test_string_match_types | PORTED | tests/trees/test_legacy_conditions.py::test_every_string_match_type
- test_string_match_case_insensitive_and_trim | PORTED | tests/trees/test_legacy_conditions.py::test_string_match_case_insensitive_and_trimmed
- test_cases_ranges_lower_and_upper_inclusive | PORTED | tests/trees/test_legacy_conditions.py::test_range_cases_close_the_end_their_end_logic_names
- test_cases_isin | PORTED | tests/trees/test_legacy_conditions.py::test_isin_cases_route_codes
- test_cases_string_match | PORTED | tests/trees/test_legacy_conditions.py::test_string_match_cases_route_prefixes
- test_composite_and_or_not | PORTED | tests/trees/test_legacy_conditions.py::test_and_or_not_composites
- test_nested_composite | PORTED | tests/trees/test_legacy_conditions.py::test_a_composite_nests_inside_a_composite
- test_default_returned_when_no_match | PORTED | tests/trees/test_legacy_conditions.py::test_the_default_row_answers_when_nothing_matches
- test_correct_output_row_indexed | PORTED | tests/trees/test_legacy_conditions.py::test_result_idx_selects_its_output_row
- test_special_numeric_values_through_range_rules | PORTED | tests/trees/test_legacy_conditions.py::test_infinities_and_nan_fail_a_finite_range; None case now an error by design (notes/tree-walker.md "Numeric features are required"): ::test_a_null_number_is_a_missing_input_not_a_silent_miss
- test_prioritized_first_match_wins | PORTED | tests/trees/test_legacy_conditions.py::test_the_first_matching_rule_wins
- test_prioritized_falls_back_to_default | PORTED | tests/trees/test_legacy_conditions.py::test_prioritized_rules_fall_back_to_the_default
- test_unary_is_in | PORTED | tests/trees/test_legacy_conditions.py::test_unary_isin_routes_listed_values
- test_cases_ranges_inputref_bounds | PORTED | tests/trees/test_legacy_conditions.py::test_range_case_bounds_come_from_params
- test_cases_string_match_inputref_pattern | PORTED | tests/trees/test_legacy_conditions.py::test_a_string_case_pattern_comes_from_a_param
- test_prioritized_module_with_parameters | PORTED | tests/trees/test_legacy_conditions.py::test_prioritized_rules_read_and_retune_params

### `tests/_legacy/rules/test_parameters.py` (6)

- test_inputref_uses_default_when_no_runtime_column | PORTED | tests/trees/test_legacy_parameters.py::test_a_param_ref_uses_the_parameters_block_default
- test_inputref_uses_runtime_struct_column | PORTED | tests/trees/test_legacy_parameters.py::test_the_params_document_overrides_the_default (params document replaces the struct column)
- test_inputref_runtime_overrides_default_per_row | DROPPED | notes/tree-documents.md "`parameters_col` ... ignored"; new behaviour pinned by tests/trees/test_ported_extra.py::test_a_param_threshold_is_one_value_per_call_not_per_row
- test_inputref_between_with_two_parameters | PORTED | tests/trees/test_legacy_parameters.py::test_both_between_bounds_can_be_params
- test_computed_feature_two_column_expression | PORTED | tests/trees/test_legacy_parameters.py::test_a_computed_feature_combines_two_columns
- test_computed_feature_uses_parameter | DROPPED | refusal pinned by tests/trees/test_tree_conditions.py::test_computed_feature_p_dot_attribute_syntax_is_refused (closed grammar rejects attribute access, carried from decider2); no notes/ entry records the decision

### `tests/_legacy/rules/test_tree_end_to_end.py` (29)

- test_nested_unary_then_cases_ranges | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_unary_rule_can_lead_to_a_cases_rule
- test_nested_cases_ranges_then_unary | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_branch_can_be_a_string_match_rule
- test_three_level_nested_tree | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_three_level_tree_routes_every_row
- test_composite_inside_nested_tree | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_composite_rule_works_inside_a_nested_tree
- test_prioritized_all_mode_returns_each_rule_independently | PORTED | tests/trees/test_legacy_end_to_end.py::test_all_mode_returns_each_rules_result_under_its_name (flat `<rule>.<col>` columns, notes/tree-walker.md)
- test_prioritized_all_mode_all_rules_default | PORTED | tests/trees/test_legacy_end_to_end.py::test_all_mode_gives_every_rule_the_default_when_nothing_matches
- test_path_unary_then_branch | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_nested_tree_visits_one_node_per_decision_and_the_leaf (path column via output_fn dropped, notes/tree-documents.md; visits replace it)
- test_path_otherwise_branch_index_is_branch_count | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_visits_the_branch_of_the_matched_condition_or_otherwise
- test_path_nested_tree_depth_reflects_decisions | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_nested_tree_visits_one_node_per_decision_and_the_leaf
- test_path_cases_ranges_records_correct_bucket_index | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_visits_the_branch_of_the_matched_condition_or_otherwise (isin cases; same branch-index visit)
- test_path_identical_inputs_always_produce_identical_paths | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_nested_tree_visits_one_node_per_decision_and_the_leaf (exact visit lists; path ids deterministic per notes/tree-documents.md)
- test_path_null_input_takes_otherwise_path | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_missing_numeric_input_is_an_error_not_a_silent_otherwise (inverted by design: notes/tree-walker.md Nulls, T4.2 follow-up)
- test_multi_column_output_all_fields_correct | PORTED | tests/trees/test_legacy_end_to_end.py::test_every_output_column_comes_from_the_reached_row
- test_multi_column_output_default_row_on_no_match | PORTED | tests/trees/test_legacy_end_to_end.py::test_every_output_column_comes_from_the_default_row_when_nothing_matches
- test_multi_column_output_in_prioritized_first_match | PORTED | tests/trees/test_legacy_end_to_end.py::test_first_match_with_several_output_columns_picks_the_winning_rules_row
- test_build_parameters_expr_no_runtime_column_uses_defaults | PORTED | behaviour (defaults when no override): tests/trees/test_legacy_parameters.py::test_a_param_ref_uses_the_parameters_block_default
- test_build_parameters_expr_empty_schema_returns_none | PORTED | behaviour (param-less tree runs): every param-less fixture, e.g. tests/trees/test_legacy_conditions.py::test_numeric_comparisons_route_boundary_values
- test_build_parameters_expr_parameter_with_no_default_uses_runtime | PORTED | behaviour (default-less param comes from the params document): tests/run/test_params_validation.py::test_a_missing_required_param_is_invalid, tests/steps/test_ctx_call.py::test_a_required_ref_takes_its_type_from_the_function_signature (no tree-specific run)
- test_prioritize_results_empty_list_returns_default | PORTED | tests/trees/test_legacy_end_to_end.py::test_prioritized_rules_with_no_rules_return_the_default
- test_optimized_execution_path | DROPPED | notes/tree-documents.md "`use_optimized_execution` ... ignored"
- test_optimized_execution_prioritized | DROPPED | notes/tree-documents.md "`use_optimized_execution` ... ignored"
- test_run_polars_expression_execute | PORTED | behaviour (build once, execute correctly) covered by every TreeConfig run, e.g. tests/trees/test_legacy_conditions.py::test_numeric_comparisons_route_boundary_values; internal class gone (no polars tier, Design.md D7)
- test_cases_ranges_empty_conditions_returns_otherwise | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_without_conditions_always_takes_otherwise[ranges]
- test_cases_string_match_empty_conditions_returns_otherwise | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_without_conditions_always_takes_otherwise[string_match]
- test_cases_isin_empty_conditions_returns_otherwise | PORTED | tests/trees/test_legacy_end_to_end.py::test_a_cases_rule_without_conditions_always_takes_otherwise[isin]
- test_composite_rule_empty_conditions_evaluates_false | DROPPED | notes/tree-documents.md "An empty composite is rejected ... decider_old evaluated an empty AND as false, which nobody should rely on"; tests/trees/test_legacy_end_to_end.py::test_a_composite_rule_without_conditions_is_rejected
- test_cases_ranges_get_required_parameters_with_inputref_bounds | PORTED | tests/trees/test_legacy_end_to_end.py::test_params_referenced_inside_cases_rules_are_the_steps_params, tests/trees/test_tree_conditions.py::test_cases_ranges_required_params_include_param_bounds
- test_cases_string_match_get_required_parameters_with_inputref | PORTED | tests/trees/test_legacy_end_to_end.py::test_params_referenced_inside_cases_rules_are_the_steps_params, tests/trees/test_tree_conditions.py::test_cases_string_match_required_params_include_param_patterns
- test_cases_isin_get_required_parameters_with_inputref | PORTED | tests/trees/test_legacy_end_to_end.py::test_params_referenced_inside_cases_rules_are_the_steps_params, tests/trees/test_tree_conditions.py::test_cases_isin_required_params_include_a_param_value_set

### `tests/_legacy/rules/test_tree_migration.py` (6)

- test_v1_parses_and_upgrades_to_v3 | DROPPED | spec-amendments "v1/v2 documents raise a clear deprecation error naming the version"; tests/trees/test_legacy_migration.py::test_a_v1_or_v2_tree_is_refused_naming_its_version
- test_v1_tree_as_basemodule_produces_correct_output | DROPPED | v1 deprecated (spec-amendments); same logic as v3/flat: tests/trees/test_legacy_migration.py::test_the_v1_age_split_runs_the_same_as_a_v3_tree_and_as_flat_rules
- test_v3_numerical_and_range_nodes_execute_correctly | PORTED | tests/trees/test_legacy_migration.py::test_v3_range_cases_band_lower_inclusive, tests/trees/test_tree_migration.py::test_v3_range_cases_band_exactly_as_decider_old
- test_v1_tree_parse_rejects_missing_nodes | DROPPED | v1 deprecated (spec-amendments); a v1 doc without nodes is still refused naming v1: tests/trees/test_legacy_migration.py::test_a_v1_or_v2_tree_is_refused_naming_its_version[doc2]
- test_v1_tree_parse_rejects_unknown_node_type | DROPPED | v1 deprecated (spec-amendments "v1/v2 documents raise a clear deprecation error")
- test_can_create_default_tree | DROPPED | Old-test coverage triage (progress.md): `Tree.default_tree()` dropped (UI convenience)

### `tests/_legacy/credit/decision_table/test_decision_table.py` (5)

- test_between_maps_ranges_to_output_labels | PORTED | tests/tables/test_legacy.py::test_between_maps_ranges_to_output_labels (every mode)
- test_between_default_when_outside_all_ranges | PORTED | tests/tables/test_legacy.py::test_between_default_when_outside_all_ranges (every mode)
- test_in_expression_categorical_lookup | PORTED | tests/tables/test_legacy.py::test_in_expression_categorical_lookup (every mode)
- test_and_expression_requires_all_conditions | PORTED | tests/tables/test_legacy.py::test_and_expression_requires_all_conditions (every mode)
- test_multiple_output_columns_all_populated | PORTED | tests/tables/test_legacy.py::test_multiple_output_columns_all_populated (every mode)

### `tests/_legacy/credit/scorecard/test_scorecard.py` (6)

- test_bound_bins_assign_values_by_range | PORTED | tests/scorecard/test_scorecard.py::test_bound_bins_assign_values_by_range
- test_multiple_variables_score_is_sum | PORTED | tests/scorecard/test_scorecard.py::test_multiple_variables_score_is_sum
- test_value_bins_match_categorical_inputs | PORTED | tests/scorecard/test_scorecard.py::test_value_bins_match_categorical_inputs
- test_default_bin_used_when_no_bin_matches | PORTED | tests/scorecard/test_scorecard.py::test_default_bin_used_when_no_bin_matches
- test_constant_score_adds_fixed_value_to_all_rows | PORTED | tests/scorecard/test_scorecard.py::test_constant_score_adds_fixed_value_to_all_rows
- test_adjusted_variable_applies_scale_and_offset | PORTED | tests/scorecard/test_scorecard.py::test_adjusted_variable_applies_scale_and_offset
