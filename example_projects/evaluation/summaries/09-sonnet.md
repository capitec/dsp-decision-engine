# Governance Replay (09)

## Business Purpose
Audits and verifies financial decision integrity by re-executing previously captured decisions and confirming they produce identical (or within-tolerance) results. This is a compliance tool, not a decision-maker: it detects divergence when a config change, parameter update, or deployment alters decision behavior unexpectedly.

## Inputs & Outputs
**Inputs:** flow_code (which decision flow, e.g., "03"), decision_id (which past decision to replay, e.g., "gr-2026-0000987654"), mode (execution mode, default "interpreted").

**Outputs:** replay_verdict (string: "reproduced" | "reproduced_within_tolerance" | "not_reproduced"), divergence_count (integer), diverged_fields (comma-separated string).

## Main Steps
1. Load the decision's captured evidence (original request, parameters, config version) from store
2. Rebuild that flow's pipeline using the recorded config version
3. Type-convert the recorded request
4. Re-score through the rebuilt pipeline with recorded params (no live calls)
5. Compare every field against recorded values using field-specific tolerances
6. Return verdict and divergence details

## Hard-to-Understand Elements

**Tolerance bands:** Money fields (anything with "amount," "fee," "limit" in the name) round to cents; rates to 4 decimals; scores/probabilities to 1e-6 absolute; everything else to 1e-12 relative. The tolerance strategy (`_compare_field`, `_compare_scalar`) is compact and declarative but the field-name matching is implicit.

**Framework workaround:** Pipeline deliberately rejects the decider step model because replay's governance capabilities (explanation, diff, swap-set) return deeply nested tree-shaped documents that the framework's `list[struct]` boundary can't serialize. Rather than force a flat encoding, this project exposes only the one scalable capability (replay) through the framework, and keeps the rest as a tested Python library.

**Warm-up override:** `inference.py` replaces the standard request handler's warm-up with a sample record because the default dummy request (empty string for all inputs) would trigger a KeyError — not a type mismatch, but a missing flow code / decision id.
