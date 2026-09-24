# Project 09: Governance Replay (Haiku)

## What It Decides (Business Terms)

This is an audit and compliance harness for decision systems. It doesn't make credit decisions itself—it replays, audits, and explains historical decisions made by transaction fraud/lending systems. Key functions:

- **Replay**: Re-run a past decision with recorded evidence to verify it would make the same decision today (unchanged logic) or detect version drift.
- **Explain**: Generate human-readable explanations of why a decision was made, tailored for three audiences (consultant, analyst, ombudsman).
- **Diff**: Show what changed between two versions of rules/logic and estimate impact.
- **Swap-set**: Run a population through two versions and measure outcome changes (approvals gained/lost, amount shifts).
- **Overlays**: Track temporary adjustments (sensitivity dials, threshold multipliers) with aging and expiry.
- **Dead logic**: Identify rules that never fired in a 90-day window and high-coverage rules (>40% of decisions).

## Inputs and Outputs

**Inputs**: decision_id, evidence (dict with inputs_as_received, parameters, outputs, rules_fired, gates_evaluated, adjusted/unadjusted values), flow_type, audience (consultant|analyst|ombud), operation, decision_date.

**Outputs**: Vary by operation—replay returns verdict (reproduced|reproduced_within_tolerance|not_reproduced) + field diffs; explain returns tailored text; diff returns change summary; swap-set returns approval/amount impact rates; dead_logic returns dead rule list.

## Main Steps in Order

1. Request arrives with operation type and evidence (historical decision record).
2. Dispatch to operation handler: ReplayEngine, ExplanationRenderer, VersionDiffer, SwapSetAnalyzer, OverlayRegister, or DeadLogicDetector.
3. Handler processes evidence and runs its analysis.
4. Return structured result (dict or text explanation).

## Hard to Understand from Code Alone

- **Evidence schema**: The exact structure of "evidence" (gates_evaluated, rules_fired, cap_chain, score_contributions, table_cells_read) is not documented in code—only visible in sample_request.json. No schema validation or type hints in inference.
- **Tolerance logic**: `_all_within_tolerance` checks if differences are <1e-6 relative error, but the comment says "scores/probabilities" and "intermediate values" have different rules; code only checks floats uniformly.
- **Stub flow**: The `_stub_flow` function is hardcoded to return `{outcome_code: "approved", offered_amount: 100000}`. Unclear how actual decision flows (fraud, lending) integrate—are they passed in, loaded from elsewhere?
- **Audience-specific output**: ExplanationRenderer has three render methods, but the flow dispatcher treats audience as a parameter and selects rendering—unclear if explanation data structure (gates, rules, cap_chain) is pre-populated or built per audience.
- **Flow type routing**: flow_type parameter is captured but never used in dispatch logic. Suggests incomplete wiring to multi-flow systems.
- **Overlay stack evaluation**: OverlayRegister tracks overlays but doesn't show how they're applied to decision values; magnitude and stack_position fields are stored but not used in code.

