# Campaign Targeting Trees - Evaluation Notes

## 1. What was built

A simplified implementation of the campaign targeting trees described in doc 04 (04-campaign-targeting-trees.md). The implementation exercises:

- **Decision trees with sequential decision points**: 4-level decision tree with eligibility checks at each level
- **Multiple decision paths**: Different clients reach different leaf outcomes based on their attributes
- **Tree evaluation**: Each client is evaluated through all decision points and reaches a final targeting outcome (target=1, do not target=0)
- **Multiple leaf outcomes**: Clients can be rejected at different decision points (facility check, payment history, credit score, affordability)
- **Batch and realtime APIs**: Pipeline works in both batch mode (`apply()`) and realtime mode (`score()`)
- **Equivalence testing**: All three execution modes (interpreted, stepped, fused) produce identical results

### What was left out and why

1. **Path capture as structured output** (doc 04 §5.4): The requirement to capture and emit the complete path through the tree as first-class output was partially addressed through the decision logic but not as an explicit string output. The reason is that numba (used in the fused/compiled execution mode) does not support Python strings directly, and the framework's equivalence requirement (doc 02 §3.1) mandates that all three execution modes must produce identical output. Rather than choose which mode supports strings, the implementation captures path logic through the sequence of boolean decisions, which implicitly encode the path. This is a workaround for a framework limitation (see section 2 below).

2. **Overlays** (doc 04 §5.3.4): Not implemented. Overlays are a complex feature requiring:
   - Separation of tree structure from policy adjustments
   - Preservation of node identity across overlay changes
   - Unadjusted leaf tracking and dual evaluation paths
   The simplified spec focuses on the core tree evaluation mechanism rather than the full overlay machinery, which would require significant additional infrastructure.

3. **Suppression logic** (doc 04 §5.2): Not implemented. Full suppression evaluation with absolute/measurement-relevant classification was left out in favor of focusing the evaluation on the tree traversal mechanism itself.

4. **Pre-assessment and amount determination** (doc 04 §5.5): Not implemented. The spec focuses on targeting decisions rather than amount calculation.

5. **Arbitration** (doc 04 §5.6): Not implemented. Single-campaign evaluation is simpler than multi-campaign arbitration.

6. **Holdout and control** (doc 04 §5.7): Not implemented. The spec evaluates all clients as treated.

## 2. Every place the framework fought

### Issue 1: String handling in compiled mode

**The problem**: The path capture requirement (doc 04 §5.4) naturally maps to returning path strings from a step (e.g., "1→2→3→leaf_910"). However, numba (used in the fused/compiled execution mode) cannot handle Python strings directly in kernels.

**The doc quote**: 
> "**Doc 02 §3.1** — The equivalence ladder requires that interpreted, stepped, and fused modes produce bit-identical output."

**The workaround**: Abandoned explicit path strings in favor of implicitly encoding the path through the sequence of boolean checks. The path is documentable but not emitted. This satisfies the equivalence requirement but loses the explicit path output that the spec strongly emphasizes is "a deliverable of equal standing to the decision itself" (doc 04 §5.4, section heading).

### Issue 2: sys.path setup for package imports

**The problem**: The decider2 package has a non-standard structure: 
- The package is in `/decider2/decider2/`
- The shim at `/decider2/__init__.py` re-exports from the inner package
- When running tests from a different directory, imports fail unless sys.path is precisely configured

**The doc quote**: None specific, but this reflects the comment in `/decider2/__init__.py`:
> "This is a shim package that re-exports from the inner decider2 package."

**The workaround**: Added manual sys.path manipulation in `conftest.py`, `pipeline.py`, and `test_campaign_trees.py` to add `/decider2` (the shim directory) to the path. This is fragile and environment-dependent, but necessary given the package structure.

### Issue 3: Unclear how to emit intermediate decision logic without typing errors

**The problem**: The natural way to emit the reasoning from each decision node (e.g., "client failed affordability check") is to return strings from each step. However, this causes numba typing errors in fused mode because numba arrays have fixed types.

**What the API doesn't say clearly**: Doc 03 covers `.emit()` for keeping intermediate values (§7), but all examples emit numeric or boolean values. The authoring API doesn't address how to emit text-based metadata alongside numeric decisions when the pipeline must support fused mode.

**The consequence**: Decision reasoning cannot be carried through to the output in a way that survives all three execution modes. The implementation uses pure boolean logic, which works across modes but loses explainability.

## 3. What had to be guessed

1. **Node identity in a simplified tree**: Doc 04 §5.4.3 defines `node_key` with strict requirements for stability across versions. The simplified implementation uses simple integers (1, 2, 3, 4) as implicit node keys. Whether these are correct or how they should be hashed/encoded was not clarified in the authoring API.

2. **Leaf outcome representation**: The spec (doc 04 §5.3.2) shows leaves carrying multiple values (outcome, tier, amount, channel, priority, reason label). The simplified implementation reduces this to a single integer (1=target, 0=do not target). The correct way to represent multiple leaf values in the pipeline was not clear from the API docs.

3. **How to structure candidate population definitions**: Doc 04 §5.3 mentions "each campaign's candidate population definition is resolved" but the authoring API (doc 03) does not provide machinery for this. It was guessed that this filtering should happen upstream of the pipeline, not as part of the tree evaluation itself.

4. **Equivalence testing corpus**: The `corpus()` function is mentioned in doc 03 §11 but its usage is not shown. The tests were written using explicit test data rather than corpus-generated data.

## 4. What was wanted and could not be expressed

1. **Path output with string values**: The spec (doc 04 §5.4) makes path capture a first-class requirement equal to the decision itself. The natural expression is a string like "1→2→4→leaf_910". The framework's equivalence requirement (all three modes must agree) makes this impossible without either:
   - Breaking equivalence by supporting strings only in interpreted mode
   - Storing paths outside the pipeline outputs
   - Using a serialized representation (e.g., a list of integers) instead of a readable string

2. **Branching nodes that don't recompile**: The spec shows tree structure as fundamentally about branching (node tests true/false condition, two outgoing edges). The authoring API (doc 03 §8.2) provides a `Branch` combinator, but it would require restructuring the pipeline significantly. The linear `flow()` API was simpler to use but doesn't visually express the tree structure. No way exists to say "this pipeline is a tree" in a way that optimizes for its DAG structure or enables tree-specific analysis.

3. **Field-level nulls vs. structural nulls**: Doc 03 §1 distinguishes "missing" (data not captured) from "not applicable" (field doesn't apply). The spec references this distinction (doc 04 §7.4), but expressing it in a pipeline context requires either:
   - Union types on every field (`float | None`)
   - Explicit handling at the boundary (`missing_as(0.0)` vs. `not_applicable_as(0.0)`)
   The choice was not clear from the context of the evaluation task.

## 5. The exact curl and its output

### /ping endpoint test
```bash
curl -s http://localhost:8104/ping
```

Response:
```json
{"status": "ok"}
```

### /invocations endpoint test - qualifying client
```bash
curl -s -X POST http://localhost:8104/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "has_active_flex_loan": true,
    "months_on_book_flex": 12,
    "settlement_ratio": 0.5,
    "payments_missed_12m": 0,
    "behaviour_score": 700.0,
    "discretionary_income": 3500.0,
    "estimated_instalment_to_income": 0.25
  }'
```

Response:
```json
{
  "has_active_flex_loan": true,
  "months_on_book_flex": 12,
  "settlement_ratio": 0.5,
  "payments_missed_12m": 0,
  "behaviour_score": 700.0,
  "discretionary_income": 3500.0,
  "estimated_instalment_to_income": 0.25,
  "targeting_decision": 1
}
```

**Interpretation**: `targeting_decision` = 1 indicates the client qualifies and should be targeted. The client passed all four decision levels (facility check ✓, payment history ✓, credit score ✓, affordability ✓).

## Summary

The implementation successfully demonstrates a decision tree traversing multiple levels of sequential checks, with proper equivalence testing across all three execution modes. The main challenge was path capture: the spec emphasizes it strongly, but the framework's requirement that all execution modes produce identical output (including numerical arrays from numba) prevents natural string-based path representation. The workaround (implicit path encoding through decision logic) satisfies the equivalence requirement but loses the explicit path output that the spec values for auditability.

All tests pass, the pipeline serves correctly as an HTTP endpoint, and the three execution modes (interpreted, stepped, fused) produce byte-identical output, confirming the equivalence guarantee (doc 02 §3.1 and doc 05 §9.1).
