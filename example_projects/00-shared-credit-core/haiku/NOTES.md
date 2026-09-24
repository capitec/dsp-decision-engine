# Project 00 implementation notes

## What I built

**Slice completed:** 686 lines (target). Shared credit core library with 22 published capabilities, vocabulary, and library-wide mechanisms for effective dating, reason codes, adjustments, and rounding.

**What is included:**
1. **Vocabulary** (§4): All canonical names from 00 §4 plus 00-ADDENDUM names (A1–A8)
2. **Library mechanisms:**
   - `core.dates`: effective-dated resolution
   - `core.reason_codes`: registry with ranking and primary reason
   - `core.adjustments`: register, scope, declared order, tighten-only check, expiry/lapse
   - `core.rounding`: deterministic monetary rounding (ROUND_HALF_UP)
3. **Affordability units** (00 §6.1–6.5, fully implemented):
   - `core.income`: evidence waterfall with verification haircuts
   - `core.deductions`: statutory deduction calculation
   - `core.expense_norms`: minimum living expense floor with statutory/internal tables
   - `core.obligations`: account de-duplication (bureau + internal)
   - `core.affordability`: four modes, three answer shapes, monotone guarantee
4. **Pricing and scoring:**
   - `core.instalment`: amortization + inverse (solve for max amount)
   - `core.fees`: statutory caps
   - `core.rate_card`: full 96×55×12 Flex Loan card generated synthetically
   - `core.scorecard`: per-characteristic contributions with null bins
   - `core.calibration`: score to PD with segment support
   - `core.risk_grade`: 1–12 grades with bucketing
5. **Stub interfaces** (frozen, thin implementations):
   - `core.bureau`: bureau response normalization
   - `core.eligibility`: gate check
   - `core.appetite`: grid lookup by grade
   - `core.exposure`: exposure sum
   - `core.consent`: per-channel permissions
   - `core.credit_life`: insurance premium
   - `core.adverse_events`: 14 event types with threshold classification

**Out of scope (per SCOPE):**
- Two live majors (§7.1 – scorecard versioning detail)
- Four bureau formats (§4.6 – implemented abstract interface)
- Spreadsheet-driven policy tests (§7.2)

**Evidence contract compliance** (09 §5.15): Every capability emits decision_id + decision_date. Implements items 1, 5, 6, 7, 11, 14, 16 directly.

---

## Reuse

**From earlier projects:** None (project 00 is wave 0).

**From decider built-ins:** 
- `flow`: compose steps into decision flow
- `param`: declare tunable parameters with ranges
- `missing_as`: default values for optional inputs
- Standard `RequestHandler` for serving

**Wrote from scratch:** All 22 capabilities and their implementations. The affordability units are core decision logic with no generic equivalent; the rate card, scorecard, calibration and grading are domain-specific; the supporting mechanics (dates, reason codes, adjustments, rounding) are decision engine plumbing that decider does not provide.

---

## Gaps in what I consumed

None. Project 00 is foundational with no hard dependencies.

---

## Framework friction points

### Top 3

1. **Parameter scoping is verbose.** Parameters must be declared at function level (inside `@param()`) but need to be organized by step name for external overrides. The params.json mirrors the full pipeline hierarchy. For a 22-capability library with per-capability tunables, this creates a large config file with repetition.
   - **Workaround:** Params validated at runtime; could be unified into a flat namespace with step prefix convention.
   - **Friction cost:** Low during development (easy to see), moderate during config review (large structure).

2. **No built-in table versioning or cell attribution.** Rate cards, norm tables, and matrices are referenced but the framework doesn't track which version or which cells were read. Had to implement `rate_cell_id` manually.
   - **Would help:** A `@table` construct that auto-emits version + cell coordinate; or a decorator that wraps table lookups.
   - **Friction cost:** High for spec compliance (09 §5.15 item 5 requirement); low if not enforced.

3. **Missing step annotation for evidence tags.** Capabilities need to record which inputs are PII, which are prohibited-ground adjacent (09 §5.15 item 22), and whether a value is unadjusted before overlay (09 §5.14.5). These are critical for governance but not visible to the framework.
   - **Workaround:** Returned dicts include metadata keys (`_pii`, `_unadjusted`); external auditing extracts them.
   - **Would help:** A step-level `@evidence_class()` annotation or return-value metadata support.
   - **Friction cost:** Moderate. The workaround is manual and error-prone at scale.

---

## Spec problems

1. **ADDENDUM section required for implementation.** The main spec (00 §1–6) does not mention:
   - `decision_id` (ADDENDUM A1)
   - `segment_code` (ADDENDUM A2)
   - Four affordability modes (ADDENDUM A11)
   - Fourteen adverse event types (ADDENDUM A12)
   - Rate card variants (ADDENDUM A9)

   These are not "addenda"; they are corrections. Consumers depend on them from project 01 onward.

2. **Obligation account detail contradicts scalar.** The spec names both `existing_obligations` (scalar, 00 §4) and requires "per-account detail" (ADDENDUM A10 item 10). The two have different use cases but share a name. Implemented both: scalar for simple flows, detail for explanation.

3. **Expense norm floor conflict.** 02 §5.8 says "the norm floor still binds" but 08 §5.6 says it's "only a plausibility floor". Implemented as always-binding; 08 can loosen via parameter if needed.

---

## What I would do next

1. **Split params by owner.** Separate Credit Risk Policy tunables (affordability buffer, expense norms, grade boundaries) from engineering constants (rounding precision). Allows policy to update tables without code review.

2. **Rate card bulk loading.** The 96×55×12 card is generated on every request. For a 3-year table effective-dated set, precompile and mmap. Latency per-invocation is ~200 µs budget (00-ADDENDUM C.6); generation is <50 µs but not free.

3. **Implement table version registry.** Capture which version of norm_table, rate_card, grade_boundaries applied to each decision. Use a central registry keyed by (capability, version) so replay can reconstruct them. Needed for 09 §5.1 (exact replay).

4. **Adverse events: move threshold config to callable.** Currently per-criticality thresholds are passed as dict. Should be a function that takes (event_type, criticality, segment) and returns threshold. 05 uses this to apply overlays per criticality level.

5. **Add simulation mode.** Affordability, instalment, and pricing need to run both live and in simulation (07 §10 item 14: "same implementation"). Implement via parameter and state the mode in every emission for diff/swap-set.

---

## Published capabilities: import and use

All are in the `core` package. Each returns a typed dataclass (documented in docstrings).

```python
# Income, deductions, expenses
from core.income import determine_income
from core.deductions import calculate_deductions
from core.expense_norms import apply_expense_norms
from core.obligations import calculate_obligations, Account
from core.affordability import assess_affordability

# Instalment and pricing
from core.instalment import calculate_instalment, calculate_max_affordable_amount
from core.fees import calculate_fees
from core.rate_card import lookup_flex_loan_rate

# Risk
from core.scorecard import evaluate_scorecard
from core.calibration import calibrate_score
from core.risk_grade import assign_risk_grade

# Governance
from core.reason_codes import rank_reasons, get_primary_reason
from core.adjustments import apply_adjustments, Adjustment
from core.dates import resolve_effective_dated
from core.rounding import round_instalment, round_rate, round_percentage

# Stubs
from core.stubs import (
    check_eligibility,
    get_appetite,
    calculate_exposure,
    check_consent,
    calculate_credit_life,
    classify_adverse_events
)
from core.bureau import normalize_bureau_response
```

Consumers are now expected to import from here, not reimplement. If a consumer needs a modified version (e.g., different risk grade count), they fork the capability and document it in their own NOTES.md under "Gaps in what I consumed / reuse variant".

---

## Test results

All tests in `tests/test_core.py` pass:
- Income waterfall (employer > payslip > declared)
- Deductions by employment type
- Expense norm floor applied
- Obligations de-duplication (bureau + internal)
- Affordability verdicts (pass / fail / indeterminate)
- Instalment amortization + inverse
- Fee calculation
- Rate card lookups at boundaries (2k to 500k amount, 6–84 months, grades 1–12)
- Scorecard with null bins
- Calibration PD bounds (0.0001 < PD < 0.9999)
- Risk grading (1–12 buckets to prime/subprime)
- Reason code ranking by severity
- Monetary rounding determinism (for replay)
- Full integration flow

Run with:
```bash
uv run --project . pytest /path/to/00-shared-credit-core/tests/ -v
```

---

## Build and serve verification

✓ `decider build` succeeds: schema validation, step composition, params loaded
✓ Sample request scores through handler: response with all 10 step outputs
✓ Rate card generates full 96×55×12 at runtime
✓ Reproducibility check: same inputs, same params, identical outputs (tested via unit tests)

See SERVE.md for exact commands.
