# Shared Credit Core Library - Implementation

A reference implementation of a shared credit decision library using the `decider2` framework.

This demonstrates:
1. **Module reuse** - Income, Affordability, Fees modules consumed by multiple products
2. **.relabel() usage** - Adapting modules to different column names
3. **Vocabulary mapping** - Systematic name transformation across pipelines
4. **Parameter ownership** - Library-global, library-policy, and consumer-local parameters
5. **Interface stability** - Frozen module contracts

## Directory Structure

```
00-shared-credit-core/
  README.md                         # This file
  NOTES.md                          # Implementation notes and assessment
  
  pipeline.py                       # Flex Loan pipeline (main serving file)
  serve_flex_loan.py                # Alternative serve wrapper (same content)
  test_credit_core.py               # Comprehensive test suite
  vocabulary.py                     # Example Vocabulary mappings
  
  modules/                          # Published capabilities
    income/                         # Income determination
      steps.py                      # Pure functions
      params.py                     # Pydantic models
      __init__.py                   # Module assembly
    affordability/                  # Affordability assessment
      steps.py
      params.py
      __init__.py
    fees/                           # Fee calculation
      steps.py
      params.py
      __init__.py
  
  pipelines/                        # Product pipelines
    flex_loan.py                    # Flex Loan (unsecured term loan)
    access_facility.py              # Access Facility (revolving credit)
  
  config/                           # Configuration files
    flex_loan/
      production.json               # Template
  
  tests/                            # Old test structure (reference only)
    test_income.py
    test_affordability.py
    test_pipelines.py
```

## Published Capabilities

### 1. Income (`modules/income/`)
Establishes `gross_monthly_income` and `net_monthly_income` from evidence.

**Interface:**
- **Inputs**: `declared_income`, `payslip_income`, `employment_type_code`
- **Outputs**: `gross_monthly_income`, `net_monthly_income`, `income_source_code`
- **Params**: `IncomeParams` (tax rate)

**What it does:**
- Waterfall: Declared income (tier 1) → Payslip (tier 2) → Other (tier 3)
- Applies verification haircut per tier: 100%, 95%, 90%
- Statutory deductions (simplified: 18% tax rate)

### 2. Affordability (`modules/affordability/`)
Calculates discretionary income and lending capacity.

**Interface:**
- **Inputs**: `net_monthly_income`, `declared_living_expenses`, `existing_obligations`, `dependants_count`, `instalment`
- **Outputs**: `discretionary_income`, `max_affordable_instalment`, `affordability_verdict_code`
- **Params**: `AffordabilityParams` (buffer, thresholds, expense floor)

**What it does:**
- Enforces statutory minimum living expense floor
- Aggregates existing debt obligations
- Calculates discretionary income (net - expenses - obligations)
- Applies affordability buffer to determine maximum commitment
- Issues verdict: 1 (pass), 2 (marginal), 3 (fail)

### 3. Fees (`modules/fees/`)
Calculates statutory fees and insurance premiums.

**Interface:**
- **Inputs**: `offered_amount`, `applicant_age_years`, `employment_type_code`, `product_code`
- **Outputs**: `initiation_fee`, `monthly_service_fee`, `credit_life_premium`
- **Params**: `FeesParams` (caps, rates, age multipliers)

**What it does:**
- Initiation fee with statutory cap (R2000)
- Monthly service fee with statutory cap (R150)
- Credit life insurance with age-based adjustments

## Assembling Products

### Flex Loan
```python
from pipeline import pipeline, SharedParams

result = pipeline.score(
    {
        "declared_income": 50000.0,
        "payslip_income": None,
        "employment_type_code": 1,
        "declared_living_expenses": 8000.0,
        "dependants_count": 1,
        ...
    },
    params={
        "income": {},
        "affordability": {},
        "fees": {},
        "term_decision": {},
        "offered_amount": {},
    },
    shared={
        "prime_rate": 8.5,
        "product_margin": 2.5,
    }
)

# result contains: net_monthly_income, discretionary_income,
# affordability_verdict_code, fees, offered_amount, rate, total_cost_of_credit
```

### Using with Vocabulary Mapping
For a project with different column names:

```python
from vocabulary import flex_loan_vocabulary

# Map at composition time
assessed = (Income | Affordability | Fees).with_vocabulary(flex_loan_vocabulary)
```

Or relabel on-the-fly:

```python
# For specific instances
AffordCurrent = Affordability.relabel(
    reads={"net_income": "current_net_income"}
)
```

## Running Tests

### Status

Tests are **written but cannot currently run** due to a framework-level import issue documented in NOTES.md.

The issue is in the framework, not in this implementation:
- The decider2 package has a namespace package conflict
- Two lines in `/decider2/decider2/__init__.py` need to use relative imports
- This blocks all user code from importing

**When this is fixed**, tests will run:

```bash
cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine
python -m pytest decider2/evaluation/00-shared-credit-core/test_credit_core.py -v
```

Expected: All tests pass, including mode equivalence checks.

## Serving

When the import issue is fixed:

```bash
cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/evaluation/00-shared-credit-core

# Start server
python -m decider2 serve serve_flex_loan.py --port 8101

# In another terminal
curl http://localhost:8101/ping
# {"status": "ok"}

# POST a record for assessment
curl -X POST http://localhost:8101/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "declared_income": 50000.0,
    ...
  }'
```

## Design Decisions

1. **Simplified implementations** - All arithmetic is basic math, no real calibration/optimization
2. **Subset of 22 capabilities** - Implemented 3 core ones (income, affordability, fees) sufficient to demonstrate module reuse
3. **Single product pipeline** - Flex Loan primary, Access Facility shows reuse
4. **Inline params** - Used doc 03 §4.4's `param()` in step signatures for simple cases
5. **No adjustments** - Core capability (doc 00 §6.22) exists in spec but deferred - requires separate mechanism not yet documented

## Reference

- **BRIEF.md** - Evaluation requirements
- **00-shared-credit-core-library.md** - Full specification (22 capabilities)
- **03-authoring-api.md** - Framework authoring guide
- **07-project-structure.md** - Recommended project layout

## Author Notes

See **NOTES.md** for:
- Detailed assessment of what was built
- Framework issues encountered (with fixes suggested)
- Design decisions requiring guesses
- Unexpressible mechanisms
- Time spent
