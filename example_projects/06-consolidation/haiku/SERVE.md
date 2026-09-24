# Project 06: Consolidation and Restructure — Serving Instructions

## Build

Set up environment variables for consumed projects and build:

```bash
export PYTHONPATH=/path/to/00-shared-credit-core:/path/to/02-affordability:/path/to/03-loan-granting-pricing:/path/to/06-consolidation

cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine

export DECIDER_API__CODE_PATH=/tmp/claude-1000/.../scratchpad/examples/haiku/06-consolidation
export DECIDER_API__PIPELINE=pipeline.build
export DECIDER_CONFIG__BASEPATH=/tmp/claude-1000/.../scratchpad/examples/haiku/06-consolidation/configs

uv run --project . decider build
```

## Serve (development)

Run the inference handler against the sample request:

```bash
export PYTHONPATH=/path/to/00:/path/to/02:/path/to/03:/path/to/06

cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine

uv run --project . python -c "
from consolidation import build
import json
with open('/path/to/06/sample_request.json') as f:
    req = json.load(f)
pipeline = build()
result = pipeline(req)
import pprint
pprint.pprint(result)
"
```

## Test

Run the test suite:

```bash
export PYTHONPATH=/path/to/00:/path/to/02:/path/to/03:/path/to/06

cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine

uv run --project . pytest /path/to/06/tests/test_consolidation.py -v
```

## Expected output

Sample response structure:

```json
{
  "decision_id": "uuid-here",
  "client_id": 12345,
  "decision_date": "2024-09-24",
  "is_eligible": true,
  "settleable_count": 3,
  "baseline_instalment": 1250.0,
  "baseline_weighted_rate": 16.2,
  "scenarios_evaluated": 48,
  "scenarios_budget": 400,
  "budget_exhausted": false,
  "winner": {
    "scenario_id": 5,
    "settlement_set": [1001, 1003],
    "product_code": 11,
    "term_months": 60,
    "instalment_relief": 650.0,
    "total_cost_delta": 8500.0
  },
  "top_3_count": 3,
  "objective_id": 2
}
```

## Paths

- **Pipeline:** `consolidation.py` (main logic)
- **Config:** `configs/v1/params.json` (parameters and thresholds)
- **Tests:** `tests/test_consolidation.py` (unit and integration)
- **Sample:** `sample_request.json` (realistic request)

## Notes

- The pipeline generates scenarios deterministically within the budget.
- Scenarios are evaluated in order of H1-H8 heuristic preferences.
- Budget exhaustion is recorded: if `budget_exhausted=true`, the search terminated due to candidate count, not scenario count.
- The winner is the viable scenario with the lowest instalment (OBJ-02, configurable).
- Rejection reason codes are recorded per scenario for explainability.
