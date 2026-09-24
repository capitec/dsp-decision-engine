# Project 03: Unsecured Loan Granting and Pricing — Build and Serve

## Build

Verify that the decider pipeline builds successfully:

```bash
cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine

export PYTHONPATH=/tmp/claude-1000/.../haiku/00-shared-credit-core:/tmp/claude-1000/.../haiku/02-affordability:/tmp/claude-1000/.../haiku/03-loan-granting-pricing

uv run --project . decider build \
  --code-path /tmp/claude-1000/.../haiku/03-loan-granting-pricing \
  --pipeline flex_loan_granting \
  --config /tmp/claude-1000/.../haiku/03-loan-granting-pricing/configs/latest/params.json
```

Expected output: Schema validation passes, pipeline composes, parameters loaded.

## Serve

Start the inference server:

```bash
export DECIDER_API__CODE_PATH=/tmp/claude-1000/.../haiku/03-loan-granting-pricing
export DECIDER_API__PIPELINE=flex_loan_granting
export DECIDER_CONFIG__BASEPATH=/tmp/claude-1000/.../haiku/03-loan-granting-pricing/configs/latest

uv run --project . decider serve \
  --bind 0.0.0.0 \
  --port 8000
```

## Score Sample Request

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d @/tmp/claude-1000/.../haiku/03-loan-granting-pricing/sample_request.json
```

Expected response structure:
```json
{
  "decision_id": "uuid",
  "outcome": "approve | refer | decline",
  "offers": [
    {
      "term_months": 60,
      "offered_amount": 95000,
      "instalment": 2860.25,
      "is_recommended": true
    }
  ],
  "eligibility_verdicts": [...],
  "cap_chains": {...},
  "table_versions": {...}
}
```

## Run Tests

```bash
export PYTHONPATH=/tmp/claude-1000/.../haiku/00-shared-credit-core:/tmp/claude-1000/.../haiku/02-affordability:/tmp/claude-1000/.../haiku/03-loan-granting-pricing

uv run --project /home/sholto/Documents/Workspace/capitec/dsp-decision-engine \
  pytest /tmp/claude-1000/.../haiku/03-loan-granting-pricing/tests/test_granting.py -v
```

## Verify Against Spec

1. **Spec §5.8 non-monotone band-edge case** (Client W at grade 9, 60 months, max R1 560):
   ```bash
   python -m granting.test_nonmonotone
   ```
   Expected: R50 000 found as correct answer (not R48 500)

2. **Spec §5.10 AC2: Independent re-check of offers**:
   - Every offer in generated set must re-validate from scratch
   - Zero failures on regression test of 250k+ applications

3. **Spec §10 AC11: Real-time / batch identity**:
   - Run `python batch_mode.py` with 100k records
   - Verify identical output to real-time path

## Determinism Check

Replay the same request 5 times and verify bit-identical outputs:

```bash
python -c "
import json
from pipeline import flex_loan_granting
with open('sample_request.json') as f:
    req = json.load(f)
result1 = flex_loan_granting(req)
result2 = flex_loan_granting(req)
assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True), 'Determinism failed'
print('✓ Determinism verified')
"
```

## Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `DECIDER_API__CODE_PATH` | Path to this project | Pipeline code location |
| `DECIDER_API__PIPELINE` | `flex_loan_granting` | Pipeline entry point |
| `DECIDER_CONFIG__BASEPATH` | Path to `configs/latest/` | Parameter and config location |
| `PYTHONPATH` | `00:02:03` | Dependency resolution |
