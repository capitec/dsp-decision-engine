# Project 11 Build and Serve

## Environment

Set up `PYTHONPATH` to include consumed projects:

```bash
export REPO=/home/sholto/Documents/Workspace/capitec/dsp-decision-engine
export SCRATCH=/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples

export PYTHONPATH="${SCRATCH}/haiku/11-business-credit-e2e:${SCRATCH}/haiku/05-business-nested:${SCRATCH}/haiku/06-consolidation:${SCRATCH}/haiku/02-affordability:${SCRATCH}/haiku/00-shared-credit-core:${REPO}"
```

## Build

```bash
cd $REPO

PYTHONPATH="$PYTHONPATH" \
  uv run --project . decider build \
  --code-path ${SCRATCH}/haiku/11-business-credit-e2e \
  --pipeline business_credit_e2e.pipeline:build \
  --config-path ${SCRATCH}/haiku/11-business-credit-e2e/configs/0.0.0
```

Expected output:
```
✓ Pipeline built successfully
✓ Schema validated
✓ Params loaded: 3 entry points
```

## Test Sample Request

```bash
cd $REPO

PYTHONPATH="$PYTHONPATH" \
  uv run --project . python -c "
import json
from business_credit_e2e.pipeline import ep1_origination

with open('${SCRATCH}/haiku/11-business-credit-e2e/sample_request.json') as f:
    request = json.load(f)

result = ep1_origination(
    application_id=request['parameters']['application_id'],
    product_code=request['parameters']['product_code'],
    amount=request['parameters']['amount'],
    term_months=request['parameters']['term_months'],
)

print(json.dumps(result, indent=2, default=str))
"
```

Expected output:
```json
{
  "application_id": 100001,
  "decision_date": "2026-01-...",
  "business_grade": 6,
  "master_scale_version": "v1.0",
  "business_verdict_code": 1,
  "declined_entity_key": null,
  "declined_rule": null,
  "product_code": 50,
  "amount": 500000
}
```

## Run Tests

```bash
cd $REPO

PYTHONPATH="$PYTHONPATH" \
  uv run --project . pytest \
  ${SCRATCH}/haiku/11-business-credit-e2e/tests/ -v
```

Expected: 7+ tests pass

## Environment Variables for Handler

None required in this context (no external APIs).

## Decider Environment Variables

```bash
export DECIDER_API__CODE_PATH=${SCRATCH}/haiku/11-business-credit-e2e
export DECIDER_API__PIPELINE=business_credit_e2e.pipeline:build
export DECIDER_CONFIG__BASEPATH=${SCRATCH}/haiku/11-business-credit-e2e/configs/0.0.0
```
