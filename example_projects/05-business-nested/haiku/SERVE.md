# Build and serve project 05 (business credit with nested entities)

## Build

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:/path/to/scratch/haiku/05-business-nested:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/05-business-nested
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/05-business-nested/configs

uv run --project . decider build
```

## Serve

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:/path/to/scratch/haiku/05-business-nested:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/05-business-nested
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/05-business-nested/configs

uv run --project . decider serve
```

## Test with sample request

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Run pytest

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:/path/to/scratch/haiku/05-business-nested:$PYTHONPATH

uv run --project . pytest /path/to/scratch/haiku/05-business-nested/tests/ -v
```

## Configuration version

Current config version: `0.0.0`
Configuration file: `configs/0.0.0/params.json`

To create a new version, copy the configs directory and increment the version:
```bash
mkdir -p configs/0.0.1
cp configs/0.0.0/params.json configs/0.0.1/params.json
# Edit configs/0.0.1/params.json as needed
```

## Assessment inputs

The pipeline accepts:
- Application data: application_id, decision_date, product_code, requested_amount, etc.
- Entity list: 1..40 entities with relationship type, ownership %, criticality inputs
- Adverse events: 0..60 events per entity with type, amount, satisfaction status, dispute flag

## Assessment outputs

Per business:
- `business_verdict_code`: 1 (clear), 2 (refer), 3 (decline)
- `business_grade`: 1..12 (lower is worse)
- `business_pd`: Probability of default (0..1)
- `declined_entity_key`, `declined_event_id`, `declined_rule`: Chain of attribution if declined

Per entity:
- Criticality class (Critical/Significant/Peripheral)
- Adverse verdict (Clear/Minor/Material/Disqualifying)
- Binding rule and contributing event IDs

Per event:
- Severity (Immaterial/Minor/Material/Disqualifying)
- Classification rule
- Thresholds used (material, disqualifying, and overlaid values)
