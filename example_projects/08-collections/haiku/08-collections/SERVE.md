# Serving the collections treatment assignment

Project 08 assigns daily treatment to delinquent accounts based on account state,
suspensions, collections score, treatment matrix, and capacity constraints.

## Build

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/08-collections:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/08-collections
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/08-collections/configs

uv run --project . decider build
```

## Serve

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/08-collections:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/08-collections
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/08-collections/configs

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
export PYTHONPATH=/path/to/scratch/haiku/08-collections:$PYTHONPATH

uv run --project . pytest /path/to/scratch/haiku/08-collections/tests/ -v
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

## Key decision flows

1. **Account state assembly (§5.1):** Computes arrears bucket, balance band from current delinquency
2. **Suspensions (§5.2):** Evaluates 20 suspension codes; blocks treatments if applicable
3. **Risk assessment (§5.3):** Collections score from 28 characteristics (simplified to ~10 key ones), bands into 1-6
4. **Treatment matrix (§5.4):** 5,376 cells (8 buckets × 6 collections bands × 7 balance bands × 4 contact bands × 4 products)
5. **Escalation path (§5.5):** Tracks episode position; escalates on non-engagement
6. **Arrangements (§5.6):** Flags for affordability assessment via project 02 (stubbed here)
7. **Capacity allocation (§5.9):** Ranks accounts in 9 pools; reports non-selection reasons

## Evidence contract (§5.15)

Every decision records:
- `decision_id`, `treatment_instance_id`, `decision_date`
- All bands, scores (adjusted + unadjusted)
- `matrix_version`, `matrix_cell_id` for audit
- `episode_id`, `path_position` for history
- All active suspensions with codes
- Pool allocation or reason for non-selection
- `adjustment_set_version`, `applied_overlay_ids` for governance
