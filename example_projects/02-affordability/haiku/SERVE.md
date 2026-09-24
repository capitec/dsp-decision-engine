# Serving the affordability assessment

Project 02 implements the seven-stage affordability assessment on top of project 00's
five affordability units. It supports single and joint applicants, four assessment modes,
and three answer shapes.

## Build

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/02-affordability
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/02-affordability/configs

uv run --project . decider build
```

## Serve

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/02-affordability
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/02-affordability/configs

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
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:/path/to/scratch/haiku/02-affordability:$PYTHONPATH

uv run --project . pytest /path/to/scratch/haiku/02-affordability/tests/ -v
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

## Assessment modes

The assessment_mode_code parameter selects the evidence rules and output shape:
- `1`: New application (full evidence waterfall, all outputs)
- `2`: Limit increase (degraded evidence, max affordable only)
- `3`: Arrangement (stressed mode, different buffer grid)
- `4`: Scenario (hypothetical, inherited evidence)

## Answer shapes

Depending on the provided inputs:
- **(a) Verdict against known instalment**: Supply `proposed_instalment` for pass/fail/marginal verdict
- **(b) Maximum affordable**: Omit `proposed_instalment` to get `max_affordable_instalment` only
- **(c) Iterative solve**: Project 03 consumes the assessment for circular solve (amount↔rate↔affordability)
