# Serving the shared credit core library

## Build

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/00-shared-credit-core
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/00-shared-credit-core/configs

uv run --project . decider build
```

## Serve

```bash
cd /path/to/repo
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:$PYTHONPATH
export DECIDER_API__CODE_PATH=/path/to/scratch/haiku/00-shared-credit-core
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/path/to/scratch/haiku/00-shared-credit-core/configs

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
export PYTHONPATH=/path/to/scratch/haiku/00-shared-credit-core:$PYTHONPATH

uv run --project . pytest /path/to/scratch/haiku/00-shared-credit-core/tests/ -v
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
