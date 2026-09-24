# Building and serving project 10 (retail credit end-to-end)

## Prerequisites

Ensure project 00 (shared credit core library) is in the Python path:

```bash
export PYTHONPATH=/tmp/claude-1000/.../scratchpad/examples/haiku/00-shared-credit-core:$PYTHONPATH
```

## Build

From the project root (the REPO directory):

```bash
export PYTHONPATH=$SCRATCH_HAIKU/00-shared-credit-core:$PYTHONPATH
cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine

uv run --project . decider build \
  --code-path /tmp/claude-1000/.../scratchpad/examples/haiku/10-retail-credit-e2e \
  --pipeline pipeline:build \
  --config /tmp/claude-1000/.../scratchpad/examples/haiku/10-retail-credit-e2e/configs/default/params.json
```

## Serve (test mode)

Run the pipeline against the sample request:

```bash
export PYTHONPATH=$SCRATCH_HAIKU/00-shared-credit-core:$PYTHONPATH
cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine

uv run --project . decider serve \
  --code-path /tmp/claude-1000/.../scratchpad/examples/haiku/10-retail-credit-e2e \
  --pipeline pipeline:build \
  --config /tmp/claude-1000/.../scratchpad/examples/haiku/10-retail-credit-e2e/configs/default/params.json \
  --input-file /tmp/claude-1000/.../scratchpad/examples/haiku/10-retail-credit-e2e/sample_request.json
```

## Test

From the project root:

```bash
export PYTHONPATH=$SCRATCH_HAIKU/00-shared-credit-core:$PYTHONPATH
cd /home/sholto/Documents/Workspace/capitec/dsp-decision-engine

uv run --project . pytest /tmp/claude-1000/.../scratchpad/examples/haiku/10-retail-credit-e2e/tests/ -v
```

## Environment variables

Set before running commands:

```bash
export DECIDER_API__CODE_PATH=/tmp/claude-1000/.../scratchpad/examples/haiku/10-retail-credit-e2e
export DECIDER_API__PIPELINE=pipeline:build
export DECIDER_CONFIG__BASEPATH=/tmp/claude-1000/.../scratchpad/examples/haiku/10-retail-credit-e2e/configs/default
export PYTHONPATH=/tmp/claude-1000/.../scratchpad/examples/haiku/00-shared-credit-core:$PYTHONPATH
```

## Notes

- Entry point 1 (new credit application) is fully implemented for product 10 (Flex Loan).
- Other entry points (2-8) are stubbed for routing.
- Product 10: R2,000–R500,000, 6–84 months. Entry point 1 volume ~31,900/day.
- Loop L1 re-runs affordability (P10) when it fails and client is consolidation-eligible (§5.23).
- Shared intermediates registry in `flow_state.DecisionState` tracks 41 values across phases.
- Two simultaneous versions of `existing_obligations` supported for consolidation scenarios (§5.21.1).
