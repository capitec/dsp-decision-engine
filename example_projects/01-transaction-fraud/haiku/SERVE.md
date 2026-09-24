# Build and serve project 01

## Prerequisites

Set environment for shared core library:
```bash
export PYTHONPATH="/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/00-shared-credit-core:$PYTHONPATH"
```

Working directory (from where commands are run):
```bash
cd /tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/01-transaction-fraud
```

## Build

Validate pipeline and schema:
```bash
PYTHONPATH="/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/00-shared-credit-core" \
  uv run --project /home/sholto/Documents/Workspace/capitec/dsp-decision-engine \
  decider build \
  --code-path /tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/01-transaction-fraud \
  --pipeline pipeline.build \
  --config /tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/01-transaction-fraud/configs/v1/params.json
```

## Serve

Start server:
```bash
PYTHONPATH="/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/00-shared-credit-core" \
  DECIDER_API__CODE_PATH=/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/01-transaction-fraud \
  DECIDER_API__PIPELINE=pipeline.build \
  DECIDER_CONFIG__BASEPATH=/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/01-transaction-fraud/configs \
  uv run --project /home/sholto/Documents/Workspace/capitec/dsp-decision-engine \
  decider serve
```

## Test

Run unit tests:
```bash
PYTHONPATH="/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/00-shared-credit-core:/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/01-transaction-fraud" \
  uv run --project /home/sholto/Documents/Workspace/capitec/dsp-decision-engine \
  pytest /tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/01-transaction-fraud/tests -v
```

## Sample request

Send test event:
```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Environment variables

| Variable | Value |
|---|---|
| `DECIDER_API__CODE_PATH` | Path to project directory |
| `DECIDER_API__PIPELINE` | `pipeline.build` (the function) |
| `DECIDER_CONFIG__BASEPATH` | Path to configs/ directory |
| `PYTHONPATH` | Include 00-shared-credit-core |
