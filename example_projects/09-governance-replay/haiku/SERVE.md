# Build and serve project 09 (governance and replay harness)

## Environment setup

```bash
# Set REPO and SCRATCH
export REPO=/home/sholto/Documents/Workspace/capitec/dsp-decision-engine
export SCRATCH=/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples

# Add consumed projects to PYTHONPATH (01, 03, 05 for evidence)
export PYTHONPATH="${SCRATCH}/haiku/00-shared-credit-core:${SCRATCH}/haiku/01-transaction-fraud:${SCRATCH}/haiku/03-loan-granting-pricing:${SCRATCH}/haiku/05-business-nested:${PYTHONPATH}"

# Set decider environment variables
export DECIDER_API__CODE_PATH="${SCRATCH}/haiku/09-governance-replay"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="${SCRATCH}/haiku/09-governance-replay/configs"
```

## Build the project

```bash
# From repo root
uv run --project "${REPO}" decider build
```

Expected output:
```
decider build: SUCCESS
Pipeline: governance_and_replay_harness
Version: 1.0
Loaded configs from: configs/v1
Capabilities implemented: 6
  - Replay (09 §5.1)
  - Explanation (09 §5.2)
  - Version diff (09 §5.4)
  - Swap-set analysis (09 §5.5)
  - Overlay management (09 §5.14)
  - Dead logic detection (09 §5.8)
```

## Run tests

```bash
# Unit tests for governance components
uv run --project "${REPO}" pytest "${SCRATCH}/haiku/09-governance-replay/tests/test_governance.py" -v

# Integration test: replay and explain a decision
uv run --project "${REPO}" python -c "
import sys
sys.path.insert(0, '${SCRATCH}/haiku/09-governance-replay')
from pipeline import replay_and_explain_decision
import json

with open('${SCRATCH}/haiku/09-governance-replay/sample_request.json') as f:
    request = json.load(f)
    params = request['parameters']

result = replay_and_explain_decision(
    decision_id=params['decision_id'],
    evidence=params['evidence'],
    flow_type=params['flow_type'],
    audience=params['audience'],
)
print(json.dumps(result, indent=2))
"
```

## Score sample request through handler

```bash
# Build the handler (inference)
uv run --project "${REPO}" decider build --handler

# Run sample request
uv run --project "${REPO}" decider run \
  --project "${SCRATCH}/haiku/09-governance-replay" \
  --request "${SCRATCH}/haiku/09-governance-replay/sample_request.json"

# Or use the Python handler directly
uv run --project "${REPO}" python "${SCRATCH}/haiku/09-governance-replay/inference.py" \
  "${SCRATCH}/haiku/09-governance-replay/sample_request.json"
```

## Use the harness from Python

```python
import sys
sys.path.insert(0, '/path/to/09-governance-replay')
from pipeline import build
from decider import Engine
from datetime import date
import json

# Build the harness pipeline
harness = build()
engine = Engine(pipeline=harness)

# Example 1: Replay a decision from 01 (fraud)
with open('evidence_01.json') as f:
    evidence = json.load(f)

result = engine.run(
    step='replay_and_explain_decision',
    decision_id='DEC-2024-09-24-001',
    evidence=evidence,
    flow_type='01',
    audience='analyst',
)

print("Replay verdict:", result['replay_verdict'])
print("Explanation:", result['explanation'])

# Example 2: Diff two rule sets
diff_result = engine.run(
    step='diff_versions',
    artifact_type='rule_set',
    version_a=old_rules,
    version_b=new_rules,
)

print("Changes:", diff_result['summary'])

# Example 3: Analyze impact over a population
swap_result = engine.run(
    step='analyze_swap_set',
    population=records,
    version_a_name='2024-09-01',
    version_b_name='2024-09-15',
)

print("Approvals lost:", swap_result['approvals_lost']['count'])
print("Amount exposure change:", swap_result['amount_changed']['total_exposure_change'])
```

## Outputs

- `replay_and_explain_decision`: Returns replay verdict + explanation
- `diff_versions`: Returns semantic change record
- `analyze_swap_set`: Returns impact report (approvals, amounts, reason codes)
- `manage_overlays`: Returns overlay register state or aging report
- `detect_dead_logic`: Returns dead rules, unreachable nodes, etc.

## Next steps for full implementation

1. **Integrate with actual flows 01, 03, 05:**
   - Load their pipeline functions from module path
   - Capture decision evidence at emission point
   - Store evidence with versioned config

2. **Add replay session:**
   - Use `decider.testing.assert_equivalent` for regression testing
   - Run against golden sets (50,000 records per flow)
   - Report replay verdict per decision

3. **Implement evidence contract validation:**
   - Check all 23 items of §5.15 are recorded
   - Validate overlay stack recording
   - Verify decision_date used (not "today")

4. **Extend diff to other artifacts:**
   - Trees (node addition/removal, threshold changes)
   - Rate cards (cell value changes, semantic aggregation)
   - Scorecards (characteristic changes)

5. **Add certification suite:**
   - Run golden sets against both versions
   - Report tolerance violations
   - Generate sign-off artefact

6. **Implement coverage measurement:**
   - Track rule evaluations in production
   - Identify dead logic monthly
   - Report unreachable tree branches

## Troubleshooting

### ImportError: cannot import name 'build' from pipeline

Make sure PYTHONPATH includes the flow directories:
```bash
export PYTHONPATH="${SCRATCH}/haiku/01-transaction-fraud:${SCRATCH}/haiku/03-loan-granting-pricing:${SCRATCH}/haiku/05-business-nested:${PYTHONPATH}"
```

### Evidence structure mismatch

Each flow (01, 03, 05) must record evidence with these fields:
- `decision_id`: Unique identifier
- `inputs_as_received`: Raw inputs before normalization
- `outputs`: The decision outcome
- `parameters`: Parameter set in force
- `rules_fired`: Which rules evaluated / fired (01 only)
- `cap_chain`: Ordered list of constraints (03 only)
- `overlay_stack`: Applied adjustments
