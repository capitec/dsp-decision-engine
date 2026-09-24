import sys
sys.path.insert(0, '/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/09-governance-replay')
from pipeline import build
from decider import Engine
from datetime import date
import json

try:
    # Try to build the harness
    harness = build()
    print("Pipeline built successfully")
    print("Type:", type(harness))
    
    # Try to create an engine
    engine = Engine(pipeline=harness)
    print("Engine created successfully")
    
    # Try a simple run
    result = engine.run(
        step='governance_harness',
        operation='replay',
        decision_id='test-001',
        evidence={'outputs': {'outcome_code': 'approved'}, 'inputs_as_received': {}},
        flow_type='01',
        audience='analyst',
    )
    print("Run completed successfully")
    print(json.dumps(result, indent=2, default=str))
except Exception as e:
    import traceback
    print("ERROR:", e)
    print(traceback.format_exc())
