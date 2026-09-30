# Debug bridge relocation

The debug bridge moved from its own package (`tools/decider-bridge`) into the
`decider` package, and its helpers were split along the session/adapter
boundary. This documents where each piece landed and the guarantees the move
preserved.

## The boundary

`decider.engine.debug.Session` is the session. It stays in core and is the only
thing that knows how to run a pipeline one checkpoint at a time.

`decider.debug_bridge` is the editor-facing adapter over that session. It owns
the stdio JSON-lines protocol, the `--debugpy PORT` attach flag, and process
launch; it is not a second session implementation.

## Where each piece landed

| Before | After | Role |
|---|---|---|
| `decider_bridge/bridge.py` | `decider/debug_bridge/bridge.py` | transport: JSON-lines loop, debugpy attach, `Bridge` |
| `decider_bridge/loading.py` | `decider/debug_bridge/loading.py` | process launch: import a pipeline's files |
| `decider_bridge/timeline.py` | `decider/debug_bridge/timeline.py` | adapter-side observation of a session |
| `decider_bridge/runs.py` | `decider/debug_bridge/runs.py` | per-record views + input overrides (the run-and-collect primitives moved out) |
| `decider_bridge/__main__.py` | `decider/debug_bridge/__main__.py` | entry point (`python -m decider.debug_bridge`) |
| `decider_bridge/controls.py` | `decider/engine/debug/controls.py` | session steering (forces, value breakpoints) |
| `decider_bridge/lineage.py` | `decider/engine/debug/lineage.py` | runtime lineage query over a session's state |
| `decider_bridge/describing.py` | `decider/contract/describing.py` | flow description: pipelines, IR as JSON, source locations |
| `decider_bridge/forks.py` | `decider/experiments/forks.py` | experiment primitives: `fork`, `sweep`, `checkpoint_key`, `merge` |
| `decider_bridge/runs.py` (`collect`, `steer`, `trace`) | `decider/experiments/runs.py` | experiment primitives: run-and-collect |

`Controls` landed in `engine.debug` because it steers a session (forces and
value breakpoints attach to a `Session`); it happens to take the described IR
dict, which `contract.describing` produces, but it imports nothing from it.
The `experiments` package is the seam task 09 builds its full experiment
interface on; it is not that interface yet.

## Transports and consumers preserved

Three transports over the same session, both consumers:

- stdio JSON-lines (`Bridge.serve`, fd 3 for replies) — VS Code.
- debugpy attach (`--debugpy PORT`) — VS Code "Debug this step in Python".
- starlette websocket (`decider.serving.session_ws.session_app`) — unchanged,
  still core.

VS Code (`tools/vscode-decider`) now runs `python -m decider.debug_bridge`; the
bridge is inside the installed `decider`, so the extension no longer adds a
bridge package to `PYTHONPATH`. JupyterLab (`tools/jupyterlab-decider`) imports
`decider.debug_bridge.bridge.Bridge` in the kernel and no longer depends on a
separate `decider-bridge` distribution.

## Cross-mode parity

A bridge session runs in interpreted mode. The guarantee that a debugged run
matches a compiled one is `decider.testing.equivalence.assert_equivalent`:
`run()`, `score()` and a resumed session agree across interpreted, stepped and
fused modes, exactly (NaN equals NaN, nothing else is approximate). Money is
int64 cents; running totals accumulate in float64. `tests/debug_bridge/test_parity.py`
pins this for a flow debugged through the bridge, and `tests/testing/test_testing.py`
covers the contract itself.

## Compile/fallback rule

Compile-time failures and per-step compiled modes refuse to fall back to Python
per step (a warning, or `Engine(strict_compile=True)` raises). Runtime errors
always propagate. The bridge reports a failing step in its reply and keeps the
trace up to the failing step; it does not silently fall back.
