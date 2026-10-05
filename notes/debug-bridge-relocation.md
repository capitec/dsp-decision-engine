# Debug bridge relocation

The debug bridge moved from its own package (`tools/decider-bridge`) into the
`decider` package, and its helpers were split along the session/adapter
boundary. This documents where each piece landed and the guarantees the move
preserved.

The `debug_bridge/` package is planned to move to `_bridge/`; see
`notes/pythonapi-redesign/directory-structure.md`.

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
| `decider_bridge/runs.py` | `decider/debug_bridge/runs.py` | per-record views + input overrides (the run-and-collect primitives moved to `replay.py`) |
| `decider_bridge/__main__.py` | `decider/debug_bridge/__main__.py` | entry point (`python -m decider.debug_bridge`) |
| `decider_bridge/controls.py` | `decider/engine/debug/controls.py` | session steering (forces, value breakpoints) |
| `decider_bridge/lineage.py` | `decider/engine/debug/lineage.py` | runtime lineage query over a session's state |
| `decider_bridge/describing.py` | `decider/contract/describing.py` | flow description: pipelines, IR as JSON, source locations |
| `decider_bridge/forks.py` | `decider/debug_bridge/forks.py` | fork primitives: `fork`, `sweep`, `checkpoint_key` (`merge` is `decider.engine.params.merge`, imported) |
| `decider_bridge/runs.py` (`collect`, `steer`, `trace`) | `decider/debug_bridge/replay.py` | run-and-collect |

`debug_bridge/` also holds modules that were not in the old package:
`datasets.py`, `draft.py`, `experiment.py`, `rows.py` and `run.py`.

`Controls` landed in `engine.debug` because it steers a session (forces and
value breakpoints attach to a `Session`); it happens to take the described IR
dict, which `contract.describing` produces, but it imports nothing from it.
`forks` and `replay` stay in `debug_bridge`; `decider.experiments` is a
separate package (experiment definitions, runner, reports) and does not hold
them.

## Transports and consumers preserved

One transport over the session, plus the debugpy flag:

- stdio JSON-lines (`Bridge.serve`, replies on the fd given by `--fd`, 3 in
  the extension) — VS Code.
- `--debugpy PORT` on the stdio bridge also listens for a debugpy attach — VS
  Code "Debug this step in Python".

VS Code (`tools/vscode-decider`) runs `python -m decider.debug_bridge`; the
bridge is inside the installed `decider`, so the extension does not set
`PYTHONPATH` or add a bridge package. There is no JupyterLab extension in
`tools/` any more.

## Cross-mode parity

A bridge session runs in interpreted mode. The guarantee that a debugged run
matches a compiled one is `decider.testing.equivalence.assert_equivalent`:
`run()`, `score()` and a resumed session agree across interpreted, stepped and
fused modes, exactly (NaN equals NaN, nothing else is approximate). Money is
int64 cents; running totals accumulate in float64. `tests/debug_bridge/test_parity.py`
pins this for a flow debugged through the bridge, and `tests/testing/test_testing.py`
covers the contract itself.

## Compile/fallback rule

In `stepped` and `fused` modes, a compile-time failure or a step the compiled
mode refuses falls back to Python for that step with a warning;
`Engine(strict_compile=True)` raises instead. Runtime errors always propagate.
The bridge reports a failing step in its reply (`error`), and `replay.trace`
keeps what ran up to the failing step.
