# The debug session

`engine/debug/` choices that stepped/fused sessions, arm/iteration stepping
and the websocket adapter build on.

- **Built only on the `Runner` protocol.** The session drives
  `runner.iterate(plan, state, params)` one checkpoint at a time and assumes
  nothing about which checkpoints a runner yields. A runner with fewer
  checkpoints (fused: kernel boundaries only) just pauses less often; a
  prefix breakpoint catches the first checkpoint under it that exists.
- **Breakpoints fire on entry.** A path or prefix matches a `before`
  checkpoint whose previous checkpoint was outside that subtree, so `"term"`
  stops once when the run enters `term`, not at every node inside it (a loop
  body is entered once per iteration). Predicates see every checkpoint.
- **Positions inside row nodes.** The runner's `visit` hook can't yield, so
  the session can't pause mid-row-node. It counts visits per locator while
  the row node runs and emits one `NodeVisited(origin with locator, rows)`
  per locator at the node's `after`; `path#locator` breakpoints pause there.
  A runner exposing a `visit` attribute gets the hook (the session copies
  the runner, so the bound `Executable` is untouched).
- **`step()`** from `before X` runs to `after X` unless X is a sequence
  (sequences are entered); anywhere else it moves one checkpoint.
  `step_into()` always moves one checkpoint, which enters the taken arms and
  each loop iteration.
- **`set` overwrites every version of the name written so far, then records
  an `override@<path>` version.** Versions are bound at resolve time, so a
  later node reads a specific version id, not "the latest"; a branch may
  still read the prior value and a loop its carry. Overwriting only the
  latest would leave those stale. The cost: `name@producer` for earlier
  producers now shows the override; the `Overridden` event keeps the
  previous summary for the audit trail.
- **`rewind(path)` replays** a fresh run silently up to `before path`, then
  puts back the current values of everything written upstream (overrides
  included) and every input. That rebuilds the runner's internal scope (arm
  row subsets, loop state) without a runner API for jumping. Routing inside
  a branch is re-derived during the replay from the original values.
- **Errors end the generator.** An exception is logged as `Error` and
  re-raised; the session then only accepts `rewind`.
- **JSON only at the edge.** Events and commands are frozen dataclasses with
  a `kind` literal; `wire.py` holds the pydantic `TypeAdapter`s. A command's
  `kind` is the `Session` method name, so `Session.apply` dispatches by name.
- **Websocket adapter (`serving/session_ws.py`).** One session per
  connection. A reader task handles `pause` inline (the session's flag is
  thread-safe) and queues everything else; one loop runs queued commands in a
  worker thread and value requests in order, so state is never read while the
  worker writes it. Command failures reply `{"kind": "rejected"}`, not
  `"error"`, which is already the run-error event. Tests drive the ASGI app
  directly because starlette's `TestClient` needs httpx, which no extra
  includes; `starlette` is in the dev group.
