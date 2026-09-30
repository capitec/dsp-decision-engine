# Migrations and deliberate behaviour changes

The redesign changes where the debugger transport lives and adds optional
identities; everything else preserves existing behaviour. Each migration below
is tested; the deliberate change is stated plainly.

## 1. Bridge packaging: `tools/decider-bridge` → `decider.debug_bridge`

The editor/debug transport moved from a separate `tools/decider-bridge`
distribution into the main package as `decider.debug_bridge`
(`tests/debug_bridge/test_relocation.py`). The session itself
(`decider.engine.debug.Session`) did not move; the bridge is an adapter over it.

- **Existing extension users:** the `.vsix` never carried the bridge. After the
  move, installing `decider` into the project environment provides the bridge;
  the extension launches `python -m decider.debug_bridge --fd 3` (and
  `--debugpy PORT` for attach) and no longer adds a bridge package to
  `PYTHONPATH`. The `decider.python` setting is unchanged.
- **Existing launch configurations:** the command line is unchanged —
  `python -m decider.debug_bridge` with `--fd`/`--debugpy` — only the import
  path changed. `tests/debug_bridge/test_bridge.py::test_the_stdio_protocol_round_trips`
  pins the entry point; `test_relocation.py` pins that it is importable as a
  module.
- **JupyterLab:** imports `decider.debug_bridge.bridge.Bridge` in the kernel;
  the separate `decider-bridge` distribution is gone.

**Deliberate change:** the old top-level `decider_bridge` module and the
`tools/decider-bridge` directory no longer exist. No shim is provided; update
imports to `decider.debug_bridge`. The protocol, transports (stdio JSON-lines,
debugpy attach, the starlette websocket in `decider.serving.session_ws`), and
the cross-mode parity guarantee (`assert_equivalent`: interpreted == stepped ==
fused, NaN equals NaN) are unchanged.

## 2. Durable ids: `decider ids` is additive and optional

`decider ids [PATH]` inserts `id="…"` tokens into source. A project that never
runs it is unchanged: no id is required, and identity falls back to the derived
path/source (`tests/ids/test_ids.py::test_a_pipeline_without_ids_still_runs_and_describes`).
With `id=None` (the default) name, path, `step_map` and execution are
byte-for-byte identical. The generator is idempotent and only ever adds tokens —
it never renames, reorders or reformats. It refuses to run unless the tree is
clean and every affected file is git-tracked, and aborts (before writing) on a
syntax error, malformed id, invalid name or duplicate id.

**Deliberate change:** none to behaviour — ids are additive metadata. The only
obligation is on the operator: commit generated ids before they become durable
references in traces, comparisons or experiments.

## 3. Pipelines without generated IDs

Nothing in the runtime, the debugger, checks, experiments or MCP requires an
id. Derived identity (`path`, `source`) still resolves every step. Durable ids
are only required where a trace, comparison or experiment needs a reference that
survives rename/extract/reorder; `decider.checks.durable_ids` reports their
absence as a client-promotable diagnostic, not a runtime error.

## 4. MCP is additive

`decider mcp` (FastMCP over stdio) is a new command; it changes no existing
entry point. Structural and summarised reads are always available; raw
record/trace tools are off by default (`decider.mcp.rawData`); running code and
persisting are confirmation-gated via a destructive annotation.

## Behaviour preserved by promise

- Cross-mode numeric parity (`decider.testing.assert_equivalent`), including
  money as int64 cents and running totals in float64.
- The compile/fallback rule: compile-time failures and per-step compiled modes
  refuse to fall back per step; runtime errors propagate.
- Serving (`decider build`, `decider serve`), the template project, `decider
  guide`, and the request/params document semantics are untouched.
