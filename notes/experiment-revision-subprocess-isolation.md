# A `Scenario.revision` scenario runs in a fresh subprocess, not in-process

## The bug this starts from

`run_experiment` checks out a scenario's `revision` into a temp git worktree
(`decider.experiments.revision.materialise`) and used to import the pipeline
module from there directly into the current process
(`decider.experiments.runner._load_flow_in`). That module shares the live
pipeline's import name — and the live one is *already* imported, because
building an `Experiment` needs the live pipeline object in the first place.
Python caches imports by name, so re-importing the "same" name just returns
the cached (live) module: `_load_flow_in` silently ran the live code for both
the baseline and the "other revision" scenario, with no error, and a
`divergences` count of zero that looked like confirmation instead of a bug.
Confirmed directly: comparing two real commits of
`example_projects/experiments/credit_pipeline.py` (one adds a $50 affordability
fee) produced identical output under the old code, before this fix.

## Chosen design

A scenario whose resolved revision differs from the experiment's own runs in
a fresh `subprocess.run([sys.executable, "-m",
"decider.experiments._revision_worker"], input=<json>, ...)` instead of
importing the checked-out tree into this process. One JSON request on stdin,
one JSON reply on stdout — the same envelope `decider.debug_bridge` already
uses, for the same reason (a boundary an IDE/caller trusts, not a library
call): `{"ok": true, "result": {...}}` or `{"ok": false, "error": "..."}`.
The subprocess uses *this* process's installed `decider` (same
`sys.executable`, so `decider` resolves from site-packages/editable-install
before the worker ever touches the checked-out tree's `sys.path` entry);
only the entry module — and anything it imports that this process hasn't
already cached — loads fresh from the tree. `materialise(sha)` is now cached
per resolved sha within one `run_experiment` call, so N scenarios on the same
revision share one checkout instead of each paying for their own.

A fresh process is the only thing that's actually correct here, not just the
cheapest fix: popping the one entry-module name from `sys.modules` (the first
fix tried) only protects that one name. If the pipeline's entry module itself
imports a package with its own modules that also changed between revisions —
realistic for any real multi-file project — eviction-by-name doesn't catch
that staleness; a subprocess sidesteps it for free, at any depth, because it
never had the live code loaded to begin with.

## Measured cost

`example_projects/experiments/credit_pipeline.py`'s own two-commit history,
via `tests/experiments/test_headless.py`:
`test_a_revision_scenario_runs_the_checked_out_code_even_if_the_live_module_is_already_imported`:
the whole experiments test suite (21 tests, one of them this one) runs in
under 4 seconds. A single revision scenario costs roughly: git worktree
checkout (~0.5s, unchanged from before) + subprocess spawn and `import
decider` fresh (~0.8s, measured directly) — call it ~1.3s. That's the
*correctness* tax for diffing real code across commits, on a path the
project never optimized for speed (`notes/experiment-module.md`: the fused
in-process `run(df, params=...)` path exists because it's ~750x faster than
session replay; revision diffing was never meant to compete with that, it
replays a session already).

## Rejected alternative: `uv run --project <checked-out-tree>`

The first design considered also running the old revision under *its own*
pinned `decider` version, via `uv run --project <tree>`, so a revision whose
engine semantics actually changed would be compared faithfully instead of
under today's engine. Measured directly: a cold `uv sync` in a fresh
worktree (even with every wheel already cached locally) took **23 seconds**
— for one revision, let alone a sweep of several. Worse, it's solving a
problem the project already has a different, deliberate answer for:
`notes/vscode-redesign/background.md:130-133` names "revision comparison
uses today's installed `decider`... historical code that requires an older
engine interface can fail to import" as a known, *accepted* limitation, with
`flow.decider` (an engine-compatibility window) as the chosen mitigation —
reject an incompatible revision outright, don't attempt to run mixed engine
versions side by side. Chasing true engine-version fidelity here would be
solving a problem the project already decided not to solve this way, at 20x
the cost of the subprocess boundary that's actually needed for correctness.

## Rejected alternative: evict the entry module from `sys.modules`

Fixes the exact bug the module-cache collision caused, in six lines, no new
process boundary. Kept initially, replaced once it became clear it only
protects the *one* module name passed as `entry` — any module that entry
module imports, which also changed between revisions, is still silently
stale. Real multi-file pipelines (anything beyond this spike's flat
`ratio → approved` fixture) make that gap real, not theoretical.

## Open decisions

- `materialise` caching is scoped to one `run_experiment()` call. A
  persistent, cross-run cache keyed by sha (reused across separate
  `experiment.yaml` runs that both diff against the same base commit) would
  cut the ~0.5s checkout cost further, but needs an eviction policy this
  note doesn't attempt.
- The subprocess's 120s timeout (`_subprocess_run`) is a guess, not measured
  against a real large frame; `frame.write_parquet`/`pl.read_parquet` at the
  input scale of the main execution path (millions of records) is untested
  on this specific path, though revision scenarios were never meant to carry
  that scale.
- `_revision_worker`'s `mode: "fused"` branch still refuses a scenario with
  overrides/row/`at` the same way `_param_only` already decided in the
  parent — if that routing logic in `runner.py` ever changes, the worker's
  copy (currently just "the parent already decided, trust it") needs to stay
  in lockstep; they're not shared code today beyond both importing
  `_fused_run`/`_fork_run`.
