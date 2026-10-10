# API layout

The rules that shape the public surface (layers, private names, checks, diff, thread safety, kernel functions, extension points); read it before moving a module or adding a public name.

## Imports go down a layer map, and a test enforces it

- L0 foundation (`exceptions`, `ops`, `_registry`, `_serde`, `_settings`, `_suggest`), L1 core (`decikit`, `steps`, `ir`, `trace`, `debug`, `config`, `_engine`), L3 libraries (`lab`; `credit` is mapped but not built), L4 tooling (`serving`, `_tooling`, `__main__`).
- Why: a library that reaches up into serving or tooling can't be imported alone; `tests/test_conventions.py` fails any upward import or same-layer cycle, and allowlists in `tests/conventions_allowlists.py` can only shrink.
- Accepted exceptions: `__init__ -> lab` (`describe` is the one deliberate second path), `_settings -> config`, and the `_engine <-> steps` cycle.
- Not enforced: `_engine/` importing nothing from `steps/trees|tables|scorecard`, `config/` or `serving/`; it holds today.
- See: [package layout](../../architecture/package-layout.md) (full map, the test table, where a new feature goes).

## A leading underscore means private; public packages are thin

- Any `_` path segment is private. A public package is an `__init__.py` over `_`-prefixed files, and every public module defines `__all__` of at most 25 names (`decikit` and `steps/trees/schema` are allowlisted).
- Why: `__all__` alone hides nothing (`dir()` and deep imports still work); the underscore is what IDEs, `help()` and docs skip and what a test can check. Tests may import private modules.
- Only core may import `decikit._engine`. Private top-level helpers (`_settings`, `_suggest`, `_serde`, `_registry`) are shared across packages by design.
- Not: compatibility shims. A moved name changes its callers in the same commit.
- See: [package layout](../../architecture/package-layout.md)

## One canonical path per name; facades keep users out of the engine

- `decikit` is the authoring vocabulary. `decikit.ir`, `decikit.trace` and `decikit.debug` re-export the engine types that step authors and tools need.
- Why: nothing a user needs should require `decikit._engine`; when a library reaches into the engine, the fix is a facade name, not an allowlist entry.
- `import decikit` must not import numba: `Engine`, `Executable` and the three `*Config` classes load through a module `__getattr__` (`tests/test_imports.py` checks in a subprocess).
- Same name, different meaning is avoided by renaming (`ParamTable`, `NodeVersion`, `WarningEvent`); `run` stays per-namespace (`Step.run`, `checks.run`, `experiments.run`).
- See: `decikit/__init__.py`

## Checks are registry modules with stable codes

- A check subclasses `Check` (root of group `decikit.checks`), has a `DCDnnn` code that is never reused (0 determinism, 1 numeric, 2 structure, 3 params, 4 ids, 5 data and runtime), and yields `Finding`s whose step ref is the one `describe()` produces.
- Why: registry roots give naming from JSON, lazy built-ins, entry-point discovery and `json_schema()` for free; a plain callable has no code, config or suppression.
- Opt-in checks (`needs_data`, `executes`) are not in `standard`; a check that cannot run appears under `report.skipped` with the reason, never silently.
- See: [Checks](../../architecture/lab/checks.md), [checks guide](../../guides/checks.md)

## Check policy is `checks.yaml` at the project root; suppression lives on the step

- The policy file sits beside `configs/` and `datasets/`, found like datasets (setting `checks_path`, env `DECIKIT_CHECKS_PATH`). Suppression is `suppress_checks=[alias or code]` on every constructor, `ConfigurableStep` and flow-file key.
- Why not `configs/`: that is the versioned store `decikit serve` activates; policy is a dev and CI concern and must never be deployed with params. Suppression is stored on `Origin`, which is outside cache keys, so adding one never recompiles.
- Not: inline `# ignore` comments or id-based suppressions in config; the step is the one place to look, and moves with the step in review.
- An unknown suppression name is itself a finding (DCD403), not a build error, because a custom check may not be imported yet.
- See: [Checks](../../architecture/lab/checks.md)

## Findings, not policy: the decider toolkit reports and the client decides

- Severity defaults are advice. `Report.ok` and `--fail-on` are the gate; SARIF output is for CI and editors.
- Why: what is an error differs per project, so the project's `checks.yaml` sets severities.

## Diff owns "what changed"; `identical=True` needs a proof

- `lab.diff` answers structure, outputs, revisions and proof; experiments use it, never the reverse. Mode agreement stays in `lab.testing.assert_equivalent`.
- `prove` returns `identical=True` only from an equal IR content hash or a z3 unsat over changed trees/tables; a counterexample or an unmodelled step gives `identical=None`, never `False`, because the step alone may never see that row.
- Why: sampled evidence is not a proof, and a source-text signature (the original) ignored constants and called different flows equal.
- Not built: normalised-document tier, threshold-enumeration tier, `first_divergent_step`.
- See: [Diff](../../architecture/lab/diff.md)

## Everything on the serving path is thread-safe; nothing else has to be

- One `Executable` is called from many threads, including its first call; `Session`, `TraceSink`, experiments, checks, diff and the bridge are single-caller.
- Rules: no per-call state on `self`; compile into locals and publish one immutable snapshot; caches are write-once with `setdefault`; module caches use `once(cache, key, build)`; `exe.report` is per thread. A warm `run(record)` takes no lock.
- Not relied on: the GIL. The rules hold on a free-threaded build.
- `tests/serving/test_concurrency.py` and the runner `self`-write test in `test_conventions.py` guard it.
- See: [Concurrency](../../architecture/engine/concurrency.md)

## Kernel functions come in two layers under one contract

- Implicit semantics (`round`, `**`, string ops on spans, decimal operators) are `@overload`/`@intrinsic` in `_engine/compile/semantics/`, only on the toolkit's own numba types or concrete builtin signatures. Functions users call live in `decikit/ops/<domain>.py` as `@register_jitable`, one source for Python and kernels.
- Why: an overload on a generic type (all ints, bare tuples) would hijack user code in the whole process. `SALT` hashes every file in `semantics/`, `ops/`, `span.py` and `decimal_type.py` by walking, so a new file cannot leave stale disk-cached machine code.
- Same answer in every mode, pure, nogil-safe; third parties write `@helper` functions or their own `register_jitable` modules, never overloads. `fingerprint` follows `module.attr` calls, so editing a helper module invalidates dependent kernels.
- Domain modules are named `text` and `decimal`, not `str`; `from decikit.ops import decimal` shadowing stdlib `decimal` is accepted since stdlib `Decimal` cannot run in a step.
- Not: a `Cents` int subclass with overloads; money is `Decimal` / `Currency`.
- See: [Foundation](../../architecture/foundation.md), `tests/test_conventions.py`

## Credit functions have three homes (not built yet)

- Per-record pure functions go in `decikit.ops.credit` under the kernel contract; fitting from data in `decikit.credit` (documents the decider toolkit already runs, e.g. a scorecard fit) at L3; population metrics (Gini, AUC, PSI) as experiment `Metric`s.
- Why: no new step constructors; heavy dependencies stay behind a `decikit[credit]` extra, polars first, and proprietary code plugs in through entry points instead of merging.
- Neither `decikit.credit` nor `ops.credit` exists in the code; only the layer map reserves the name.

## Extension points are registry roots plus entry points

- Seven roots (`ConfigurableStep`, `ConfigStore`, `DatasetProvider`, `Check`, `Metric`, `Plot`, `Block`) resolve a tag by registry, then by `importlib.metadata` entry points in group `decikit.<group>`, and only ever return registered subclasses so config cannot name arbitrary code.
- Closed on purpose: IR node types. Engines are a registry root (`Engine`; see `engine.md`). `Callback` is a plain class because callbacks are not saved with an experiment definition.
- Custom step types build IR with `ctx.call` / `ctx.expand` from `decikit.ir`; relax `ctx.call` before adding a constructor.
- See: [Package layout](../../architecture/package-layout.md) (extension points), [extending guide](../../guides/extending.md)
