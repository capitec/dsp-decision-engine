# decikit

Use `uv` for everything: `uv run pytest`, `uv run python`. Python 3.10+.

To learn the API as a user: `uv run decikit guide` (`decikit/GUIDE.md`) and the
docstrings. Every Python example in `GUIDE.md` runs in `tests/test_guide.py`;
keep it that way.

`docs/` is the docs site (nav in `zensical.toml`; `uv run --group docs zensical
build`). `docs/design_decisions/` is not published: the decisions behind each
area, the experiments behind them and `known-limitations.md`.
Read an area's decision file before changing that area. `docs/design_specs/`
(also unpublished) holds proposals for work not yet built, one folder per rework.

## Comments and docstrings

- Docstrings only on public API, written as user docs: a one-line summary, the
  arguments that aren't obvious, and a short example.
- Private helpers get no docstring unless the behaviour is non-obvious.
- Inline comments explain why, never what.
- Never reference docs, section numbers, stages, experiments, review findings or
  "the agent" in code. No `doc 03`, `§4.2`, `EXPERIMENTS`.
- Design rationale ("we chose X over Y because Y failed") goes in
  `docs/design_decisions/decisions/<topic>.md` (a measurement as an experiment in
  `docs/design_decisions/experiments/`), not in code. Key points only; an open
  gap goes in `docs/design_decisions/known-limitations.md`.

## Files

- One purpose per file, under 500 lines. When a file grows past that, split it
  into a package (`foo/__init__.py` plus files) instead.
- A file-length exception goes in `ALLOWLIST` in `tests/test_conventions.py`;
  import, `__all__` and layer exceptions go in `tests/conventions_allowlists.py`.
  Each needs a reason, and stale entries fail the test.
- `uv run pytest tests/test_conventions.py -q` enforces the line limit and the
  banned references.

## Writing code

- Invoke the `ponytail:ponytail` skill in `ultra` mode and apply it throughout:
  the simplest code that passes the tests, no speculative abstractions, stdlib
  first.
- If a simplification would change behaviour a test pins down, keep the
  behaviour and leave a `# ponytail: <what could be simpler>` comment.
- Tests go in `tests/<area>/`. Test names describe behaviour
  (`test_missing_required_param_is_invalid`, not `test_case_3`).

## Layout

- `decikit/` is the package. Its layout, the layer map and where a new feature
  goes: `docs/architecture/package-layout.md`. The code is the truth where they
  differ.
- `_engine/` imports nothing from `steps/trees|tables|scorecard`, `config/` or
  `serving/`.
- Packages import only downward per the layer map in `package-layout.md`;
  `tests/test_conventions.py` enforces it.
- `benchmarks/` holds plain scripts (`uv run python benchmarks/<name>.py`),
  not tests. Each backs an experiment in `docs/design_decisions/experiments/`.

## Contract

- `docs/architecture/` plus the code describe the system (steps and IR: `steps-and-ir.md`,
  `engine/ir.md`); `docs/design_decisions/` holds the why.
- If the docs and the code disagree, or the intended behaviour is ambiguous, stop
  and report. Don't guess.

## Numeric and numba traps

- No generated source: fused kernels are built in-process from LLVM IR, and
  user steps use numba's disk cache with the key extended by content
  (`fingerprint` plus `SALT`). Key compiled code by content, never by path or
  name.
- Serving kernels are `nogil=True`. `fastmath` is off by default.
- Kernels use CPython-equal `round(x, n)` and `x ** n`; int64 still wraps silently. Money
  is `Currency` / `Decimal[p, s]`: an int64 mantissa with the scale in the type, half-even
  operators, `decikit.ops.decimal.round` for other modes. Running totals accumulate in float64.
- Compile-time failures and steps compiled engines refuse fall back to Python per
  step (with a warning; `FusedEngine(strict_compile=True)`, or any numba engine's, raises). Runtime errors
  always propagate.
- Nested polymorphic pydantic fields need `SerializeAsAny`, or subclass fields
  are silently dropped on dump.
- Single-record calls take a dict, not keyword arguments. Cache converted params
  bundles; conversion costs more than validation.
- Each git worktree has its own numba cache, so the first test run in a fresh
  worktree is slow.
