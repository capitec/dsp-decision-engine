# decider

Use `uv` for everything: `uv run pytest`, `uv run python`. Python 3.10+.

## Comments and docstrings

- Docstrings only on public API, written as user docs: a one-line summary, the
  arguments that aren't obvious, and a short example.
- Private helpers get no docstring unless the behaviour is non-obvious.
- Inline comments explain why, never what.
- Never reference docs, section numbers, stages, experiments, review findings or
  "the agent" in code. No `doc 03`, `§4.2`, `EXPERIMENTS`.
- Design rationale ("we chose X over Y because Y failed") goes in
  `notes/<topic>.md`, not in code.

## Files

- One purpose per file, under 500 lines. When a file grows past that, split it
  into a package (`foo/__init__.py` plus files) instead.
- An exception goes in `ALLOWLIST` in `tests/test_conventions.py`, with a reason.
- `uv run pytest tests/test_conventions.py -q` enforces the line limit and the
  banned references.

## Porting code

- Invoke the `ponytail:ponytail` skill in `ultra` mode and apply it throughout:
  the simplest code that passes the tests, no speculative abstractions, stdlib
  first.
- If a simplification would change behaviour a test pins down, keep the
  behaviour and leave a `# ponytail: <what could be simpler>` comment.
- Port behaviour, not text. Rewrite comments to the rules above.
- Port tests alongside the code into `tests/<area>/`. Test names describe
  behaviour (`test_missing_required_param_is_invalid`, not `test_case_3`).
- When a source test covers behaviour the new design removes, drop it and list
  it with the reason in your report.

## Layout

- `decider/` is the package. Its layout follows `Design.md` §8, with the file
  split for `steps/` and `engine/ir/` in `IR.md` §9; the code is the truth
  where they differ.
- `engine/` imports nothing from `steps/trees|tables|scorecard`, `config/` or
  `serving/`.
- `benchmarks/` holds plain scripts (`uv run python benchmarks/<name>.py`),
  not tests. The last numbers against decider2 are in
  `notes/benchmarks-vs-decider2.md`.

## Contract

- `IR.md` is the contract, amended by `notes/spec-amendments.md`. Where they
  disagree, the amendments win.
- If the spec is wrong or ambiguous, stop and report. Don't guess.

## Numeric and numba traps

- Numba cache: generated code lives in real files (not `exec`) and is
  byte-identical between runs. Key compiled code by content, never by path or
  name.
- Serving kernels are `nogil=True`. `fastmath` is off by default.
- `round()` differs between CPython and numba, and int64 wraps silently. Money
  is int64 cents; running totals accumulate in float64.
- Only `NumbaError` triggers the Python fallback. Runtime errors propagate.
- Nested polymorphic pydantic fields need `SerializeAsAny`, or subclass fields
  are silently dropped on dump.
- Single-record calls take a dict, not keyword arguments. Cache converted params
  bundles; conversion costs more than validation.
- Each git worktree has its own numba cache, so the first test run in a fresh
  worktree is slow.
