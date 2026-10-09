# Known limitations

Gaps in the current code that users and maintainers should know about before relying on it; each was checked against `decider/` when this file was written.

## Compilation and caching

**Arbitrary object globals are keyed by identity.** A step reading an instance of a user class, or a container holding one, is fingerprinted with `id()` (`_engine/compile/fingerprint.py`); numbers, strings, dates, Decimals, regexes, enums, classes and plain containers of those are keyed by value. A class is keyed by its own methods' bytecode and constants, not by what its methods call, and a jitclass by its numba type, which differs per process.
Impact: the numba disk cache from `decider build` misses on another process; put constants in literals, `param()`s or `param_table`s.

**The disk cache is tied to the CPU.** Cached kernels miss silently on a host with a different CPU.
Workaround: build on the serving host type, or compare `cpu_target()` recorded at build time.

**Two steps in one fused kernel cannot read one column as `Columnar` of different Item types.** Binding raises `TypeError`.
Workaround: read the column with one Item type everywhere, or give the second reader its own column.

**A compiled step returning a dict or a list of dicts runs in Python.** `compile_call` accepts it, but no kernel stores a dict, so the fused kernel fails at its first launch and splits into per-call Python runs, silently (no `FallbackWarning`, absent from `fallbacks()`, and `strict_compile=True` does not raise).
Impact: such a step costs a Python call per row although it looks compiled; its output is still checked against the declared type. Return a `Struct[...]` tuple or a `Columnar[...]` list of tuples to compile it.

**Compile caches grow without eviction.** Step dispatchers, fused kernels, Arrow schema plans, the params cache and `RequestHandler._history` are unbounded.
Impact: a long-lived process that stages many versions or sees many distinct params documents leaks memory; restart periodically.

**The Arrow shim is compiled with `cc` on first import.** POSIX only, and it needs a C compiler or a cache directory the process can write to (`_engine/boundary/_arrow/_build.py`).
Impact: Windows, and slim containers without a compiler, cannot run compiled modes; the MCP editor-socket path on Windows is untested.

**`score()` is not instantaneous.** A short flow costs about 28 µs per record fused and 59 µs interpreted; a one-row frame through `run()` costs about 150 µs.
Workaround: serve single records as a dict through `score()`; see [serving latency](../architecture/serving.md).

## Numbers

**int64 arithmetic wraps silently.** Kernels do not saturate, and interpreted mode raises `OverflowError` where fused wraps.
Workaround: `Decimal` / `Currency` for money, and the `int64_overflow` check (DCD101).

**`decimal.round` is approximate in two corners.** It is one ulp off on rare halfway cases just past 2^53, and approximate beyond 22 digits (`ops/decimal.py`).
Impact: only matters for very large or very precise values.

**A literal `x ** 2` can differ from libm `pow` by one ulp** because LLVM folds it into `x * x` (`_engine/compile/semantics/numbers.py`).
Workaround: write a variable exponent if bit-equality with CPython matters.

**Decimal does not mix with floats freely.** `float / Decimal` and `Decimal ± float` are errors in Python and kernels; a float may multiply or divide a decimal.
Workaround: `float(d)` for ratios; see the [types guide](../guides/types-and-money.md).

**A float read as an `int` must be whole.** `7.0` is accepted; `7.5`, NaN and inf raise `NotIntegralError` naming the input or step, for an `int` step input, an `int` tree feature, and an `-> int` output, in frames, records and every mode. A step that reads a float another step wrote as `int` receives the float unchanged (only an `int` output is checked), and `-2**63` still reads as null in a tree.
Workaround: round in a step (`int(x)`, `round(x)`), or declare the column `float` (the tree `feature_types` entry `float`).

## Steps and flows

**Frame-step `writes` dtypes are not verified or cast.** A declared dtype is kept when the step returns it, but a different dtype is accepted as-is (only missing columns and row-count changes raise).
Impact: a wrong dtype surfaces later, at the reader; declare `writes={"col": pl.Float64}` and return exactly that.

**Flow files cannot express everything.** `optimise(record=...)` takes a `TypedDict` class, so it is Python only; `each(execution_mode=...)` is accepted in Python but has no flow-file key (both modes run the same scatter and gather).
Workaround: write those steps in Python.

**A debug `Session` is a replay, not a live run.** Frame steps read inputs from the frame so they miss an input override, and an edit inside a loop reruns the loop from its first iteration (`_engine/debug/session.py`, `edit.py`).
Impact: stepping through frame steps after an override can show the original input.

## Checks and diff

**Source checks need source files.** Steps defined with `exec` (GUIDE examples) or in notebook cells get no AST findings.
Workaround: define steps in modules.

**Several checks are heuristics.** DCD003 (mutation), DCD109 (accumulators) and DCD503 (int64 range) are not flow-sensitive and may miss or over-flag; DCD104 treats any earlier test mentioning the divisor as a guard.
Impact: a quiet report is not a proof; use the `suppress_checks` field for known false positives.

**`constant_output` (DCD506) only checks outputs.** A branch arm that no row takes is not detected on data.
Impact: dead arms are found only by the static structure checks.

**Table overlap and gaps skip tables over 40 rows,** and tree/table reachability needs the `prove` extra (`z3-solver`); without it the z3 checks are reported under `skipped`.
Workaround: `pip install 'decider[prove]'`; split very large tables.

**`diff.prove` stops at the IR hash and z3.** There is no normalised-document tier, no threshold-enumeration tier and no `first_divergent_step`.
Impact: an arithmetic change that preserves outputs reads as `identical=None`, not proven equal.

**`checks.yaml` is found from the current directory,** as datasets are, so `decider check` and the MCP `check_report` depend on the working directory.
Workaround: pass `--config` / `config=` explicitly.

## Serving

**`RequestHandler.process_fn` runs the pipeline on the event loop.** With Starlette one worker serves requests one at a time, so `nogil` kernels never overlap inside it.
Workaround: run several worker processes; moving scoring to a thread pool is untested and should be measured first (`engine/concurrency.md`).

**`decider serve` fails when `os.cpu_count()` is `None`** and no worker count is set (`_settings.py` computes `os.cpu_count() * 2 + 1`).
Workaround: set `serve.workers` explicitly in restricted containers.
