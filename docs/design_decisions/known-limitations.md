# Known limitations

Gaps in the current code that users and maintainers should know about before relying on it; each was checked against `decikit/` when this file was written.

## Compilation and caching

**Object globals are keyed by identity.** A step reading an instance of a class of its own is fingerprinted with `id()` (`_engine/compile/fingerprint.py`); numbers, strings, dates, Decimals, regexes, enums, classes and plain containers of those are keyed by value. A class is keyed by its own methods' bytecode and constants, not by what its methods call, and a jitclass by its numba type, which differs per process.
Impact: the numba disk cache from `decikit build` misses on another process, and `decikit.lab.diff` reads the step as always changed. Binding warns with `HiddenStateWarning` (`Engine(strict_compile=True)` raises); put the data in literals, a config document, `param()`s or `param_table`s.

**A callable object read as `module.NAME` is not seen.** The walk follows the attributes a step reads off a module, but skips any callable that is not a function, module or class (so a ufunc or `np.searchsorted` is not flagged). An instance with `__call__` read that way (`transforms.normalise(x)` where `normalise = Normaliser(mean, std)`) is neither keyed nor warned about; read as a plain global it is.
Impact: editing its data leaves the cache key and `decikit.lab.diff` unchanged. Keep the data in a document or a `param()`, or import the object by name.

**A library is versioned by the distribution of its top-level module's name.** `importlib.metadata.version("yaml")` finds no `PyYAML`, so such a library is keyed by the Python version instead.
Impact: upgrading it changes the key of a step that calls one of its jitted functions only if that function's own bytecode changed, not if only a function it calls did; clear the numba cache (`NUMBA_CACHE_DIR`) after upgrading such a library.

**Frame steps are not checked for hidden state.** `Engine.bind` lints scalar and row steps only, so a frame step reading a registry (the fraud rule set) is not warned about.
Impact: such a step also reads as always changed in a diff. Checking frame nodes needs `_renamed_frame` (relabelled frame steps) exempted first.

**The disk cache is tied to the CPU.** Cached kernels miss silently on a host with a different CPU.
Workaround: build on the serving host type, or compare `cpu_target()` recorded at build time.

**Two steps in one fused kernel cannot read one column as `Columnar` of different Item types.** Binding raises `TypeError`.
Workaround: read the column with one Item type everywhere, or give the second reader its own column.

**A kernel numba refuses only at its launch takes every step in it to Python.** A step writing a `dict` or a `list` falls back alone at build, but a kernel can still fail at its first launch on a value numba only meets then (a `dict` param, a float returned for a `Decimal` output, an output declared as an `IntEnum` or `bytes`), and then each of its steps runs in Python. It is reported (`split at launch: ...` in `fallbacks()`, a `FallbackWarning` in fused mode, refused by `strict_compile=True`), but only when that kernel first runs, and under strict every step of the kernel needs `@allow_fallback`, even one that compiles on its own.
Impact: a Python call per row for each step of that kernel. Give the param a typed value or a `param_table`, and return what the output declares.

**Compile caches grow without eviction.** Step dispatchers, fused kernels, Arrow schema plans, the params cache and `RequestHandler._history` are unbounded.
Impact: a long-lived process that stages many versions or sees many distinct params documents leaks memory; restart periodically.

**The Arrow shim is compiled with `cc` on first import.** POSIX only, and it needs a C compiler or a cache directory the process can write to (`_engine/boundary/_arrow/_build.py`).
Impact: Windows, and slim containers without a compiler, cannot run compiled modes; the MCP editor-socket path on Windows is untested.

**`run(record)` is not instantaneous.** A short flow costs about 28 µs per record fused and 59 µs interpreted; a one-row frame through `run()` costs about 150 µs.
Workaround: serve single records as a dict through `run()`; see [serving latency](../architecture/serving.md).

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

**An all-null output of a type polars cannot name comes back `Null`.** An all-null or empty output takes its declared dtype when the annotation names one: numbers, `str`, `bytes`, `date`, `datetime`, `time`, a `Literal` of strings, `list[T]` of those, `Struct[...]`, `Columnar[...]` and a `TypedDict` of numbers, text and bools. A bare `dict` or `list`, `list[dict]`, `dict[str, T]`, `Any`, `timedelta`, a `Literal` of numbers and a `TypedDict` with a field outside those take their dtype from the values (`declared_dtype` and `_typed_dtype` in `_engine/run/state.py`).
Impact: such a column's dtype depends on the data; declare a typed shape where a consumer needs a fixed schema.

**An output with no return annotation, or `-> Any`, is stored differently by a kernel.** A step numba compiles writes it into a float64 column with no null mask (`numpy_dtype` reads an unknown annotation as float), while interpreted mode, and a step running in Python, keep the value as it is. A step returning an int gives `Int64` and `3` from `run(record)` interpreted, but `Float64` and `3.0` stepped and fused; one returning `None` on some rows gives a null interpreted and raises `TypeError: expected float64, got None` stepped and fused.
Workaround: annotate the return (`-> int`, `-> float | None`). Whether an unannotated output should be refused, run in Python or typed by its value is an open API decision.

**`run(record)` returns a plain `dict` or `list[dict]` output as the step built it; a frame row returns what polars made of it.** The record path hands back the Python value, while `run(frame)` stores the column as a polars struct or list of structs (`_series` in `_engine/run/state.py`). So a frame row has every key any item of the column has, `None` where an item lacked it (`"changes": {}` comes back as `{"amount": None, "grade": None}`), keys in the struct's field order rather than the item's, an int beside a float as a float, and a non-string key as text. A declared shape (`TypedDict`, `list[float]`) still turns an int the step returned for a float field into a float in the frame only. Loan granting's `waterfall_rules` and consolidation's `classified_accounts`, `settlements`, `scenarios`, `chosen_offer`, `runners_up` and `execution_package` differ this way between a record and a one-row frame.
Impact: compare such outputs by value with missing keys read as `None`, or declare a `Struct`/`Columnar` shape, whose fields are fixed. Round-tripping the record path's value through the same polars build would make the two equal at about 57 us per `dict` and 250 us per twelve-item `list[dict]` output per record.

**A `param_table` value is a numpy scalar in interpreted mode.** Arithmetic on it follows numpy, so `x / t.d[0]` with a zero gives `inf` and a warning where fused raises `ZeroDivisionError`.
Workaround: convert with `float(t.d[0])` in the step, or guard the divisor.

**Flow files cannot express everything.** `optimise(record=...)` takes a `TypedDict` class, so it is Python only; `each(execution_mode=...)` is accepted in Python but has no flow-file key (both modes run the same scatter and gather).
Workaround: write those steps in Python.

**A debug `Session` is a replay, not a live run.** Frame steps read inputs from the frame so they miss an input override, and an edit inside a loop reruns the loop from its first iteration (`_engine/debug/session.py`, `edit.py`).
Impact: stepping through frame steps after an override can show the original input.

**`run()` scores the whole frame as one batch.** Nothing splits it (only the experiment runner slices, at 100,000 rows), so memory grows with rows × (inputs + outputs): about 11.7 KB a row at 400 inputs and 633 outputs.
Workaround: split very large frames and concatenate the results.

## Checks and diff

**Source checks need source files.** Steps defined with `exec` (GUIDE examples) or in notebook cells get no AST findings.
Workaround: define steps in modules.

**Several checks are heuristics.** DCD003 (mutation), DCD109 (accumulators) and DCD503 (int64 range) are not flow-sensitive and may miss or over-flag; DCD104 treats any earlier test mentioning the divisor as a guard.
Impact: a quiet report is not a proof; use the `suppress_checks` field for known false positives.

**`constant_output` (DCD506) only checks outputs.** A branch arm that no row takes is not detected on data.
Impact: dead arms are found only by the static structure checks.

**Table overlap and gaps skip tables over 40 rows,** and tree/table reachability needs the `prove` extra (`z3-solver`); without it the z3 checks are reported under `skipped`.
Workaround: `pip install 'decikit[prove]'`; split very large tables.

**`diff.prove` stops at the IR hash and z3.** There is no normalised-document tier, no threshold-enumeration tier and no `first_divergent_step`.
Impact: an arithmetic change that preserves outputs reads as `identical=None`, not proven equal.

**`checks.yaml` is found from the current directory,** as datasets are, so `decikit check` and the MCP `check_report` depend on the working directory.
Workaround: pass `--config` / `config=` explicitly.

## Serving

**`RequestHandler.process_fn` runs the pipeline on the event loop.** With Starlette one worker serves requests one at a time, so `nogil` kernels never overlap inside it.
Workaround: run several worker processes; moving scoring to a thread pool is untested and should be measured first (`engine/concurrency.md`).

**`decikit serve` fails when `os.cpu_count()` is `None`** and no worker count is set (`_settings.py` computes `os.cpu_count() * 2 + 1`).
Workaround: set `serve.workers` explicitly in restricted containers.

**Arrow request bodies reach the kernels without a full validation pass.** `serving/_parse.py` reads Arrow IPC with polars, and the boundary shim reads string views with nanoarrow's unchecked accessor, so a malformed view that polars accepts could crash the worker. Whether polars rejects such views was not tested.
Workaround: accept Arrow bodies only from trusted callers, or serve JSON.
