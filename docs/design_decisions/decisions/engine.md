# Engine

Why the engine (IR, wiring, runners, kernels, compile cache, fallback, `score`, `each`) is shaped the way it is; read it before changing any of those or proposing an alternative.

## Pipeline structure lives in Python

- Config holds params documents and `ConfigurableStep` documents and refers to code only by registered tag; flow files (YAML beside the code) are the one copy of structure. Not: `output_fn` pointers or code in config.
- Why: the previous engine kept order in JSON and in code, and they drifted (51, 120 and 192 steps for one flow); values duplicated between config and code made edits silently ineffective.
- See: [config store](../../architecture/config-store.md), [package layout](../../architecture/package-layout.md)

## Compiled code is keyed by content, never by origin

- Dispatchers and fused kernels are shared in-process by `fingerprint(fn)`: bytecode with constants, closure cells and read globals (a class by name plus its own methods and constants, so it holds across processes); not name, file, line or `Origin` path. Params bundle classes get a qualname built from their fields so they pickle by reference.
- Why: renaming or nesting a step changes its path, not its code, and a moved file misses numba's cache (4.6 s against 0.18 s warm at 200 args). A name-only key collided two same-named NamedTuples, a correctness bug (the slower dispatch once blamed on it measured 1.03x, see [params bundles](../experiments/params-bundle-dispatch-cost.md)). Not: `co_code` alone (omits `co_consts`); not clearing one cache, since CPython's `.pyc` and numba's index each served a stale constant alone.
- See: `decikit/_engine/compile/fingerprint.py`, [cache](../../architecture/engine/cache.md)

## A step that reads a hidden object is warned about when it is bound

- `Engine.bind` calls `hidden_state(fn)` for each scalar and row step, in every mode, and warns with `HiddenStateWarning` once per closure variable or global that holds an instance of a class, or a `dict`, `list` or `set`. `@allow_fallback` does not silence it, and the opt-out is a warnings filter on the category. `hidden_state` is the cache key's own walk with a collector (`_Walk.found`), so the two cannot disagree. A module attribute read as `mod.NAME` is walked and keyed by value like a global.
- Why: an object is invisible to `decikit.lab.diff` (strict fingerprint refuses it, so the step always reads as changed) and to numba, and 113 of 903 steps in the 12 examples read one. Every Python step there is `@allow_fallback`, so a check on the fallback path would flag nothing. The earlier key walk also ignored `mod.K`: a new value was served from numba's disk cache in the next process, and it descended into the standard library (`statistics.mean`, `dataclasses.asdict`) and flagged steps for no reason.
- A function of the standard library or an installed package is keyed by name, version, its own code and its closure, not walked. The name and version alone collided: two closures from one installed factory shared one compiled dispatcher, so `stepped` and `fused` answered with the first closure's value. The step function itself is always walked, so a step type shipped as a wheel is checked like one in the project.
- Mutable containers count although the key sees an exact literal `dict` or `list` by value: a step reads them at call time, so they can be edited after the bind, and numba compiles neither. A tuple, frozenset, enum member or date is data. A frame step is not checked (the fraud rule set and `_renamed_frame` would need an exemption first).
- Cost: 0.5 to 10.5 ms once per bind for the 12 example pipelines (53 to 128 steps each); nothing per call.
- `Engine(strict_compile=True)` raises `EngineError` for the same findings (`warn_hidden_state(plan, strict=self.strict_compile)`). A test builds every example from its latest config version and asserts no `HiddenStateWarning`; the one entry in its allow-list is governance-replay's `FLOWS`, the registry of the flows the replayer replays.
- Not: a digest of arbitrary objects (a changed method would read as unchanged), raising always (a project would fail to build before it could migrate; the default stays a warning).
- See: `tests/compile/test_hidden_state.py`, `tests/wiring/test_hidden_state_warning.py`

## A table seeded from a config document warns when a params document shadows it

- `param_table(..., seeded_from="<document>")` records that the default rows were read from a config document. Staging a config version (`RequestHandler.stage()`, so `decikit build` too) emits `ShadowedTableWarning` per table whose param the version's params document sets, naming the table, the document and the params key (`decikit/_engine/params/shadow.py`). Per-call overrides at run time (experiments) are never checked.
- Why: the params rows replace the document's without a trace, so editing the document appears to do nothing. Loan granting's `shared.rate_card` and consolidation's card tables are seeded this way.
- Three ways off, all deliberate: filter the category; `DECIKIT_BUILD__WARN_SHADOWED_TABLES=false` (`settings.build.warn_shadowed_tables`); `allow_override=True` on the table, for a table the params document is meant to replace.
- Not: an error (replacing the rows is legitimate), or a check on every run (the document is fixed per version).

## A table default is hashed by column, not by cell

- `decikit.lab.diff` hashes a `param_table` default as one buffer per column (dtype, name and bytes), once per snapshot, and shows a long default cut at 80 characters without building its whole `repr`. `diff.structure` on a 63,360-row default took 6.3 s, 5.3 s and 2.6 s (loan granting, consolidation, campaign trees) and takes about 0.27, 0.20 and 0.09 s.
- Why: the generic walk made a Python call per cell, six million for loan granting, and `_show` rendered the whole table to keep 80 characters. Columns are typed (`int`, `float`, `bool`), so dtype and bytes fix the value; a changed cell, row order, row count or column type changes the digest, and it does not depend on the process's hash seed.
- Not: memoising across snapshots (a default edited between two diffs must hash again).

## No source is generated; only user steps are disk-cached

- Fused kernels are built from LLVM IR by a numba `intrinsic` once per process. User steps use `cache=True` with the index key extended by `SALT` (hash of `compile/semantics/`, `span.py`, `decimal_type.py`, `ops/`) and `fingerprint(fn)`; `decikit build` records the CPU target and `decikit serve` warns on a mismatch.
- Why: `exec`'d code cannot be cached and a regenerated file changes mtime so every entry misses; warm is about 0.18 s against 4.9 to 47 s cold. numba's key omits `co_consts` and called overloads (silent stale answers). Not: forcing `NUMBA_CPU_NAME=generic` in the engine (the container image defaults to it; see serving), or normalising mtimes.
- See: [experiment](../experiments/kernel-compilation-choices.md), [cache](../../architecture/engine/cache.md)

## Fusion follows a fixed rule, never a cost heuristic

- One kernel per innermost run of scalar calls; row calls, frames, branches, loops and scatter-gather end the run; a call that cannot compile splits it; stepped mode is one kernel per call. Any other grouping is authored.
- Why: fusion is non-monotone in body cost (split/fused at 1M rows: 13.0 at 20 cheap modules, 0.22 at 20 heavy ones; register spills), and no static signal predicts it. Not: a step cap, a "break-even rows" rule, or lost vectorisation as a signal.
- See: [experiment](../experiments/kernel-compilation-choices.md), [compile](../../architecture/engine/compile.md)

## Kernels are `nogil=True` and never `fastmath`

- Every kernel is `njit(nogil=True)`; `step(nogil=...)` is recorded on the node but the compiler ignores it. `fastmath` is off.
- Why: 16 threads on one kernel give p99 883 us with `nogil` and 254 ms without (convoy on the GIL); with `fastmath` 46 to 73% of rows differed from Python, so modes could not agree bit for bit. Not: GC tuning (`gc.disable()` or `freeze()` left p99.9 at 1.15-1.17 ms) or `sched_setaffinity` pinning (max +65%) as a tail fix; the tail is OS preemption ([experiment](../experiments/single-record-tail-latency.md)).
- See: [experiment](../experiments/kernel-compilation-choices.md), [concurrency](../../architecture/engine/concurrency.md)

## Kernels run serial; there is no `prange`

- Every kernel is a plain `range` loop over rows; nothing in `decikit/` compiles with `parallel=True`.
- Why: the crossover is about 90 µs of serial time against a 65-70 µs fork/join floor, which a warmup probe cannot predict (52.8% right) and a row count cannot either (66.7%); `parallel=True` compiles 1.9-7.9x slower; a `ZeroDivisionError` in a serial loop is `inf` under `prange`, so the choice would change answers.
- See: [experiment](../experiments/kernel-compilation-choices.md)

## Fall back to Python only on compile failure, one step at a time

- `FALLBACK_ERRORS = (NumbaError, UnsupportedBytecodeError)`; around `dispatcher.compile` alone, `NotImplementedError`, `AssertionError` and `TypeError` count too. The kernel splits around the step; a kernel that fails whole on launch becomes one `Fallback` per call. A runtime error always propagates. A scalar step declaring `*args` or `**kwargs` (a hand-built `CallNode`) falls back too: no kernel can bind a column to it. `@allow_fallback` accepts a fallback silently; `Engine(strict_compile=True)` refuses it.
- Why: a kernel cannot call Python except through `objmode`, slower than giving up (4 steps, 200k rows: all-njit 7.1 ms, Python 167 ms, per-row `objmode` 548 ms). Bare `Exception` would hide bugs as fallbacks. Not: `objmode` escapes, which cost about 2.7 µs a row (77x all-njit, 615x inside `prange`).
- See: [experiment](../experiments/kernel-compilation-choices.md), [cache](../../architecture/engine/cache.md)

## A declared dict, list, `Struct` or `Columnar` output is checked where the step runs

- No kernel stores a dict, so a step returning a dict or `list[dict]` runs in Python (as a `Fallback`, or as a kernel that fails at its first launch and splits into per-call Python runs). It is validated with strict pydantic on every call, cached per annotation; strict, so an int is a float but a str or bool is not. A `Struct` or `Columnar` step returning tuples does run in a kernel: its tuple type, as numba infers it, is compared with the `Item` once at compile time (`compile/declared.py`); a mismatch is a fallback, which the Python check then refuses.
- Why: scalar outputs were refused by the numpy store, but a dict or list went through unchecked and `run()` then turned `{'v': 1.5}` beside `{'v': 'high'}` into text. Comparing numba's inferred dict type with the TypedDict would be free but pointless: the dict never reaches a kernel. Validation costs about 1.4 us for 2 dicts of 1 field and 3 us for 6 of 5 fields (microbenchmark), 3.6 and 6.2 us on a whole `score()`. That was before the flags and masks around the check were prepared at bind and made on demand; see [per-step overhead](../experiments/per-step-overhead.md).
- A plain `dict` or `list[dict]` declares no shape, so `run()` keeps the loose building there (int beside float widens, non-string keys become text) but refuses a number beside text. A declared shape gets a polars dtype: polars' strict inference truncates `1.5` to `1` in a later row of a nested list.
- A `typing.TypedDict` is rebuilt from `typing_extensions` before pydantic sees it (`pydantic_ready`), for output checks and request validation: pydantic refuses `typing`'s before Python 3.12, which left outputs unchecked and failed JSON arrays there. A numpy number is checked, and served, as the Python number it is; only the retry after a failed validation converts, so a valid output pays nothing.
- See: `tests/run/test_declared_outputs.py`

## Only numbers and `Text` spans enter a kernel

- A step reading or writing a plain `str`, or reading a `date`, `list` or `dict`, runs in Python without asking numba. `Text` and `bytes` are `(address, length)` spans.
- Why: numba would type a `date` as float64 and fail at the first run; no array holds a variable-length string, and one compiled call per row costs more than the Python body it replaces.
- See: `compile_call` in `decikit/_engine/compile/njit.py`, [runners](../../architecture/engine/runners.md)

## All modes are one interpreted driver with compiled calls

- `SteppedRunner` replaces only how a scalar or row call runs; `FusedRunner` also replaces sequences, branches and loops. A fused kernel yields one `before`/`after` pair with its first step's origin.
- Outputs agree in every mode, nulls included: state keeps raw columns plus masks, each reader applies its own null policy, and Python steps see Python scalars so `x / 0.0` raises as in a kernel. Fused is 3.3x to 4x interpreted on real projects; the rest is the object boundary and fallbacks.
- See: [runners](../../architecture/engine/runners.md), [experiment](../experiments/decider-vs-decider2.md)

## A branch or loop of plain scalar steps packs into one kernel

- Fused mode emits a `Fork` (`if/else`, `switch`) or `Repeat` (counted loop) as LLVM blocks in the same intrinsic. Packs only if every call is scalar and compilable, none has a nullable output, merged or carried names share one numeric, bool or `Struct[Item]` dtype, and nothing inside but the merge or carry is an output; otherwise the same answers come from the unpacked path, also taken per launch on a guarded null or `ArmOutOfRange`.
- Why: the old opcode program with a float64 register per row ran a branch at 3.3M rows/s; emitted blocks run 41.6 to 44.3M (loop 0.64M against 6.33M).
- See: [experiment](../experiments/decider-vs-decider2.md), `decikit/_engine/compile/packed.py`

## Lazy validation validates a kernel's nodes before it launches

- With `params_validation="lazy"` the driver validates every node of a kernel just before launch, and a branch or loop packs only if no call but its own condition has params (eager validation packs regardless).
- Why: a kernel is straight-line, so every node runs on every row; suspend/resume inside it would validate the same nodes in the same order. Revisit if a hot branch with params in its arms needs packing. Known difference: an invalid param is reported before an earlier runtime error in the same kernel.
- See: [params](../../architecture/engine/params.md)

## Wiring binds each read to a version, with strict scopes

- An unknown name is an input column unless it is close to an earlier output (difflib 0.8, within two edits, not digits only): then `WiringError`, else a warning. Reading a name as an input column that a later node writes is an error (self-read allowed), as is one input column read with two types. Frame steps of unknown lineage are barriers.
- Branch arms and loop bodies are scopes; only `modifies` and `carries` leave, and reading an arm-internal name afterwards is an error (the write would otherwise be silently lost). Ids are positions, never path text.
- See: [wiring](../../architecture/engine/wiring.md)

## Nested configurable steps dispatch by registry lookup

- A `StepRef` resolves the `type` tag in a per-root dict (`WrapValidator` plus `SerializeAsAny`); no union is rebuilt on registration, built-in tags resolve lazily, and params models need `extra="forbid"` or a misspelled param is ignored.
- Why: the rebuilt-union registry left stale unions on late registration, was superlinear (333 ms for 175 types) and gave 1332-character errors; lookup costs about 15 us more on a 63-node config, once. Without `SerializeAsAny` pydantic dumps a subclass as its base, silently.
- See: [configurable steps](../../architecture/configurable-steps/index.md), `decikit/_registry.py`

## `score` takes a dict and caches converted params bundles

- `score(record, params)` takes one dict; with no frame steps it builds no polars frame. Converted bundles are cached per params document and node, and a reused document is hashed once. Not: positional calls (transposition risk), a reused record buffer as default (brittle, thread-unsafe).
- Why: 400 keyword arguments cost 1190 us, a dict 60 us; converting a validated model costs 170.6 us against 78.3 us to validate it, a cached bundle 0.23 us.
- See: [experiment](../experiments/single-record-calling-convention.md), [engine](../../architecture/engine/index.md)

## One executable serves all threads; shared state is immutable or per thread

- `run` and `score` on a shared `Executable` are safe from many threads, including the first call: caches fill through `once`, the compiled snapshot is published whole, `report` is thread-local, a bundle and its UTF-8 buffer are one `setdefault` entry.
- Why: kernels are `nogil`, and an unsynchronised span buffer was a use-after-free. Sessions and trace sinks are single caller by contract.
- See: [concurrency](../../architecture/engine/concurrency.md), `decikit/_engine/concurrency.py`

## `each` is one scatter-gather primitive; polars vectorises it above 1024 elements

- `each` lowers to a `ScatterGatherNode` with a real IR child; `EachMode.PER_ROW` and `BATCH` are accepted but identical. The polars route is used only on a whole-frame `List(Struct)` column with at least 1024 elements. Not: `prange`, or threading over parent rows.
- Why: per-row cost a flat 170 to 960 us per record and batch about 4 ms fixed; the primitive is 219 to 473 us for one record and 8 us per record at 10k, level with batch. The batch cost is not a numba loop, and a shared child `Executable` raced.
- See: [experiment](../experiments/each-scatter-gather.md), `decikit/_engine/run/runners/interpreted.py`
