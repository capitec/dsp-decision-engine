# Deferred findings from the whole-repo review (2026-09-27)

A full review of `decider/` turned up bugs and design smells. Five were fixed
(see the branch's commits around this note); the rest are recorded here rather
than fixed, each with why it is deferred and what its impact is. They are not
wrong-by-design, just not worth the risk or churn today.

## fingerprint keys object globals by `id()` and modules by name

`decider/engine/compile/fingerprint.py:_value` hashes a non-literal, non-array
global as `object:<qualname>:<id()>` and a module as `module:<name>`. The `id()`
is process-local and the module name carries no version.

- Impact: a step reading a dict/lookup-table global, or whose module changes
  without its file path changing, gets a fingerprint that is stable within one
  process but not across processes or versions. The warmed numba disk cache
  (`decider build`) misses on the serving host, and a dependency upgrade can
  leave stale machine code.
- Deferred because: the docstring already declares this ("modules by name, other
  objects by identity"), hashing arbitrary objects or module source canonically
  is not obviously correct, and the common case (a `set` membership literal,
  hashed per-process because `repr` follows the hash seed) was fixed separately.

## a fused kernel that fails at run time falls back onto kernel-typed values

`decider/engine/compile/units.py:Kernel.run` catches `FALLBACK_ERRORS` from the
fused kernel and re-runs each call's `py_func` over the `values` dict, which at
that point holds kernel representations (int codes for `Raw[str]`, span tables
for `bytes`, record arrays for structs), not the Python values the function
expects. The compile-time fallback in `compile_plan` is correct because the
stepped runner re-derives Python values; this runtime path is not.

- Impact: a step that compiles alone but fails to type together inside a fused
  kernel returns wrong answers or crashes instead of falling back cleanly.
- Deferred because: it only fires when a kernel numba accepted at compile time
  then rejects at call time (an unusual typing edge), reproducing it needs a
  contrived step, and the fix touches the compiler's fallback plumbing.

## the HTTP handler runs the pipeline on the event loop

`decider/serving/handler.py:process_fn` calls `executable.score`/`run`
synchronously inside an `async` method; `session_ws.py` already wraps the same
work in `run_in_threadpool`.

- Impact: concurrent requests on one worker serialize, and a request cannot be
  interrupted. Kernels are `nogil`, so they would parallelise in a thread.
- Deferred because: it is a throughput change with a serving-visible behaviour
  change, not a correctness bug, and needs a load test to justify.

## `_POW10` and the `22`-digit cut in `cpython_round` are coupled by a magic number

`decider/engine/compile/cpython.py` builds `_POW10` as `range(23)` and
`cpython_round` switches at `0 <= ndigits <= 22` / `-22 <= ndigits < 0`.

- Impact: `round(x, ndigits)` changes behaviour discontinuously at `ndigits == 23`;
  a change to one constant without the other silently shifts that boundary.
- Deferred because: it is a latent maintainability trap, not a bug today, and
  the exact-`10**n` cutoff is dictated by float precision, not a style choice.

## unbounded content-keyed caches

`_KERNELS` (`kernel.py`), `_DISPATCHERS`/`_REASONS` (`njit.py`), `_schema_plan`
(`boundary/extract.py`, `maxsize=None`), `RequestHandler._history`
(`serving/handler.py`) and `ParamsCache._results` all grow without eviction.
Each is already marked `# ponytail`.

- Impact: a long-lived serving process that stages many versions or sees many
  distinct params documents leaks memory without bound.
- Deferred because: the leak needs a long-running process to matter, eviction
  policy is a real decision (LRU vs. cap vs. keyed weakref), and each cache has
  a different natural lifetime.

## `os.cpu_count()` may return `None`

`decider/settings.py` computes `os.cpu_count() * 2 + 1` for the default worker
count.

- Impact: `TypeError` on import in a restricted/container environment where
  `os.cpu_count()` is `None`.
- Deferred because: it is a one-line guard and only bites in unusual sandboxes.

## Arrow view leak and signed/unsigned mismatch

`decider/engine/boundary/_arrow/view.py` raises without releasing a
partially-built `ArrowSchema`/`ArrowArray` on a failed import (only the
`rc == 1` branch calls `release()`), and reads the header's column addresses as
signed int64 while the C shim writes `uint64_t`.

- Impact: leaked exports and polars backing buffers on an import failure; a
  negative address (only if one exceeded 2^63, which user-space addresses do not
  today) would wrap to a bogus pointer.
- Deferred because: both are latent (the leak only on the error path, the
  mismatch unreachable on current platforms), and the shim is being replaced.
