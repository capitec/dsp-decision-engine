# notes/

Decision records for the consolidated `decider`, each with its measured reason.

- `spec-amendments.md`: changes to IR.md and Plan.md agreed with the user after sign-off. Where they disagree with IR.md, this file wins.
- `numba-compile-cache.md`: cached code lives in real files imported by module name, generated names come from a content hash, and the CPU target is recorded and checked with the cache.
- `fusion-is-explicit.md`: kernel grouping is a fixed rule, never a cost heuristic, because fusion is non-monotone and depends on body cost.
- `kernel-flags-nogil-fastmath.md`: serving kernels use `nogil=True` (without it, p99 was 12.7× the budget at 16 threads), and `fastmath` is off by default.
- `money-rounding-and-overflow.md`: money is int64 cents, running totals are float64, and there is no bare `round()`, because CPython and numba round differently.
- `registry-dispatch-by-lookup.md`: nested steps dispatch by tag lookup instead of rebuilt unions, and `StepRef` uses `SerializeAsAny`.
- `structure-in-python.md`: pipeline structure is code, and config carries params and `ConfigurableStep` documents only.
- `origins-out-of-cache-keys.md`: origins are debug metadata, and compiled code and generated names are keyed by content.
- `single-record-path.md`: `score()` takes a dict, not kwargs, and converted params bundles are cached.
- `python-fallback-numbaerror-only.md`: only compile failures trigger the Python fallback, at kernel boundaries, and runtime errors always propagate.
- `tree-documents.md`: v3 trees and flat rules both validate into one graph-shaped `Tree`; v0-v2 are rejected by name.
- `table-rows-as-runtime-arrays.md`: table rows, thresholds and enable flags are runtime arrays, so editing them never recompiles.
- `arrow-shim-build.md`: the boundary's nanoarrow shim is compiled with `cc` on first import into a content-keyed cache; no pure-Python path; a REQUIRED null is an error.
- `interpreted-runner.md`: state holds nullable columns and each reader applies its null policy; row subsets for arms and loops; checkpoints before/after each node; params validated per node at run time.
- `compiled-runners.md`: stepped/fused are the interpreted driver with compiled calls; a fused kernel pauses once with its first step's origin; `str` values are codes from one table per runner; nulls behave the same in every mode; `score()` skips polars.
- `lazy-validation-fused.md`: lazy validation in fused mode validates a kernel's nodes as it launches; no kernel suspend, because a kernel has no control flow.
- `packed-control-flow.md`: in fused mode a branch or loop of plain scalar steps is one kernel with LLVM control flow; the packing rules, when a launch runs unpacked, and throughput against decider2.
- `tree-walker.md`: a tree is one row node walked by one njit function over packed arrays passed by address, with an independent Python reference; string outputs are `Literal` indices; strings are matched as byte spans in the kernel, regex/case/trim in Python.
- `wiring-scopes.md`: how names bind to versions: input columns, typo and forward-reference errors, branch/loop scopes, frame barriers, outputs.

`progress.md` will hold the task log.
