# Review: feature/decider-v2..feature/decider-v2-gap-fixes (2026-09-26)

13 commits, 3290/369 lines. Non-tooling scope: `decider/types.py`, `decider/fields.py`,
`engine/run/representations.py`, `engine/compile/rows.py`, `engine/compile/njit.py`,
`engine/run/runners/{stepped,interpreted}.py`, `engine/run/state.py`, `ir/decls.py`,
`steps/helpers.py`. Suite on the branch: **23 failed, 1642 passed, 1 xfailed**.

## Blockers

1. **`stepped._run` asks for a representation for every object column** (`stepped.py:137`).
   `representation_for(float)` raises `TypeError: no representation for
   RepresentationKey(base_type=float, ...)`. Any object-dtype column feeding a non
   str/bytes input hits it — i.e. every missing input and every `missing_as` fill.
   13 failures (`test_inputs_and_score.py`, `test_score.py`, serving's 400).
   `interpreted._argument` has the guard stepped lost:
   `elif base_annotation(decl.annotation) in (str, bytes)`. Same guard in stepped fixes all 13.

2. **`_prepare_function` treats attribute names as callees** (`njit.py:130`).
   It scans `fn.__code__.co_names`, which holds attribute names too, so
   `rates.rate` resolves `globals()["rate"]` → the test module's *step* `rate` →
   "calls 'rate' without @helper or @python_only" → `strict_compile` raises.
   3 param-table failures. Restrict the scan to names actually called (walk the
   bytecode's `LOAD_GLOBAL`+`CALL`, or `dis` the code object).

3. **`_prepare_function` refuses callees numba can already compile.**
   A global with a numba `@overload` (or a bare-imported numba-supported numpy
   function) now falls back with "without @helper". `test_modes_can_be_narrowed`
   returns 2.0 instead of 3.0. The probe should stay authoritative: try the
   compile first, and only use the undecorated-callee text as the *explanation*
   when it fails.

4. **Semantic `str` inputs recompile the kernel per batch.** `_typed` builds
   `np.asarray(values, dtype=f"U{max_width}")`, so the kernel's argument type is
   `Array(UnicodeCharSeq(6))` — keyed on the longest string in the batch. Caught by
   `test_retuning_a_string_literal_never_recompiles`. Also: the probe validates
   `unicode_type`, not `UnicodeCharSeq`, so a step that probes clean can still fail
   at kernel build; and `_typed` maps a null string to `""`, losing the null.
   Needs a fixed representation (width buckets, or keep codes/spans for kernels and
   let semantic-`str` steps be dispatcher-backed per-row fallbacks like `str` outputs are).

## Design defects (no test yet)

5. **`State.representation` is never invalidated** (`state.py:116`). Keyed by
   `(version.id, kind)`, but `State.write` mutates `self.values[vid]` in place, so a
   loop body (same version each iteration) or a per-arm `write(rows=)` serves the
   first build forever — silent wrong answers for `Raw[str]`/`Rows[Item]`/`bytes`
   produced inside a loop. `restore()` keeps entries for kept ids whose values were
   rolled back, same problem after `rewind`. Drop a version's entries in `write`.

6. **`_compile`'s new fallback warning fires for compiled fallbacks too**
   (`stepped.py:85-97`). A `str` output or a `Rows[Item]` input is a `Fallback`
   backed by a real `Dispatcher` — it does *not* run "in Python, row by row", yet the
   message says so and `strict_compile=True` rejects it. Gate the warning on
   `not isinstance(unit.fn, Dispatcher)`, the same test `_run` uses. Also
   `next(c.id for c in plan.calls if ...)` is an O(n²) path lookup.

7. **`raw_str()` codes are process-order dependent** (`types.py:79`). A global
   counter assigns codes in call order, and a code baked into a step body becomes a
   numba constant. `fingerprint` hashes the global's *value*, so the disk cache key
   moves with it — safe, but it also means the disk cache misses whenever import
   order shifts. Derive the code from the string (hash) if the cache should hold.

8. **Layering**: `engine/compile/njit.py` imports `decider.steps.helpers`, which runs
   `decider/steps/__init__.py`. It works today by import order only. Move
   `helper`/`python_only` next to `decider/types.py` (or into `engine/compile/`).

## Intentional, tests need rewriting (5 failures)

`_str_params` now returns `(), None` for scalar nodes, so the "several `str` inputs"
and "`str` param without a `str` input" strict errors are gone, and
`test_a_string_compared_with_a_literal_in_the_body_runs_in_python_when_compiled` no
longer warns. Correct under the new semantics — the tests pin the old rule.

## Good

Gap 7 (`fingerprint(self._py_func)` in the cache key) is the right fix.
`Executable.fallbacks()` is the public surface these reasons needed.
`fields.py` is metadata-only and the engine ignores it.
`rows.py` reuses `param_table`'s flat-arrays-plus-offsets shape.

## Branch history

Only three refs are not ancestors of HEAD: `worktree-agent-a7a1162a3b07a6e78`
(702ee5e, the zero-copy Arrow spike, 1 commit), `worktree-debug-tools`
(tooling — another agent), and `backup/pre-scrub-rewrite` (the deliberate
pre-deletion backup of decider2/experimentation). Every `decider2:` stage
worktree branch is already in history. Still to do: merge 702ee5e with a revert
commit ahead of it so history keeps it at no net change.
