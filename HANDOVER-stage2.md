# Handover — Stage 2 of the boundary rework (strings at the node)

Branch `worktree-agent-a9975ac0810282b0c`, based on `b88b116` (Stage 1b).
Commits: `78091fd` (the whole Stage 2 change, WIP) and the commit that
carries this file (three test-expectation fixes in
`decider2/tests/test_trees_strings.py`). The diff is the source of truth
for *what* changed; this file records what only I know: why, what bit,
and where I stopped.

Design read: `decider2/docs/BOUNDARY-REWORK.md` §1.3b, §1.6, §2, §3.1,
§6, §8, §9 (Stage 2 rows). Build/venv notes at the bottom.

## Where I stopped, precisely

I had just run the affected test files (17 files) once:
**255 passed, 34 failed**, every failure in the NEW
`tests/test_trees_strings.py`. I diagnosed all 34 as ONE wrong
expectation in my own test, not a kernel bug: a REQUIRED-null row is
routed and `apply()` writes an int64 terminal back as the `0` placeholder
that `runtime/invoke.py::_scatter_back` documents ("an int64/bool
terminal has no NaN equivalent, so it gets 0/False"), never `None`. Every
failing parametrisation differed only at index 0 (the `None` row):
`0 != None`. I fixed the three expectations (`[0 if e is None else e ...]`,
`[0, 0]`, `[1, 0, 1]`) and launched a re-run of that one file in the
background. RESULT, seen after writing the rest of this file: **55 passed** in 104 s. Nothing else in the 17-file run
failed — the inverted refusal tests, the rewritten typed-features tests,
`test_shim.py`, serving, CLI, boundary all passed.

The FULL suite has not been run since the change. Baseline before the
change: **596 passed** in `.venv-stage2` (with the extension), 270 s.

## Design decisions, and the option rejected each time

1. **A node's literal patterns are ONE `list[str]` param, not one `str`
   param per pattern.** The design text (§3.1, §2.2) says one `str`
   `ParamDecl` per pattern with a chain of STR nodes. I rejected that
   because §6 item 3 and acceptance item 5 demand "change the pattern
   COUNT, assert zero compiles and the SAME `Driver`" — which is only
   satisfiable if the count is a *params* value. A new tree document
   always builds a new `path_fn` closure over new structure arrays (the
   numba cache key hashes closure cell contents) and a new per-process
   packed kernel, so a document edit can never be zero-compile. With the
   list param, `params={"t": {"root_patterns": ["a","b","c"]}}` on the
   same pipeline is measured at **0 compiles, same Driver** (see
   `test_editing_adding_and_removing_patterns_never_recompiles`). An
   `InputRef` pattern stays a `str` param named by its key (a group of
   one) so `test_cases_string_match_inputref_pattern` still passes
   unchanged with `params={"vip_tier": "gold"}`.
   Consequence: a STR node tests a pattern GROUP; the table is
   `(pat_bytes uint8[:], pat_off int64[:], grp_off int64[:])`, not the
   design's `int64[:, 2]` bounds. Param names: `<node_id>_patterns`
   (unary), `<node_id>_patterns_<i>` (i-th condition of a Cases node or
   composite). No test pinned the old `<feature>_pat_<n>` names.

2. **Spans in the registry are an `(n, 2)` int64 table per `bytes`
   input, materialised once per batch by `sm_gather_row` per row
   (`FrameView.spans()` → `materialize_spans`, new in `_arrow/frame.py`).**
   Rejected: threading `(gather_addr, plan_addr)` into the typed kernel
   and gathering inside the loop. That would have changed
   `_build_typed_kernel`'s calling convention (Stage 3 is editing the
   packed-input half of `gather.py` concurrently) and mixed two gather
   paths for one row (numerics still come from numpy columns in Stage 2).
   The table keeps the kernel purely array-based; Stage 3 replaces it.
   Cost: one extra n×2 int64 allocation per string column per batch.

3. **`SpanPlan` holds the `FrameView` in a pydantic `PrivateAttr`
   (`_view`)** so the addresses stay valid while the `ExtractedColumn`
   (hence `ExtractedFrame`, hence the `apply()` call) is alive. Rejected:
   a `keepalive` field on `ExtractedColumn`/`ExtractedFrame` (Stage 3's
   file, bigger diff) and copying the bytes (defeats zero-copy). It is a
   hidden lifetime; the docstring says so. `release()` happens in
   `FrameView.__del__`.

4. **`load_u8` moved to a new extension-free module
   `_arrow/intrinsics.py`** (re-exported from `_shim`) so
   `import decider2.trees` does not load the compiled extension. Only
   `apply()` on a string tree needs it (through `SpanPlan`); `score()`
   never does; numeric trees never do. Rejected: duplicating the 8-line
   intrinsic in `trees/interpreter.py`.

5. **`resolve_params` skips `_resolve_str_param_code` for `typed_args`
   steps** (their `str` params are bytes patterns, no dictionary). The
   hand-written-step `str` path (`record is its own dictionary`,
   `CodesPlan`) is untouched — Stage 7's surface, exactly as §1.6/§7 say.

6. **`_assert_score_agrees_with_apply` now skips only when a NON-packed
   step reads a `str` input** (keyed on the step, per §8.2 item 1). A
   tree's `bytes` input is always compared. I did NOT try removing the
   skip entirely for hand-written `str` steps; my reading is they would
   pass (`==` on codes agrees with `==` on one-entry dictionaries), but it
   is unmeasured — see "next steps".

7. **Empty-pattern semantics follow Python `str`:** `""` is exact-equal
   only to `""`, but is a prefix, suffix and substring of everything.
   `match_bytes` documents this. An `InputRef` pattern's default is `""`,
   so an un-tuned `starts_with` InputRef matches every row (pinned in
   `test_an_inputref_pattern_is_a_str_param_retuned_by_its_key`). That is
   faithful to Python and to the old default, but it is a footgun worth a
   sentence in docs.

## What in the brief / design turned out wrong or harder

- **§6 item 3 / acceptance 5 contradict §3.1's "one `str` ParamDecl per
  pattern"** — see decision 1. The design should say "a node's literal
  patterns are one `list[str]` param".
- **§1.6 "convert back to Python types before calling step functions"
  does not apply to trees.** A tree's step function IS the njit
  `path_fn`; interpreted mode calls it with the same spans. There is no
  Python-typed form to convert to. The independence property lives at
  the `score()` rung exactly as §1.6 says, and the injected-drift test
  proves it: corrupting `SpanPlan.extract` leaves interpreted≡stepped≡
  fused agreeing and fails `fused↔score`.
- **The design's `_TYPED_DUMMIES` trick needs a 2-D STR dummy** (shape
  `(0, 2)`), or the per-kind tuple is heterogeneous. Easy to miss.
- **`np.frombuffer(bytes)` is read-only; the offsets arrays must be made
  read-only too**, or the pattern table's numba type differs between
  the empty and populated cases and you get a second specialisation on
  the first retune. `_pattern_table` enforces this and the test
  `test_a_pattern_count_change_and_a_threshold_change_share_one_kernel_signature`
  checks the `typeof` is identical for 1 vs 3 patterns.
- **Routed rows write `0` into an int64 terminal, not null** — the
  pre-existing `_scatter_back` placeholder. My corpus test assumed
  `None`; that was the 34 failures. Any future string test with a null
  row must expect `0`.
- **A `bytes` input needs `_dummy_value`/`_warmup_value`/`corpus._base_
  type` to produce a `str`**, or `precompile()`/`corpus()` build a
  Float64 column for it and `FrameView.bind` raises `ArrowKindError`.
  Done in `graph/pipeline.py` (one line — a Stage 4 file),
  `testing/recompile.py`, `testing/corpus.py`.
- **Absent `bytes` column**: `_synthesize_absent_column` builds
  float64 zeros; a typed kernel then sees a 1-D array in the STR slot and
  fails typing even at n=0. Special-cased in `boundary/extract.py` to an
  `(n, 2)` `[0, -1]` table.
- **The shell guard in this environment rejects heredocs and `$(...)`**;
  file edits had to go through the Edit/Write tools. `uv pip install`
  refused `decider2[dev]` and `$PWD`; spell paths out.

## Traps hit

- System CPython 3.14 has no `Python.h`; the uv-managed
  `~/.local/share/uv/python/cpython-3.14-linux-x86_64-gnu/bin/python3.14`
  does. `.venv-stage2/` in the worktree root (gitignored? — check; I did
  not add it) was built from it: `uv venv --python <that> .venv-stage2`,
  then `VIRTUAL_ENV=<abs path> uv pip install -e ./decider2 pytest
  starlette uvicorn httpx2`. `httpx2` is needed by `test_serving.py` and
  is not a declared dev dependency (pre-existing gap).
- `driver.py` re-exports every `gather.py` name; deleting `_BYTES_DUMMY`
  needed one line removed from that import list (the only `driver.py`
  change).
- `test_typed_features.py` imports `_packed_row_args`/`build_packed_kernel`
  from `compile.driver` — the re-export path — so the typed-row shape
  change surfaces there.

## Acceptance items — exact status

1. Full suite green — **PARTIAL.** 17 affected files: 255 passed, 34
   failed before the expectation fix; re-run of the new file launched,
   result unseen. Full suite not run since the change. `test_flagship.py`
   and `test_spec_conformance.py` untouched (never edited).
2. Four match types on a real String column vs a Python reference over
   the §8.2 corpus — **DONE in code, verification PARTIAL** (the test
   exists and its only failure was the null-row expectation; hand smoke
   test of `contains` passed in all four rungs).
3. Frame-shape corpus byte-identical — **DONE in code, verified in the
   17-file run** (`test_frame_shapes_give_byte_identical_outputs` passed).
4. `assert_equivalent` all four rungs incl. `score()` for string inputs
   — **DONE and verified** (`test_assert_equivalent_drives_all_four_rungs
   _over_the_generated_corpus` passed, counting one `score()` per corpus
   row). Unskipping exposed nothing; hand-written `str` steps keep the
   skip (decision 6) and were not tried without it.
5. Adding a pattern is zero compiles — **DONE and verified** (0 compiles,
   same `Driver`, for text edit / 1→3 patterns / 3→1 with a 5000-byte
   pattern; hand smoke test also 0). Caveat: via the `list[str]` param
   (decision 1); a document edit still compiles, inherently.
6. Injected-drift test at the `score()` rung — **DONE and verified**.
7. `regex`/`case_sensitive=False`/`trim_whitespace` still refused, old
   refusals inverted — **DONE and verified** (`test_trees.py`,
   `test_trees_ported_conditions.py` ×2, all passing).

Not done: BOUNDARY-REWORK.md not updated with a "Landed (Stage 2)" note
or the list-param deviation; `tests/PORTED.md` not updated; the
`_arrow/__init__.py` docstring still says "Nothing in decider2/ imports
this package yet". No benchmark was run (not required; `free -h` showed
16 GB available if one is wanted).

## Next steps, in order

1. Read the background result of `pytest tests/test_trees_strings.py`
   (or re-run it, ~5 min). Expect green.
2. Run the full suite in `.venv-stage2` (~5 min) and compare with 596.
   Expected delta: +~65 new tests, −0. Watch `test_boundary_plans.py`
   (I added `SpanPlan` to the `ColumnPlan` union; a test enumerating
   union members would notice) — it passed in the 17-file run.
3. If green, amend/commit, then add the doc notes above (BOUNDARY-REWORK
   §7 Stage 2 "Landed" paragraph stating decision 1; PORTED.md lines for
   the three inverted tests).
4. Optional: try `_assert_score_agrees_with_apply` with NO skip at all
   over `test_spec_conformance.py`'s `sector_rate` shape to see whether
   hand-written `str` steps already agree; report either way.
5. Optional: the `""`-InputRef-default footgun (decision 7) — consider
   whether an un-tuned InputRef pattern should match nothing rather than
   everything for prefix/suffix/contains; that is a semantics decision
   for the owner, not a bug.

## Things I suspect but have not verified

- `SpanPlan` on a frozen pydantic model with `PrivateAttr`: works in the
  runs above, but a `model_copy()`/pickle of an `ExtractedColumn` would
  drop or fail on the view. Nothing does that today.
- `FrameView.__del__` releasing polars' buffers relies on GC timing; if a
  caller keeps `registry["s"]` (the span table) beyond `apply()` the
  addresses dangle. Same class of risk the design's §10 item 2 names;
  nothing in `src/` does it.
- `_typed_params` is called per ROW in interpreted/stepped mode
  (`_packed_row_args`), re-encoding patterns each row. Correct, slow,
  pre-existing shape (those modes were already O(n) Python).
- The walker's `feat_kind` for LEAF rows now reads the kind of feature 0
  (was F64 when feature 0 was float); leaves never read it. Harmless,
  but `test_an_undeclared_tree_encodes_exactly_as_before`'s byte-identity
  claim holds only for numeric-only trees (it does — it passed).
- I did not re-run `test_a_typed_tree_path_fn_is_a_genuine_cache_hit_in_a
  _fresh_process` in isolation after the walker change; it was in the
  17-file run and passed (numeric tree). A string tree's `path_fn` cache
  hit in a fresh process is NOT separately tested — worth adding
  (`_CHILD` in `test_typed_features.py` is the template).
