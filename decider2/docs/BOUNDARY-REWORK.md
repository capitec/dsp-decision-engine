# The boundary rework — polars → nanoarrow → numba, and the row that crosses it

*Design and staged plan for the change the owner asked for: "adopt nanoarrow
with its unsafe accessors, trusting polars as the producer … this would redesign
how trees, tables and the whole engine polars→numba boundary works as a whole."
Nothing in `decider2/src/` is modified by this document. Every claim about
current behaviour cites a file and line; every number is from a measured
prototype (`experimentation/nanoarrow-accessors/`,
`experimentation/arrow-strings-in-tree/`, the two branches named in §0.3) or
from the probes written for this design
(`experimentation/boundary-rework-design/`, numbers in its README). Where a
prototype's conclusion was refuted by a probe, that is said in place. Line
numbers are from the typed-features worktree (`bffa3fa`, the post-Stage-0
code) unless marked "(main)"; files the branch did not touch —
`runtime/invoke.py`, `graph/pipeline.py`, `boundary/*`, `testing/*` — are
identical in both.*

---

## 0. Summary — one page

**What changes.** A polars `DataFrame` stops being taken apart column by
column into numpy arrays and dictionary codes. It crosses the boundary **once**,
as one Arrow struct through polars' own `__arrow_c_stream__()`, is decoded by
vendored **nanoarrow 0.9.0** behind a ~150-line C shim (the project's first
compiled artefact), and reaches every compiled walker as a **typed row**: five
scalar buffers `(f64[:], i64[:], bool[:], int32[:], span[:])` where a string
feature is a `(address, length)` pair pointing straight into polars' memory
(`length == -1` is null). A tree tests a string at the node — exact, prefix,
suffix, contains — against a pattern table that is a kernel *argument*.
`cast(pl.Categorical)`, `CodesPlan`, the hoisted `__m_<column>` matcher steps,
`_resolve_str_param_code` and "the record is its own dictionary"
(`runtime/invoke.py:752-763`) all go. The `types.py` seam keeps every name; one
documented shape changes (the typed row tuple, §2).

**Trusted producer, settled.** The kernel reads through
`ArrowArrayViewGetStringUnsafe` / nanoarrow's decoded `buffer_views`; no
bounds check is written by this project, in any language, on the hot path.
A corrupt Arrow buffer therefore crashes or answers wrong
(`nanoarrow-accessors/RESULTS.md` §6). The premise is that the producer is
polars, which validated the data when it built it. This document does not
reopen that.

**What it costs.**
- A compiled extension: nanoarrow amalgamation (9,183 lines, Apache-2.0) +
  shim, `gcc -O2` 1.6–1.9 s, 76–81 KB, no dependency beyond libc; one abi3
  wheel per platform (5 wheels + sdist), cibuildwheel in CI. No wheel and no
  compiler → the install fails, by design (§4).
- Per row, batch: the C gather costs **+2–3 ns per column per row** over
  today's numpy `_fill_array` (328 vs 284 ns/row on a 17-column tree at 1M
  rows; the walk dominates both). No crossover to worry about: the string
  path is *cheaper* than today's `cast(Categorical)` at every size
  (`arrow-strings-in-tree/RESULTS.md` §3).
- Warm-start: +0.2 s per process (the inlined walk compiles inside the
  per-process packed kernel, EXPERIMENTS §X on the inline branch).
- Four test files over the old ladder are rewritten; the tree interface loses
  its synthetic `<tree>__m_<column>` output columns (§9).

**What it buys.**
- Correct: an Int64 above 2⁵³ compares as an int64 (typed-features branch,
  doc 03 §1.2); a string compares as its bytes, never as a batch-specific
  integer; `sector < 5` on a string is a build error.
- Capable: `starts_with`/`ends_with`/`contains`/`isin` at the node, lazily
  (gated at 1%: 3–6× ahead of the frame-tier workaround), no precomputed bool
  columns; changing a pattern or adding one is a value change (today adding a
  literal changes a tuple *type* and recompiles, §6).
- Simpler: the dtype ladder collapses from five plan classes and 26
  dtype×nullability cases to "Arrow type → kind"; nullable numerics stop
  copying (today ~550–630 µs/100k rows per column, doc 05 §1.5); Boolean,
  Date, Datetime and Duration stop copying; polars' chunk alignment problem
  disappears because a frame-level export is one aligned struct (measured, §1.4).

**The single-record number.** The Arrow boundary is **not on the `score()`
path** and the plan does not pretend it is. `score()` takes a dict and never
builds a frame (`runtime/invoke.py:712`); a 17-column `pl.DataFrame([record])`
alone costs 200–260 µs, above the whole 60 µs budget. Measured this session on
the flagship pipeline: `score()` p50 **409 µs**, `apply(1-row)` **1352 µs**, and
a profile shows ~85 % of `score()` is schema-invariant Python redone per call —
`inspect.signature` with annotation `eval` per step per call in
`compile/driver.py:675-711` (main: 485-511), the interface re-walk with difflib in
`graph/pipeline.py:410-510`, `flatten_for_runtime` — with the kernel at ~1.4 µs
(EXPERIMENTS §N1). Stage 4 hoists that into a per-pipeline plan and marshals
the record straight into the same typed row (a string becomes
`value.encode()` + address; no Arrow, no polars). Target: **≤ 60 µs** on the
flagship, estimated 20–40 µs. For `apply()` on a one-row frame the boundary
itself goes from 122 µs (17 columns, numpy) to ~40 µs (one Arrow handshake +
one C call), of which 27 µs is polars building the struct export.

---

## 0.1 The decision this document takes as given

"If the only risk is corrupt data and we are working off the premise that
inputs will be polars, then we should trust that the data polars uses is
correct and use unsafe, because polars has already validated it in a sense."
That is Arrow's own trusted-producer model; `RESULTS.md` §6 of the nanoarrow
strand recorded exactly what it gives up (SIGSEGV on 2/5 constructed
corruptions, silent wrong answer on 2/5). The one thing this document adds is
that **Arrow C++ does check what nanoarrow does not** (Appendix A) — which is a
report to file upstream, not a reason to change course.

## 0.2 What was refuted while designing this

1. **"Use nanoarrow's accessors per element" is the wrong grain.**
   `ArrowArrayViewGetDoubleUnsafe` switches on storage type per call; 17 of
   them per row cost 374 ns/row against 284 for numpy (probe V1). Resolving
   each column once per batch from nanoarrow's *decoded* view
   (`buffer_views[1].data`, `offset`, `storage_type`) and then doing a plain
   load per row (probe V2, 328 ns/row) is the shape. Still zero hand-decoding:
   the shim reads nanoarrow's fields, never polars' bytes.
2. **Per-Series import is the wrong unit.** The prototypes imported one
   Series at a time and had to *check* chunk alignment across columns and
   refuse (`arrow-strings-in-tree/kernel.py:245-254`). A `DataFrame` exports
   as one `+s` struct whose children are aligned by construction, and polars
   1.41 rechunks the frame in place while exporting (probe: 2/2/1 chunks →
   one struct, `n_chunks` 1/1/1 afterwards). The alignment problem does not
   exist at the frame level. Cost: ~1.5 µs per column at n=1 in polars' own
   export, whether per Series or per frame.
3. **"The Arrow path improves the single-record number" — no.** §0 above.
   `score()` never touches polars, and the 352–1548 µs of §R is in the graph
   and driver layers, not the boundary. The rework's single-record win is
   Stage 4, and it is a Python change.
4. **The typed-features branch's `_get_buffers()` string slot is the wrong
   buffer.** It copies (17–28 ns/row, `verification-probes/README.md`
   "probe_arrow_zerocopy"); the C Data Interface does not. Its *slot* is
   right and is kept (§2).
5. **The hand-decoded numba kernel as the shipped fallback** — assessed and
   rejected for the package (§4.4): it works, but it is a second decoder with
   42 lines of layout knowledge exercised only on the platforms nobody tests.

## 0.3 Inputs this design builds on

| what | where | status |
|---|---|---|
| nanoarrow shim, `@intrinsic` pointer-as-argument call, cache proof, build cost | `experimentation/nanoarrow-accessors/` (`nashim.py`, `c/nashim.c`, RESULTS §7–8) | basis of §1, §4 |
| STR node (exact/prefix/suffix/contains), 38 tests incl. nulls, UTF-8, sliced, multi-chunk, zero rows | `experimentation/arrow-strings-in-tree/` (`kernel.py`, `test_kernel.py`) | basis of §3.1, §8 |
| typed feature arrays, `feat_kind`, build-time rejections, 563 tests | branch `worktree-agent-a47fc4a23192c7421` (`bffa3fa`), `TYPED_FEATURES.md` | basis of §2; its raw-string slot is re-pointed |
| `inline="always"` on walkers, Branch/Loop 22 %, §X | branch `worktree-agent-a4e1be627b635397e` (`b7dfc89`) | merged in Stage 0 |
| polars plugin: the ~1 µs kernel / ~235 µs plumbing decomposition at n=1 | `experimentation/polars-plugin-trees/RESULTS.md` §2a | corroborates §0's single-record reading |
| verification probes (refcount trap, per-category mask condition, zero-copy witness) | `experimentation/verification-probes/README.md` | §2.3, §3.4 |
| probes for this design (frame import, gather shapes, chunking, pyarrow) | `experimentation/boundary-rework-design/` | §1, Appendix A |

---

## 1. The new boundary

### 1.1 Shape

```
polars DataFrame
   │  .__arrow_c_stream__()                the WHOLE frame, one capsule; polars rechunks in place
   │                                        (never .select(): 400 µs of planner even at n=1 — measured;
   │                                         df[cols] is 20 µs and is the fallback for a very wide frame)
   ▼
struct ArrowArrayStream ──► shim.import_frame() ──► struct ArrowArrayView (nanoarrow)
                                                        children[k] = one view per column
   │  shim.resolve_all(view, kinds, slots, child_idx)   ONE C call per batch
   ▼
RowPlan (C struct, per batch):  ColDesc[ncols] = {data*, validity*, offset, width, kind, slot, view*}
                                + the five row buffers' addresses
   │  per row: shim.gather_row(plan, i)     ONE C call per row (a function-pointer ARGUMENT)
   ▼
typed row  (f64[:], i64[:], b8[:], i32[:], span[:])   → walk_tree / scan_table / walk
```

Every address the kernel sees — the shim's function pointers, the `RowPlan`,
the row buffers — is a `uint64` **argument**. Nothing process-specific is
captured (EXPERIMENTS §W; cache proof in `nanoarrow-accessors/RESULTS.md` §7,
re-proven in Stage 1's acceptance test).

### 1.2 Which columns import through Arrow, and how each dtype lands

All declared inputs of the pipeline (`interface.inputs`, the same set
`extract_frame` walks at `boundary/extract.py:213-224`). The ladder of
`boundary/dtypes.py` (five `ColumnPlan` classes, `plan_column` at
`dtypes.py:315-376`) becomes one table keyed on the Arrow type nanoarrow
reports (`view->storage_type`), decided once per (pipeline, frame schema):

| polars dtype | Arrow (measured, polars 1.41.2) | kind | per row | copies? |
|---|---|---|---|---|
| Float64 / Float32 | `g` / `f` | F64 | load | no |
| Int64/32/16/8, UInt64/32/16/8 | `l`,`i`,`s`,`c`,`L`,`I`,`S`,`C` | I64 (widened in the gather) | load | no |
| Boolean | `b` (bitpacked) | BOOL | `ArrowBitGet` | **no** (today: always copies, doc 05 §1.4) |
| Date / Datetime / Duration / Time | `tdD`, `tsu:`, … (int32/int64) | I64 | load | **no** (today: `AS_INTEGER` copy) |
| nullable any of the above | validity bitmap on the child | same | `ArrowBitGet(validity, j)` | **no** (today: `COPY_VALIDITY`, ~550–630 µs/100k rows) |
| String | `vu` (Utf8View) | STR | `ArrowArrayViewGetStringUnsafe` → `(addr, len)` | no |
| Categorical / Enum | dictionary: `I`/`C` indices over a `vu` dictionary | CODE (index) + the dictionary view | load index | no (Stage 6; until then today's `CodesPlan`) |
| Decimal(p, s) | `d:p,s` (int128) | — | frame-tier cast to scaled int64 (doc 03 §1.2) **before** export, then I64 | one cast, as today |
| List / Struct / Array / Object / Null | `+l`, `+s`, … | — | `KernelSplitPlan` as today (`dtypes.py:373-376`) | — |

A column whose Arrow type does not match its declared kind (a Float64 column
declared `int`) is cast in the frame tier before export — one explicit,
reported polars cast, never a silent per-row truncation (today
`_as_readonly` does `astype` at `driver.py:293-298`). `explain_boundary()`
keeps its job (doc 05 §1.5 "what the framework owes the author") and reports
the Arrow type, the kind and whether a frame-tier cast was inserted.

### 1.3 What the handshake costs and when it is paid

| step | cost | paid |
|---|---|---|
| plan: kinds/slots/child indices, ctypes structs, row buffers | µs | once per (pipeline, frame schema); the `FrameView` object is pooled per pipeline instance (**per thread** — see §10 risk 2) |
| `__arrow_c_stream__()` of the whole frame | 3 µs + ~1.5 µs/column of the *frame* (polars building the struct export; 27 µs at 17 columns, n=1). Child indices for the declared inputs are looked up in the per-schema plan (keyed on `tuple(df.columns)`); `df[cols]` (20 µs at 17 columns, no planner) only when the frame is more than ~2× wider than the input set. `df.select()` is refuted: ~400 µs at any size, it goes through the lazy planner. | per call |
| `import_frame` (schema + first chunk + `ArrowArrayViewInitFromSchema` + `SetArrayMinimal`) and release | ≈ polars' own cost; 25–27 µs total at 17 columns incl. the line above | per call |
| `resolve_all` | one ctypes call, ~1–2 µs | per call |
| `gather_row` | ~3 ns call + ~2 ns/column | per row |

Today's equivalent — `_get_buffers()["values"].to_numpy(...)` per column plus
`cast(pl.Categorical)` for every String — is 122 µs at 17 columns, n=1
(probe). The Arrow route is 3× cheaper at one row and it is *flat* in rows
(O(columns), never O(rows); `verification-probes/README.md`).

### 1.4 Chunking

Refuted assumption: "columns can have different chunk layouts and the kernel
must refuse or realign". At the **frame** level polars exports one aligned
struct and rechunks the frame in place doing so (probe `probe_chunks.py`,
cases (a)). That in-place rechunk is the same copy `extract_frame` already
pays via `rechunk_once` (`boundary/extract.py:203`) — it moves, it does not
add.

The design still walks struct chunks with a row cursor (`import_frame`
returns "more chunks" when a second `get_next` yields an array, as
`sm_import_single` does at `nanoarrow-accessors/c/nashim.c:48-55`), because a
future polars may stop rechunking. Children of one struct chunk are aligned
by the Arrow spec, so there is no alignment check to write.

A sliced frame carries its Arrow `offset` on each **child** (probe (b):
`offset=3` on children, 0 on the struct). `resolve_col` takes
`view->offset` from nanoarrow; the gather indexes `offset + i`. Stage 1's
tests slice at four offsets against a column with nulls, as
`arrow-strings-in-tree/test_kernel.py` does.

### 1.5 Nulls, per kind

- **REQUIRED**: routed at the frame level before export, unchanged
  (`boundary/nulls.py:311-379`, `extract.py:204-210`).
- **MISSING_AS / NOT_APPLICABLE_AS**: the fill value rides in the `ColDesc`
  and is applied in the gather when the validity bit is clear — no
  `np.where` copy (`nulls.py:252-255` today). `FillInfo.filled_count` is
  computed from the bitmap (`null_count` on the child) rather than a mask.
- **OPTIONAL**: the gather writes `valid[c]`; a packed step's `Optional`
  is `valid[c]`; an ordinary fused step keeps its `__valid__` bool array,
  built by `np.unpackbits` over the bitmap (a small copy, only for OPTIONAL
  columns, as `validity_mask` copies today at `nulls.py:93-99`).
- **STR**: `len == -1`. A null string never matches any string test; a
  `CMP` on a STR feature is a build error (already: `trees/encode.py:353-364`).
- **Sentinels are gone**: no NaN-under-null, no garbage values read by
  accident (doc 05 §2's "leftover garbage" trap is closed by construction:
  the gather never reads a value whose bit is clear).

### 1.6 Ordinary (hand-written) steps

`compile/kernel.py`'s fused kernel indexes whole columns, `cols[j][i]`
(`kernel.py:272-277`), and this rework leaves that calling convention alone.
Its numeric columns become numpy views over the **resolved** Arrow addresses
(`np.frombuffer` over `(c_double * n).from_address(addr)`, 1.2 µs/column
measured, no copy; today's `to_numpy(allow_copy=False)` is 3.6 µs and
`np.ctypeslib.as_array` 5.5–22 µs, neither is used),
taken **after** the export, never before (polars may rechunk during export).
A `str`-annotated input of an ordinary step keeps the dictionary-code
convention (`_NUMBA_BY_ANNOTATION[str] = int32`, `driver.py:80-91`) until
Stage 7; that is the one consumer for which `CodesPlan` survives, and it is
named as such (§5).

### 1.7 `Categorical` / `Enum`

They arrive already encoded: uint32 (Categorical) or uint8 (Enum) indices
plus a `vu` dictionary (probe (c)). Two facts decide their treatment:

- A string test on an already-encoded column is the **per-category mask**
  case (EXPERIMENTS §T: 33.8×; `verification-probes` 0.94 ns/row) — *valid
  precisely because the column arrives encoded*, the condition the
  verification probe warned to re-check. It holds here.
- The dictionary is only known per batch (Categorical) or per schema (Enum).

So (Stage 6): a STR node on a CODE feature is compiled as
`mask[code]`, where `mask` is a per-call bool array built by running the
*same* STR matcher over the dictionary view (n_distinct rows, the dictionary
imported like any string column). A value change, never a recompile. Until
Stage 6 these columns keep today's `CodesPlan` path unchanged
(`dtypes.py:141-163`), so nothing regresses while String moves first.

---

## 2. The row representation

### 2.1 The typed row

Unifies `TYPED_FEATURES.md` §2.1 with `arrow-strings-in-tree`:

```
FeatureKind:  F64=0  I64=1  BOOL=2  CODE=3  STR=4          (types.py:23-48, values unchanged)
row  =  (f64: float64[nf], i64: int64[ni], b8: bool[nb], i32: int32[nc], span: int64[2*ns])
span[2j] = address of the bytes of string feature j on this row
span[2j+1] = byte length, or -1 for null
```

Changes against the branch: `span` holds `(address, length)` instead of
`(start, end)` offsets into a whole-column byte buffer, and the sixth tuple
member (`sbytes`, the byte buffer) is dropped — the address *is* the
locator. Every helper that read the sixth member (`_typed_input_arrays`
`driver.py:301-318`, `_fill_spans` `:345-354`, `_typed_row_args` `:357-372`,
`_build_typed_kernel` `:428-481`, `walk_tree` `trees/interpreter.py:147`,
`path_fn` `trees/encode.py:861-870`, the test
`test_the_raw_string_slot_carries_polars_bytes_zero_copy`) changes with it.
`_TYPED_DTYPES` (`driver.py:260-266`) loses its STR line's comment, not its
entry.

Why `(address, length)` and not a view index: the walker matches bytes with
scalar loads (`arrow-strings-in-tree/kernel.py:75-110`), and the measured
rule from three strands is **never pass an array into a per-row helper**
(`kernel.py:62-73` there; `verification-probes` "NRT (4, 11)"). Two int64s per
string feature is the smallest row state that lets the node read bytes
without touching an array.

### 2.2 What a `Step`'s `(args, params)` becomes

`types.Step` keeps every field (`types.py:83-153`). For `typed_args=True`:

- `args` = the five-tuple above (was a six-tuple; the `Step.typed_args`
  docstring at `types.py:100-112` is the one documented shape that changes).
- `params` = `(floats, ints, pat_bytes, pat_bounds)` — the two homogeneous
  tuples of today (`_typed_params`, `driver.py:321-342`) plus a `uint8[:]`
  byte table and an `int64[:, 2]` bounds table holding every `str`-annotated
  param of the step in first-appearance order (`pattern_table`,
  `arrow-strings-in-tree/kernel.py:196-204`). `_typed_params` stops raising
  on a `str` param (`driver.py:337-341`). The numba type of the pair is
  fixed regardless of how many patterns or how long, so **adding, removing or
  editing a pattern is a value change** (§6).

Annotation → kind stays `feature_kind()` (`types.py:51-60`): `float`→F64,
`int`→I64, `bool`→BOOL, `str`→CODE, `bytes`→STR. A tree feature declared
`str` in `feature_types=` is emitted by the encoder as a `bytes`-annotated
path-step input (the wire type), and by the boundary as a STR column. The
name `bytes` is the branch's reserved spelling and is kept so `types.py`
does not grow a new annotation vocabulary; a later cleanup may rename it.

Non-typed packed steps (`packed=True` without `typed_args`: tables' row/out
steps, Branch/Loop, the single-input matcher) keep the one-float64-array /
`raw1` convention (`_packed_args_kind`, `driver.py:158-187`) untouched in
Stages 1–4.

### 2.3 What stays load-bearing from the branches

- `inline="always"` on `walk_tree` and the cached `path_fn`
  (`trees/interpreter.py:118`, `trees/encode.py:838`), on Branch/Loop `walk`
  and `fn_*` (inline branch), and the rule stated in both: **no non-inlined
  layer between the per-row kernel and the walker, and no array into a
  per-row helper.** The C gather obeys it: its arguments are two integers.
- `_fill_array`'s loop over a homogeneous tuple (`driver.py:229-247`) is
  replaced for typed steps by the C gather; it stays for the float64
  convention.
- The refcount finding (`verification-probes` A/B 1.04×) — decider2's
  structure arrays are loop-invariant, so their NRT traffic hoists — still
  holds; the typed row buffers are hoisted above the loop exactly as
  `_build_typed_kernel` does today (`driver.py:440-455`).

---

## 3. Trees, tables, Branch and Loop

### 3.1 Trees — move now (Stage 2)

Walker signature, from `trees/interpreter.py:118-122`:

```python
@njit(inline="always")
def walk_tree(feats, thr_f, thr_i, pat_bytes, pat_bounds,
              kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc)
```

`feats` is the five-tuple; `pat_bytes`/`pat_bounds` are the pattern table
(per call). Node kinds: `LEAF, CMP, IS_TRUE, IS_FALSE` (`interpreter.py:112-115`)
plus **`STR = 4`** with `op ∈ {EXACT, PREFIX, SUFFIX, CONTAINS}` and
`thr_slot` = pattern index. The STR branch is `arrow-strings-in-tree/kernel.py:166-189`
minus its bounds checks and view decode: `addr, ln = span[2j], span[2j+1]`;
null → `else_`; else `_match_at(addr, ln, pat_base + bounds[p,0],
bounds[p,1]-bounds[p,0], op)`. `_match_at`/`_eq_at`/`_load_u8`
(`kernel.py:38-110`) come across unchanged; they are byte comparison, not
Arrow knowledge.

Encoder (`trees/encode.py`):
- `encode_string_match` (`:555-639`) stops raising for
  `starts_with`/`ends_with`/`contains` (`:583-594`) and stops hoisting a
  matcher step (`:620-639`); it appends one STR node per pattern in a chain
  exactly as `_encode_isin` chains `CMP(EQ)` (`trees/schema.py:745-774`),
  each pattern registered as a `str` `ParamDecl` via `_add_param`
  (`:469-495`) so it is retunable.
- It **keeps raising** for `regex`, `case_sensitive=False` and
  `trim_whitespace=True` (`:595-608`) — §5 says why, and the messages
  change to name the frame-tier workaround only for those three.
- `_slot_kind` (`:296-306`) maps a `str` feature to STR instead of "matcher
  output in I64"; the `via_matcher` flag on `add_cmp` and the
  `_StringMatcher` class go.
- `test_an_undeclared_tree_encodes_exactly_as_before`
  (`test_typed_features.py:283`) changes meaning for trees with string
  nodes and is rewritten to pin the new encoding; numeric-only trees encode
  byte-identically.

`isin` over strings: a chain of STR/EXACT nodes, same shape as numeric
`isin`. Fine for tens of values; see §5 for thousands.

### 3.2 Tables — move later (Stage 5)

`scan_table` (`tables/interpreter.py:72-80`) gains a `svars` span array
beside `vars_` and a per-table byte table for its string cells. `EQ` and `IN`
conditions on a string variable (`tables/schema.py:473-506`, `:528-571`,
routed today through `is_string_column` `:225-239` and the hoisted matcher
`tables/encode.py:339-358`) compare bytes: `eq_val[local, r]` and `in_vals[j]`
become pattern indices into the table's byte table, which rides in `shared`
like every other table array (`tables/encode.py:375-387`). `vars_` stays a
float64 array for the numeric variables — the "cast a code to float"
comment at `tables/encode.py:364-371` and the float64 `in_vals` convention at
`schema.py:494-500` are what go. Until Stage 5 tables keep today's matcher
path unchanged: the matcher step is a single-`str`-input packed step, which
still receives a code from `CodesPlan` (§1.6), so nothing regresses.

Why later: EXPERIMENTS §X (inline branch) measured `scan_table` at
459–464 ns/row dominated by its 150-cell scan, not by the boundary, and
tables' string literals are already data. The change is a correctness and
simplicity change, not a performance one, and it has its own test surface
(`test_tables.py` 461 lines, `test_tables_ported.py`).

### 3.3 Branch and Loop — never

Every value in a construct lives in one float64 `regs` array
(`graph/control_flow/interpreter.py:79-84`, `_engine.py:260-309`); a
`(address, length)` pair has no honest home there. `boundary_inputs`
(`_engine.py:206-240`) already rejects a `str` leaf alongside other inputs
and tells the author to hoist the comparison into a single-input step. That
rule stands; the hoisted step becomes a single-input STR step (a `bytes`
`raw1` gather yielding the span pair) instead of a code matcher, so the
construct branches on an int exactly as it does today. `walk` and the
`fn_*` wrappers keep `inline="always"` from the inline branch.

### 3.4 The STR node across the three

| | tree | table | Branch/Loop |
|---|---|---|---|
| where the bytes come from | `span` in the typed row | `svars` beside `vars_` | never inside; a hoisted single-input STR step |
| pattern storage | per-call `(pat_bytes, pat_bounds)` from `str` params | the table's own byte table in `shared` | the hoisted step's params |
| modes | exact/prefix/suffix/contains, isin as a chain | `eq`, `in` (exact) | whatever the hoisted step supports |
| Categorical input | Stage 6: `mask[code]` per node | Stage 6 | same |

---

## 4. The compiled artefact

### 4.1 What is shipped

`decider2/src/decider2/_arrow/`:
- `vendor/nanoarrow.c`, `vendor/nanoarrow/nanoarrow.h` — the 0.9.0
  amalgamation, regenerated with the release tarball's `bundle.py`
  (procedure and sha512 in `nanoarrow-accessors/vendor/VERSION`), plus
  `LICENSE.txt`/`NOTICE.txt` (Apache-2.0; NOTICE obligations carried into
  the wheel).
- `shim.c` — `import_frame`, `resolve_col`/`resolve_all`, `gather_row`,
  `get_string`, `view_*`/`release` accessors: the union of
  `nanoarrow-accessors/c/nashim.c` and
  `boundary-rework-design/c/rowshim.c`, ~150 lines, **no layout knowledge**:
  every value it hands out is a nanoarrow field or accessor result
  (`rs_resolve_col` reads `buffer_views`, `offset`, `storage_type`; the
  per-row loads are `data[offset + i]`, which is what nanoarrow's own
  `ArrowArrayViewGet*Unsafe` do). No `sm_get_string_checked`.
- `_shim.py` — the ctypes loader, `FrameView`, `RowPlan`, and the three
  `@intrinsic`s (`call_gather`, `call_get_string`, `load_u8`) lifted from
  `nashim.py`/`rowshim.py`.

### 4.2 Build backend and wheel matrix

Today `decider2/pyproject.toml` is plain `setuptools.build_meta`, pure Python,
`requires-python >= 3.10`. The shim uses no Python C API, so it is built as a
`setuptools.Extension` with `py_limited_api=True` and a one-line
`PyInit__nashim` that returns an empty module — which makes it an **abi3**
extension (`decider2/_arrow/_nashim.abi3.so`), loaded by
`ctypes.CDLL(_nashim.__file__)`. One wheel per platform, not per Python
minor (EXPERIMENTS §V's finding for a Python-ABI-free `.so`, applied). CI:
cibuildwheel over `manylinux2014_x86_64`, `manylinux2014_aarch64`,
`macosx_11_x86_64`, `macosx_11_arm64`, `win_amd64` — five wheels plus the
sdist. nanoarrow itself ships 58 PyPI wheels including MSVC and wasm32, so
the C compiles everywhere the matrix needs. Build cost is not the price
(1.6–1.9 s, 76–81 KB); the CI matrix and the "no wheel" story are.

A parallel strand (`experimentation/packaging-first-binary/`, in progress
while this was written — its `proto/pyproject.toml`, `ci/wheels.yml`,
`alt-backends/{skbuild,meson,hatch}`) prototyped exactly this shape
independently and Stage 1 should adopt its files rather than re-derive them.
Four things it found that this section would otherwise have missed:
`license-files` (PEP 639, setuptools ≥ 77) must carry nanoarrow's
`LICENSE.txt`/`NOTICE.txt` into the wheel (Apache-2.0 §4(d));
`include-package-data = false` or the C sources ride along in the wheel
(+335 KB); the header must be listed in `depends` or the sdist cannot build
(`uv build` sdist → wheel fails with "nanoarrow/nanoarrow.h: No such
file"); `NANOARROW_DLL=""` or the header sets its own symbol visibility. Its
CI matrix costs ~22 runner-minutes per run with a native arm64 runner
(no QEMU), and it proposes running it on tags and a weekly cron, not every
push. That strand also carries a `pure.py` fallback; §4.4 gives this design's
reasons for not shipping one, and Stage 1 should reconcile the two before
merging rather than ship both.

### 4.3 A platform with no wheel

`pip install` builds the sdist, which needs a C compiler. Without one the
install **fails at install time** with the standard setuptools message —
never at first request. `decider2` deploys from `docker/` into controlled
images; a missing wheel is a CI configuration fault, and the design treats it
as one. `boundary.arrow.available()` exists for diagnostics
(`explain_boundary` says "shim not loaded") but nothing in the request path
branches on it.

### 4.4 Is there a pure-numba fallback? Assessed: no, not in the package

The hand-decoded kernel (`arrow-strings-in-tree/kernel.py`) could be one: it
reads the same polars memory with pure numba, passes 38 tests, disk-caches,
and is 6–10 % *faster* per row than the nanoarrow call at a million rows
(`nanoarrow-accessors/RESULTS.md` §2.2). Against shipping it:

- It owns 42 lines of Utf8View layout (RESULTS §4) and refuses `u`/`U` by
  name; the owner's decision is that the project owns zero.
- Two decoders of one format are two equivalence surfaces, and the fallback
  one would run only where CI does not.
- The numeric side would still need the resolved addresses, which come from
  the shim — so the fallback would cover strings only, and the boundary would
  need a third path for numerics (today's numpy one).

Decision: keep it in `experimentation/` as the documented emergency option
(with an env switch for tests, `DECIDER2_ARROW_BACKEND=numba`, if a future
stage wants to exercise it), not in the wheel. The frame-tier alternative
(evaluate STR nodes in polars into bool columns and branch on `is_true`) is
the other honest fallback — it is what the docs prescribe today (doc 05
§1.5, EXPERIMENTS §O) — but it changes the compiled program per platform,
which is worse for the equivalence property than failing to install.

---

## 5. What has to stay outside the kernel

| need | verdict | why, and where it lives |
|---|---|---|
| **regex** | frame tier, as today | no engine in nopython numba (EXPERIMENTS §O); PCRE2 from C works but backtracks — `^(a+)+$` is 211 µs/row with a limit and 42 ms/row without (`SUMMARY-strings-and-types.md` §5), a DoS axis opened by a rule edit. If in-kernel regex is ever wanted the engine is Rust's linear-time crate, and that is a second language. `encode_string_match` keeps raising for `regex` with the frame-tier message. |
| **`isin` over thousands of values** | frame tier, or Categorical + mask | a chain of EXACT nodes is k byte-compares per row. Past ~50 values, declare the column Categorical and use the Stage 6 dictionary mask (O(distinct) per batch, one lookup per row), or precompute `is_in` in polars. Not built here: a hashed byte set in the kernel. Said before it is promised. |
| **Unicode case folding**, `case_sensitive=False` | frame tier (`str.to_lowercase()`), as today | ASCII folding is a few lines on the `(addr, len)` pair; full Unicode folding is a table the kernel does not have. Keep the refusal at `trees/encode.py:595-602`. |
| **`trim_whitespace`** | frame tier, as today | trivial for ASCII space, not for Unicode whitespace; keep `:603-608`. |
| **producing a string** | impossible, as today | a step cannot return `str` (EXPERIMENTS §T, verified); kernels output ints/floats/bools only, which is why equivalence never compares strings (§8). |
| **`str` inputs to hand-written steps** | dictionary codes until Stage 7 | §1.6, §7 Stage 7. |
| **List / Struct / Array** | `KernelSplitPlan`, as today | nanoarrow decodes them, nothing downstream consumes them. |
| **Decimal** | frame-tier cast to scaled int64, as today | doc 03 §1.2; the `except BaseException` discipline (`dtypes.py:379-418`) moves to the cast. |
| **a validated boundary** | no — by decision | §0.1; Appendix A for what Arrow C++ would check. |

---

## 6. Caching and the no-recompile guarantees

What the design must preserve: `assert_no_recompile` (same `Driver` object,
same `len(driver.signatures)` after a values-only retune,
`testing/recompile.py:103-149`), `assert_no_compilation_after_warmup`
(`:152-193`, counting numba's own `numba:compile` events), and the seven
on-disk cache conditions of doc 05 §4.2. Note: `decider2 build --verify` is
described in doc 05 §8 and referenced from `graph/pipeline.py:298-302`, but
`cli.py` has only `serve` (`cli.py:80-107`); the guarantee is enforced by the
`decider2.testing` functions today, and that is what the acceptance tests
below call.

How it is preserved:

1. **Pointers are arguments.** The shim's function addresses
   (`GATHER_ADDR`, `GET_STRING_ADDR`), the `RowPlan` address and the row
   buffer addresses enter the kernel as `uint64` parameters of the packed
   kernel and of `path_fn`. `nanoarrow-accessors/RESULTS.md` §7 shows a
   `cache=True` kernel taking `sm_get_string`'s address this way loads from
   disk in three processes where `libnashim.so` landed at three different
   addresses. The intrinsics are pure LLVM (`inttoptr` + `call`/`load`),
   with no dynamic global — the property EXPERIMENTS §V lost and §W
   recovered.
2. **The cached specialisation is `path_fn`** (`cache=True, inline="always"`,
   `trees/encode.py:838`); `walk_tree` has no entry of its own (EXPERIMENTS
   §X on the inline branch). The packed per-row kernel closes over the
   step's dispatcher and is per-process by design (`driver.py:484-494`,
   `compile/kernel.py:50-57`), warmed by `precompile()`. Adding two array
   arguments and three intrinsics to `path_fn` does not change that.
3. **Retune is a values change.** Thresholds ride in `(floats, ints)` tuples
   as today; patterns ride in `(uint8[:], int64[:, 2])` — the numba type is
   the same for one pattern or a hundred. Today, a tree's `params` tuple
   *length* is part of its type, so adding a literal recompiles; after Stage 2
   adding or editing a pattern does not. The negative control in
   `test_retuning_an_int_threshold_never_recompiles`
   (`test_typed_features.py:262`) gains a sibling: change a pattern's text,
   change the pattern count, assert zero compiles and the same `Driver`.
4. **The structural fingerprint** (doc 08 §4.1b) hashes rule shape,
   operators and wiring, never values; a pattern is a value.
5. **No process-specific state in anything hashed.** `_driver_key`
   (`driver.py:1113-1130`) keys on step identity, groups, owners,
   `build_dir`; the `FrameView`/`RowPlan` are per-call or per-pipeline
   objects that never reach a key.
6. **Warm-start cost** is the one regression, +0.2 s per process, already
   measured on the inline branch and accepted there.

---

## 7. The staged plan

Ordering rule: the riskiest assumption first. That is not "strings match"
(proven twice) but **"a compiled shim can be built, shipped, loaded and
called from a cached numba kernel inside the real driver, on a whole-frame
import that handles slicing, nulls and dictionaries."** Stage 1 proves it
with no consumer. Each stage is independently mergeable into
`feature/decider-v2`, keeps the full suite green, keeps
`tests/test_flagship.py` and `tests/test_spec_conformance.py` **unchanged**,
and has its own acceptance test. Worktrees are marked ∥ where they can run
concurrently; shared files are named.

| # | stage | files | single-record effect | parallel |
|---|---|---|---|---|
| 0 | merge the two branches | `trees/*`, `compile/driver.py`, `types.py`, `graph/control_flow/interpreter.py`, `docs/EXPERIMENTS.md` | none (score() unchanged; §X's +0.2 s warm-start) | serial, first |
| 1 | the shim + `FrameView`, packaging, cache proof | `_arrow/*`, `pyproject.toml`, CI | none | serial, second |
| 1b | move row-gather machinery out of `driver.py` into `compile/gather.py` (pure move) | `compile/driver.py`, `compile/gather.py` | none | serial, before 2/3/4 |
| 2 | trees' STR node; typed row = spans; patterns as params; interpreted/stepped/score spans from Python bytes | `trees/*`, `compile/gather.py` (typed half), `types.py` docstring, `testing/equivalence.py`, `runtime/invoke.py:752-763` | `score()` gets prefix/suffix/contains; no speed change | ∥ with 3, 4 |
| 3 | whole-frame boundary for `apply()`; ladder → Arrow table; nullable numerics stop copying | `boundary/*`, `compile/gather.py` (packed-input half), `runtime/invoke.py` (`apply`) | `apply(1-row)` boundary 122 → ~40 µs | ∥ with 2, 4 |
| 4 | single-record plan: hoist schema-invariant work; pooled row buffers | `graph/pipeline.py`, `runtime/invoke.py` (`score`), `compile/driver.py` (`Segment.run`, `_return_dtype`) | **409 → ≤ 60 µs** target | ∥ with 2, 3 |
| 5 | tables' string cells over spans | `tables/*` | none | after 2 |
| 6 | Categorical/Enum via the dictionary + per-batch mask; `_get_buffers` gone | `boundary/*`, `trees/encode.py`, `_arrow/shim.c` | none | after 2 and 3 |
| 7 | (optional) hand-written `str` steps get real strings from spans; `CodesPlan` deleted | `compile/kernel.py`, `runtime/invoke.py`, `boundary/*` | `score()` string inputs lose the code hack | after 2 and 3; probe first |

### Stage 0 — merge the two branches

Both branches have **zero file overlap with `feature/decider-v2`** (its 8
commits since their merge-bases are all under `experimentation/`). They
overlap with each other on `trees/interpreter.py`, `trees/encode.py`,
`compile/driver.py` (docstring) and `graph/control_flow/interpreter.py`. The
typed-features branch already carries `inline="always"` on `walk_tree` and
the cached `path_fn`; the inline branch adds Branch/Loop inlining and
EXPERIMENTS §X. Merge `bffa3fa` first (563 tests), then cherry-pick
`cdabaf1`+`b7dfc89` resolving `trees/*` in favour of the typed code and the
inline branch's doc text. Acceptance: 563 tests green;
`assert_no_compilation_after_warmup` green; `bench_typed_features.py` mixed
tree within the 115–156 ns/row band both branches report.

### Stage 1 — the shim, `FrameView`, packaging, cache proof

Build `decider2/_arrow/` (§4.1–4.2). No caller in `src/` changes.
Acceptance tests (new `tests/test_arrow_shim.py`):
- import of a frame with every row of §1.2's table; a Decimal column raises
  the frame-tier-cast error; a List column is reported, not crashed;
- nulls per kind; sliced at four offsets with nulls; a 2/2/1-chunk frame
  imports as one struct and the caller's frame is rechunked (pin the
  side-effect so a polars change is noticed); a zero-row frame; a String
  column with >1 variadic data buffer (1M rows, checked by `n_buffers`);
- Categorical/Enum expose a dictionary view of the right length and index
  type;
- a `cache=True` kernel taking `GATHER_ADDR` as an argument is a genuine
  cache hit in a fresh process (the shape of
  `test_a_typed_tree_path_fn_is_a_genuine_cache_hit_in_a_fresh_process`,
  `test_typed_features.py:409`);
- CI builds the wheel for at least `manylinux2014_x86_64` and installs it in
  a clean image without a compiler.
Single-record: none.

### Stage 1b — `compile/gather.py`

`driver.py` is 1302 lines and Stages 2, 3 and 4 all touch it. Move
`_packed_args_kind`, `_gather*`, `_fill_array`, the `_typed_*` family,
`_packed_input_arrays`, `_build_typed_kernel`, `build_packed_kernel` and
`_packed_row_args` (`driver.py:158-540`, `:808-844`) into `compile/gather.py`
unchanged; `driver.py` imports them. Acceptance: suite green, `driver.py`
diff is deletions plus imports. After this, Stage 2 owns the typed half of
`gather.py`, Stage 3 the packed-input half, Stage 4 `driver.py`'s `Segment`
classes — three worktrees, no shared hunk.

### Stage 2 — trees' STR node

§2 and §3.1. The boundary side for this stage is minimal so it does not
wait on Stage 3: a `SpanPlan` in `boundary/dtypes.py` that imports **one
String Series** through `FrameView` (the per-Series path of the prototypes,
alignment checked by `n_chunks`) for `bytes`-declared inputs; numeric
columns keep today's numpy path. `score()` marshals a string as
`value.encode("utf-8")` (0.1 µs; 0.5 µs with the address taken) kept alive
for the call, address + length into the span buffer; `record_categories` and the `registry[...] = np.array([0])` hack
at `invoke.py:752-763` are deleted for tree inputs. Interpreted and stepped
modes build spans from Python bytes too (`_typed_row_args`), so the three
rungs compare **two independent producers of the same bytes** — the C gather
against Python's encoder — which is what makes the ladder verify the
boundary (§8).
Acceptance:
- the 38 tests of `arrow-strings-in-tree/test_kernel.py` ported to run
  through `tree_module(...)` + `assert_equivalent`, plus a four-mode string
  corpus (§8.2);
- `assert_no_recompile` on: change a pattern, add a pattern, change a
  threshold;
- `test_typed_features.py` green with the rewritten string-slot test;
- `tests/test_spec_conformance.py` and `tests/test_flagship.py` unchanged
  and green (they use no trees).
Single-record: `score()` supports prefix/suffix/contains in trees; latency
unchanged (the string path adds one `encode()`).

### Stage 3 — the whole-frame boundary

§1 in full for `apply()`. `extract_frame` (`extract.py:178-229`) becomes:
route REQUIRED nulls → frame-tier casts (Decimal, declared-kind mismatches)
→ `FrameView.import` of the whole frame (`df[cols]` when much wider; never
`select`) → `resolve_all` → hand packed steps
the `RowPlan`, hand ordinary kernels numpy views over resolved addresses
(§1.6). `boundary/dtypes.py`'s five plan classes become the Arrow table plus
`KernelSplitPlan` and (until Stage 7) `CodesPlan`. `explain_boundary` keeps
its signature. `SpanPlan` from Stage 2 folds into the frame import.
Acceptance:
- `test_boundary_plans.py`, `test_boundary_dtypes.py`,
  `test_boundary_extract.py`, `test_boundary_nulls.py` rewritten to the new
  table (this is the breakage, §9);
- a nullable Int64 column at 100k rows extracts in O(columns) time and with
  zero resident-memory growth (the `verification-probes` witness, as a test);
- `assert_equivalent` over the boundary corpus (§8.2) for a numeric-only
  pipeline and a mixed one;
- protected tests unchanged.
Single-record: `apply()` on a one-row frame drops ~80 µs of boundary; `score()`
untouched.

### Stage 4 — the single-record plan

Not an Arrow change; the stage that meets the constraint. Build once per
`Pipeline` (or per `ServeHandle` generation) a `ScorePlan`: the flattened
steps/groups/owners (`pipeline.py:263`), `interface` (`:499-510` — today
re-walked with difflib per call), terminal names, each segment's
`_return_dtype` and `kernel_signature` roles (`driver.py:675-711`,
`:869-895` — today `inspect.signature(eval_str=True)` per step per call), the
`ParamSpace`s, the driver key, and pooled typed row buffers + output
buffers (EXPERIMENTS §N1: pooled output takes "kernel dispatch" from 39.9 µs
to 1.38 µs). `score()` then does: validate params (N3, ≤ 6 µs), marshal the
record into the pooled row (one Python loop over `interface.inputs`, no
`np.array([value])` per input as at `invoke.py:670-693`), call each
segment's kernel, read back terminals. `resolve_params` per call
(`invoke.py:779-782`) is reduced to the value merge.
Acceptance: a latency test on the flagship pipeline, p50 ≤ 60 µs over 2000
calls with GC on (today 409 µs; the pre-existing `test_flagship.py` score/apply
agreement unchanged); `assert_no_compilation_after_warmup` green;
`ServeHandle` generation swap still zero-compile (N4). Conflicts: touches
`driver.py`'s `CompiledSegment.run`/`PackedCompiledSegment.run` — Stage 3
touches the latter's input building; coordinate on that one method or
sequence 4 after 3.

### Stage 5 — tables (§3.2). Acceptance: `test_tables*.py` green with the
matcher steps gone from `interface.outputs`; `assert_equivalent` on a table
with string `eq`/`in` cells including nulls and >12-byte cells.

### Stage 6 — Categorical/Enum (§1.7). Adds `shim.get_dict_string`/dictionary
import, the per-batch mask built by running the STR matcher over the
dictionary view, and deletes `_extract_codes` (`dtypes.py:141-163`),
`cat.get_categories()` and `_get_buffers()` from the boundary. Acceptance:
`test_codes_plan_*` replaced by dictionary tests; a string node on a
Categorical column agrees with the same node on the same data as String, in
every mode; `assert_no_recompile` across two batches with different
dictionaries.

### Stage 7 — hand-written `str` steps (optional). Probe first: the cost of
building a numba `unicode_type` from a span inside `compile/kernel.py`'s row
body (`kernel.py:283-311`), per row per `str` input. If ≤ ~100 ns, a step
written `sector == "private"` becomes correct rather than a build error,
`startswith` works in a step, and `_check_str_inputs_are_covered_by_params`
(`invoke.py:162-196`), `_resolve_str_param_code` (`:199-244`),
`ExtractedFrame.categories` and `CodesPlan` are deleted.
`test_spec_conformance.py`'s three `sector_rate` tests still pass unchanged:
a `str` param becomes a unicode kernel argument (same numba type across
values → no recompile), and an absent literal simply never matches;
`test_a_string_input_is_never_silently_zeroed` accepts a correct answer.

---

## 8. The equivalence property

### 8.1 How `interpreted ≡ stepped ≡ fused` survives

The three rungs call the **same** `path_fn` dispatcher with the **same**
`(args, params)` shapes (`driver.py:820-830` makes that a design rule); only
who fills `args` differs. After Stage 2:

| rung | who fills the typed row | who supplies the string bytes |
|---|---|---|
| interpreted / stepped | Python, per row (`_typed_row_args`) | `str.encode("utf-8")`, a Python `bytes` kept alive |
| fused (`apply`) | the C gather over the Arrow view | polars' own Utf8View buffers via nanoarrow |
| `score()` (fourth rung, `equivalence.py:160-215`) | Python, once | `str.encode("utf-8")` |

So agreement across the ladder asserts that nanoarrow's decode of polars'
memory yields the bytes Python's encoder yields — the boundary is under test,
not assumed. Strings are never *produced* by a kernel (§5), so output
comparison (`_values_equal`, `equivalence.py:70-84`: exact, NaN==NaN) is
unchanged.

### 8.2 What `assert_equivalent` must additionally check

1. **Stop skipping the `score()` rung for `str` inputs**
   (`equivalence.py:192-194`). The skip existed because `score()` had no
   dictionary; it has bytes now. This may expose pre-existing disagreements
   in pipelines with hand-written `str` steps — those keep the skip until
   Stage 7, keyed on the step being non-packed, not on the input.
2. **A string corpus every string-bearing pipeline is driven with**, added to
   `testing/corpus.py`: null; empty string; 1, 12 and 13 bytes (the inline /
   out-of-line boundary of Utf8View, invisible to the design but not to a
   bug); multi-byte UTF-8 with a multi-byte pattern; a pattern longer than
   the value; a pattern at position 0, at the end, absent; identical
   prefixes differing only after byte 12.
3. **A frame-shape corpus**: the same rows as a fresh frame, a sliced frame
   (`df[a:b]`), a `pl.concat` of two frames, a frame with an added
   single-chunk column, and — for Stage 3 — a nullable numeric column with
   the null on row 0 and on the last row. Each must give byte-identical
   outputs to the fresh frame.
4. **Categorical ≡ String**: from Stage 6, the same values as `pl.String`,
   `pl.Categorical` and `pl.Enum` must give identical outputs.
5. **Injected drift stays localised** (doc 05 §9 item 3): a test that
   corrupts one span's length in the *interpreted* rung must fail on the
   interpreted↔stepped pair and not on stepped↔fused — proving the rungs
   are independent producers.

---

## 9. What breaks

Concrete, by stage.

- **Stage 0**: nothing user-visible. Warm-start +0.2 s.
- **Stage 1**: `pip install` of the sdist now needs a C compiler;
  `uv.lock` changes; CI gains a wheel job. A platform without a wheel and
  without a compiler cannot install decider2 (§4.3).
- **Stage 2**:
  - `types.Step.typed_args`'s documented six-tuple becomes a five-tuple;
    every helper listed in §2.1 changes; `feature_kind(bytes)` keeps its
    value but its meaning is "span", not "offsets".
  - `pipeline.interface.outputs` loses the synthetic `<tree>__m_<column>`
    matcher columns for trees. In `tests/` only
    `test_typed_features.py:185` reads one (`__m_sector` on the path step's
    inputs); it is rewritten to find the `bytes` input instead.
  - `UnsupportedInKernel` is no longer raised for `starts_with`/`ends_with`
    /`contains`. Inverted, deliberately: `test_trees.py:479` (the
    `contains` row of its parametrised refusal table; the `regex`,
    `case_sensitive` and `trim_whitespace` rows stay),
    `test_trees_ported_conditions.py:181-205` (`test_string_match_types`:
    `contains`/`starts_with`/`ends_with` become positive tests, `regex`
    stays a refusal) and `:298-321` (prefix grouping, becomes positive).
    `test_trees_ported_conditions.py:212-229` (`case_sensitive=False`,
    `trim_whitespace`) stay as they are.
  - The retune contract changes in the author's favour: adding a pattern no
    longer recompiles. The negative control "changing a field's *type*
    recompiles" (doc 05 §9 item 5) is unchanged.
  - `score()` for a tree with a string feature now `encode()`s the value;
    a non-`str` value for a `bytes` input is a `TypeError` naming the
    column, not a silent code 0.
- **Stage 3**:
  - Public names in `boundary.dtypes` — `DtypeTier`, `EntryMode`,
    `ZeroCopyPlan`, `CopyPlan`, `ScaledInt64Plan` and the `ColumnPlan` union
    — are replaced; `probe_column` (the try-the-conversion gate,
    `dtypes.py:392-418`) has nothing left to probe for fixed-width types and
    is kept only for Decimal; `explain_boundary` rows change content.
    `test_boundary_plans.py` (369 lines), `test_boundary_dtypes.py`,
    `test_boundary_extract.py`, `test_boundary_nulls.py` are rewritten.
  - `NeedsKernelSplit` survives for List/Struct; the Decimal path raises it
    from the frame-tier cast instead of from `_decimal_as_cents`.
  - The caller's frame may be **rechunked in place** by the export (probe
    (a)). Harmless — it is what `rechunk_once` already did to the copy — but
    a caller holding numpy views taken *before* `apply()` on a multi-chunk
    frame now holds views onto the old chunks. Documented; Stage 3 takes its
    own views after the export.
  - `FrameView` structs are pooled per pipeline; concurrent `apply()` from
    two threads on one `Pipeline` must get separate pools (§10 risk 2).
  - `parallel=True` groups: the packed kernel is not `prange` today and
    stays so; the row buffers are per-kernel, not per-thread.
- **Stage 4**: `Pipeline.interface` becomes cached; a pipeline mutated after
  first use (there is no public mutator, but tests build them incrementally)
  must invalidate it. `score()`'s output dict content is unchanged.
- **Stage 5**: tables' matcher columns leave `interface.outputs`;
  `shared` bundles gain byte-table fields (the bundle *type* changes once,
  at build, not per call).
- **Stage 6**: `ExtractedFrame.categories` and `TierResult.categories`
  (`extract.py:62-66,165-175`, `nulls.py:126-133`) go; `_get_buffers` leaves
  the package; a Categorical column with a per-batch dictionary pays
  O(distinct) per string node per batch.
- **Stage 7**: `_check_str_inputs_are_covered_by_params` stops rejecting a
  bare literal — a build-time error becomes a correct answer, which is a
  behaviour change authors will notice only by things working.

---

## 10. The three biggest risks

1. **The first binary.** Everything below Stage 1 assumes the shim builds
   on five platforms, loads under the abi3 pattern, and its addresses can be
   kernel arguments without touching the disk cache in the real driver
   (not the prototype's). The prototypes proved each piece separately; the
   real driver wraps `path_fn` in a per-process closure and warms it through
   `precompile()`. Stage 1's cache test and CI wheel job exist to find out
   before any consumer depends on it.
2. **Lifetime and aliasing of raw addresses.** A span is valid only while
   the `FrameView` holds its Arrow references; a `FrameView` pooled per
   pipeline is not thread-safe (serving runs concurrent `score()` — N4);
   polars rechunks a frame *during* export; numpy views must be taken after
   it. None of these is hard, all of them are the kind of bug that answers
   correctly in the test and wrongly under load. Mitigations: pool per
   thread (`threading.local`), release in `finally`, take views after
   export, and a test that runs `apply()` from 16 threads on one pipeline
   against a single-threaded oracle.
3. **Per-row cost on wide packed steps.** The C gather adds ~2–3 ns per
   column per row (17 columns: +44 ns/row over numpy `_fill`, probe V2), and
   LLVM cannot inline an opaque call. At 100 columns that is +250 ns/row on
   a walk that costs ~100. Mitigation, measured as V3 (numba loads from the
   resolved addresses, no per-row C call): 320 vs 328 ns/row — a small win
   that gives numba the loop; kept in reserve because it moves the validity
   bit read into numba (one line of Arrow knowledge). Every stage's
   acceptance includes `bench_typed_features.py` so the regression is a
   number, not a surprise.

---

## Appendix A — Does Arrow C++ validate string-view elements? Yes

pyarrow 25.0.1 was `uv pip install`ed into the project venv for this probe
only (`experimentation/boundary-rework-design/probe_pyarrow_validate.py`); it
is not and should not become a dependency. Method as in
`nanoarrow-accessors/corrupt_child.py`: a polars String column is imported
zero-copy (`pa.chunked_array(series)` through `__arrow_c_stream__`,
`string_view` type), the 16-byte view of row 1 (a long, out-of-line string)
is overwritten in polars' own buffer, then `validate()` and
`validate(full=True)` are called, each case in its own subprocess.

| corruption | `validate()` | `validate(full=True)` |
|---|---|---|
| `buffer_index` = 1000 of 1 | passes | **`ArrowIndexError: View at slot 1 references buffer 1000 but there are only 1 data buffers`** |
| `offset` = 2³¹−1 | passes | **`ArrowIndexError: View at slot 1 references range 2147483647-2147483684 of buffer 0 but that buffer is only 80 bytes long`** |
| `length` = 2³⁰ | passes | **`ArrowIndexError: … range 0-1073741824 of buffer 0 but that buffer is only 80 bytes long`** |
| `length` = −1 (0xFFFFFFFF) | passes | **`ArrowInvalid: View at slot 1 has negative size -1`** |
| clean / sliced | passes | passes |

(Reading the corrupted element afterwards segfaults or aborts pyarrow, as
expected of an unchecked accessor after validation was refused.)

So Arrow C++'s full validation checks exactly the three per-element facts
`sm_get_string_checked` had to hand-write (`nanoarrow-accessors/c/nashim.c:117-135`):
buffer index against the variadic count, offset+size against that buffer's
size, and non-negative size. nanoarrow 0.9.0's `ArrowArrayViewValidateFull`
(`vendor/nanoarrow.c:4099`) has no `STRING_VIEW`/`BINARY_VIEW` case, and
neither does `main` (`src/nanoarrow/common/array.c`, fetched 2026-09-23: it
validates offset buffers, unions, run-ends, list views and dictionary
indices only). **This is a gap in nanoarrow relative to the reference
implementation and is worth an upstream issue** ("ValidateFull does not
validate string_view/binary_view elements; Arrow C++ does" — with the table
above as the reproduction). It does not change the decision: this design
validates nothing per element by choice, and even Arrow C++ only does so
under `full=True`, which is O(rows).
