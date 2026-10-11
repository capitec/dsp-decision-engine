# merge-core-bench

One vector suite and one set of measurement scripts that every merge-core approach (a CRDT library, a JSON-patch
crate, a custom Rust core) is judged with. The expected results come from the pure Python live ops as they were
before any approach replaced them; `reference/` holds frozen copies of that file and of its TypeScript twin, so the
baseline stays put when an approach rewrites the product code.

## Files

| file | what |
|---|---|
| `live_ops_cases.json` | the 89 hand-written op-level cases (they lived in `backend/tests/data/`); `make_vectors.py` tags them |
| `vectors/op_level.json` | those 89 cases, tagged, plus op-level trap cases: the one fixture the product test suites run (`backend/tests/test_live_ops.py`, `ui/src/lib/__tests__/live-ops.test.ts`) |
| `vectors/neutral.json` | format-neutral `roundtrip` / `online` / `offline` / `stored` cases derived from those, plus traps |
| `vectors/real_trees.json` | seeded random edits of the small real trees; documents stored once in `docs`, cases point at them with `{"$doc": id}` |
| `make_vectors.py`, `traps.py` | write the three vector files and `out/bench_inputs.json` (git-ignored) |
| `run_vectors.py`, `run_vectors.mjs` | run every vector against an adapter |
| `adapters/pure_py.py`, `adapters/pure_ts.mjs` | the reference adapters (`reference/` copies; node runs the `.ts` with its type stripping) |
| `bench.py`, `bench.mjs` | time an adapter against the reference, interleaved in one process |
| `fuzz.py`, `fuzz.mjs` | differential fuzz of the Rust core against the reference, exact JSON text; the wasm replays the Python cases |
| `sizes.sh`, `coldload.mjs`, `coldload_browser.py`, `coldload_app.py` | artefact sizes, cold load in node, cold load of a test page in headless Chromium, cold load of the built app |

## Adapters

Python: a module with `diff apply stale_ops conflicts rebase_orders is_related touched reconcile` (op-level, only
for approaches that keep the op wire format) and `roundtrip online offline load export` (every approach). JS: an ES
module with the same names in camelCase; `export` is a reserved word, so define it as
`const exportDoc = ...; export { exportDoc as export }`. A JS adapter that needs async set-up (wasm) exports a
`ready` promise; the runners, `bench.mjs` and `coldload.mjs` await it.

- `roundtrip(old, new) -> doc`: encode old -> new your way, apply it to a fresh replica of `old`; expected `new`.
- `online(base, first, second) -> (doc, dropped)` (JS: `[doc, dropped]`): the server applies `first`'s edit, then
  `second`'s (made against `base`). `dropped` lists the paths of `second`'s changes that were dropped, in the
  notation of `touched`: the change's JSON-pointer path with rules, nodes, edges and subtrees addressed by key, an
  order change as `<list>/#order`. Compared as a set.
- `offline(base, mine, theirs) -> {"merged": doc} | {"conflicts": [paths]}`: `theirs` is on the server; `mine`
  arrives as one batch. Conflict paths use the same notation and are compared as a set.
- `load(doc) -> state`, `export(state) -> doc`: the stored JSON in and out.

Something the approach cannot do raises `NotImplementedError` (JS: throws `Error('not implemented: <why>')`), or
is left undefined; it counts as `not-implemented`. An adapter may define `XFAIL = {case name: reason}`: a failure of
a listed case counts as `xfail`, a pass as `xpass`. Only the reference adapters should need it.

## How cases are compared

Values compare as JSON: `1` equals `1.0`, `true` is not `1`, object key order is ignored (JSONB does not keep it)
except in cases with a `keys` list, which the result's top-level keys must match in order. Every case also checks
that no input was changed, and the runner then mutates the returned document and checks the inputs again (a result
that shares a value with an input fails). `apply` and `stale` cases check that the dropped ops are the caller's op
objects (`live_sync.py` filters them by identity). `raises` cases check the Python exception class; in JS any
throw passes.

## Commands

Everything runs through `REPO/.scratch/heavy.sh`. From `ui/tree-editor/backend` (decikit is needed for the real
trees), regenerate the vectors and the bench inputs:

    uv run --no-project --with-requirements requirements.txt --with-editable ../../.. --with ../merge-core --with pytest \
        python ../merge-core-bench/make_vectors.py

From `ui/tree-editor/merge-core-bench` (the runners and benches need only the standard library and node 22.18+):

    python3 run_vectors.py adapters/pure_py.py [--json out/py.json]
    node run_vectors.mjs adapters/pure_ts.mjs [--json out/ts.json]
    python3 bench.py adapters/<approach>.py [-n 30] [--json out/bench-py.json]    # reference: adapters/pure_py.py
    node bench.mjs adapters/<approach>.mjs [-n 30] [--json out/bench-js.json]     # reference: adapters/pure_ts.mjs
    ./sizes.sh <wheel> <.wasm> <glue.js> <tarball>
    node coldload.mjs adapters/<approach>.mjs [-n 20]
    uv run --no-project --with playwright python coldload_browser.py <dir with index.html> <port> [-n 20]

The reference browser page for `coldload_browser.py` (and a plain-JS reference for `coldload.mjs`, since importing
the `.ts` adds node's type stripper to the timing): strip the TypeScript into `out/browser-ref/live-ops.js` with
node's `module.stripTypeScriptTypes` and add an `index.html` holding
`<script type="module">import "./live-ops.js"; window.__ready = performance.now()</script>`.

## Vectors

449 cases, 735 KB in all: `op_level.json` 127 cases (40 KB), `neutral.json` 82 (37 KB), `real_trees.json` 240 cases
over 270 shared documents (658 KB). Real trees get seeded edits when their compact JSON is at most 12 KB (22 of the
29 trees; the generated sets above 32 rules, `campaign-scale`, `fraud-live` and `fraud-shadow` are skipped): per tree 3 chained
roundtrip steps and 2 pairs of concurrent edits (each as `online` and `offline`), flat trees both as stored and with
rule ids stamped. Of the 60 real-tree offline cases 41 merge and 19 conflict; 3 of the 60 online cases drop edits.
Regenerating gives byte-identical files.

Bench inputs (`out/bench_inputs.json`, 3.1 MB, not committed): `generated-521` (the 521-rule split set),
`fraud-live` (521 rules, the largest real tree) and `campaign-scale` (627 nodes, the largest node tree). The 50
edits are rule renames (node moves); `mine50` and `theirs50` edit opposite halves of the list, so the offline task
always merges. "Apply of a 50-edit batch" is `roundtrip(base, doc with 50 edits)`: format-neutral adapters have no
separate apply.

## Twin disagreements (the reference TypeScript copy against the Python one)

Recorded as `XFAIL` in `adapters/pure_ts.mjs`:

- No `rebaseOrders` in TypeScript (only the server rebases), so its offline merge undoes a reorder they made
  since the base when I inserted (3 offline cases, one from a real tree; 3 `rebase` cases not implemented).
- Integers past 2**53 lose precision (JavaScript numbers), so `diff` sees no change between 2**53 and 2**53+1.
- Integer-like object keys are listed first in JavaScript, whatever the insertion order (`apply` of `/1` onto
  `{"b": 1}` gives keys `1, b`; Python `b, 1`).
- `diff` sorts keys by UTF-16 code unit, Python by code point: an astral character (an emoji) sorts before
  U+E000..U+FFFF in TypeScript, after in Python, so the op order differs.
- `conflicts` takes only ASCII digits as possible index segments, as `apply` does. Python's `str.isdigit` also takes
  other scripts' digits (`١`, `１`), so for such a key under something they replaced the Python copy reports a
  conflict; the vectors now pin the ASCII rule (the Python reference is `XFAIL` on the `١` case; `½` is not a digit
  to Python either).
- `touched` does not check its ops: an op without a path or that is not an object gives `undefined`; Python raises
  `KeyError` / `TypeError`.

Seen but not a vector (value comparison cannot tell, or no caller depends on it): both copies' `diff` returns ops
whose values are the new document's own objects (aliased); TypeScript's JSON clone turns `-0` into `0`; TypeScript
`apply` throws a `SyntaxError` for an op whose `value` is `undefined` (Python cannot express it).

## Baselines

Pure Python (`bench.py adapters/pure_py.py -n 30`) and pure TypeScript in node 22.22 (`bench.mjs
adapters/pure_ts.mjs -n 30`), one heavy.sh slot on the shared box, 2026-10-11; median (p95) in ms. `load` and
`export` are identity (under 0.01 ms).

| document | task | Python | TypeScript |
|---|---|---|---|
| generated-521 | roundtrip one edit | 16.1 (16.3) | 17.1 (18.1) |
| | roundtrip 50 edits | 23.3 (24.6) | 21.0 (22.4) |
| | offline 50 vs 50 | 34.4 (35.0) | 50.2 (51.7) |
| | online 1 vs 1 | 31.7 (33.7) | 34.1 (36.1) |
| fraud-live | roundtrip one edit | 20.4 (20.8) | 22.0 (23.1) |
| | roundtrip 50 edits | 27.2 (28.2) | 25.6 (27.3) |
| | offline 50 vs 50 | 41.4 (41.7) | 63.1 (65.3) |
| | online 1 vs 1 | 40.8 (59.1) | 43.9 (45.9) |
| campaign-scale | roundtrip one edit | 14.4 (14.7) | 13.5 (13.8) |
| | roundtrip 50 edits | 20.1 (20.9) | 16.9 (17.4) |
| | offline 50 vs 50 | 30.7 (31.2) | 39.7 (40.8) |
| | online 1 vs 1 | 28.8 (29.5) | 27.0 (27.8) |

Cold load, N=20, median: node, importing the `.ts` through type stripping 85.6 ms, the stripped `.js` 9.0 ms (two
interleaved rounds agreed within 0.3 ms); headless Chromium, the stripped `.js` page ready 24.7 ms after navigation
start (min 22.7, max 34.2).
