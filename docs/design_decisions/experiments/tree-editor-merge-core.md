# Writing the live-sync merge core once: a CRDT library, a JSON-patch crate or a Rust core?

Date: 2026-10-11 · Branch: `spike/merge-core` (not merged) · Prototypes: `spike/merge-core-automerge`,
`spike/merge-core-yrs`, `spike/merge-core-jsonpatch`, `spike/merge-core-rust` · Harness:
`ui/tree-editor/merge-core-bench/`

## Purpose
The live-collab spike (`tree-editor-live-collab.md`) wrote its diff/apply/merge core twice, in Python
(`live_ops.py`, 311 lines) and TypeScript (`live-ops.ts`, 284). The owner would rather have one core, ideally an
existing library, else Rust wrapped for Python and WebAssembly. Which can keep the semantics: per-field last writer
wins by arrival while connected, an edit inside something concurrently deleted or replaced dropped and its author
told, an offline batch merged only if nothing overlaps and otherwise rejected, every accepted state validated, and
the stored document still the decikit tree JSON?

## Method
- One harness first: 446 vectors generated from the pure Python core (the 89 shared cases tagged by rule, trap
  cases, and seeded edits of the 22 real trees up to 12 KB as round trip, online and offline cases), runners for
  Python and node, and one bench (generated-521 and fraud-live, 521 rules; campaign-scale, 627 nodes), each
  approach interleaved with the pure copy in one process, medians of N=30 (Automerge N=10) on a shared box.
  Format-neutral adapters (`roundtrip`, `online`, `offline`) let a CRDT be judged on the same cases as the op cores.
- Four Sonnet workers in parallel, one approach each, fixed time box: Automerge (Rust core: `@automerge/automerge`
  wasm, PyPI `automerge`), Yjs/Yrs (`yjs` and `pycrdt`), the Rust `json-patch` crate with PyO3 and wasm-bindgen, and
  a custom Rust crate (PyO3 abi3 + wasm-bindgen). An Opus review of the winner, then wiring it into the app.

## Result

**The harness found that the two existing copies already disagree** in 6 places: TypeScript has no
`rebaseOrders` (its offline merge would undo a reorder; harmless today because only the server merges), integers
past 2**53, integer-like keys listed first, key sort by UTF-16 vs code point, `isdigit` accepting non-ASCII digits
in Python only, and `touched` on malformed ops. TypeScript's `conflicts`, `staleOps` and `isRelated` (about 90
lines) are used only by tests.

| | 1 Automerge | 2 Yjs / Yrs | 3 json-patch crate | 4 custom Rust |
|---|---|---|---|---|
| vectors (446 at the time; 124 op-level n/a for CRDTs) | 317 pass, 5 fail; 16 fail with actor ids swapped | 309 (yjs) / 307 (pycrdt) pass, 12-14 fail; 31 fail with ids swapped | 446 Python, 444 wasm | 446 Python, 444 wasm |
| LWW by arrival | no: higher (counter, actor) wins; delete vs write inverted | no: higher client id wins | yes | yes |
| drop inside deleted/replaced, author told | glue, 6/6 | maps yes, lists of scalars no; telling is glue | yes | yes |
| reject on offline overlap | glue (~55+30 lines per language) | not provided; 71 lines server glue; order placement differs (5 cases) | yes | yes |
| stored doc stays decikit JSON | derived copy; the Automerge binary must be stored too | derived copy; the Y state must be stored too | yes | yes |
| logic in both Python and JS | ~100 lines each (encode, edit, keys) | ~118 lines each (mapping, edit) | 2-line index mapping | 3-line index mapping |
| Python package | `automerge` 0.2.0.dev4 pre-release, 2.4 MB wheel per CPython, aborts the process on a second transaction after an error | `pycrdt` 0.14.8, 1.1 MB wheel per CPython | 292 KB abi3 wheel | 273 KB abi3 wheel |
| browser package | 3.6 MB wasm (1.1 MB gz) | yjs 29 KB gz | 241 KB wasm (98 KB gz) | 204 KB wasm (87 KB gz) |
| one-edit round trip, generated-521, ms (Python / browser engine; pure copies 16-18) | 833 / 384 | 167 / 75 | 16 / 16 | 20 / 20 |
| offline 50 vs 50, ms (pure: 35 / 52) | 2235 / 887 | 388 / 198 | 30 / 33 | 51 / 50 |
| cold load node / Chromium, ms (pure JS: 9 / 25) | 65 / 584 | 82 / 48 | 29 / 34 | 31 / 35 |

- **CRDTs** merge everything and order concurrent writes by actor or client id, not by arrival; rejecting an
  overlapping offline batch, telling an author an edit was dropped, the rule-order rule and keyed lists all become
  glue, and much of that glue (the JSON-to-CRDT mapping and the edit) must exist in both Python and JS. Both need
  their own state stored beside `project_data`: two replicas rebuilt from the same stored JSON are unrelated
  histories, and merging them silently lost one side's edit in both prototypes. Both are 4 to 64 times slower on
  521 rules because the editors produce whole documents that must be diffed into CRDT edits.
- **The json-patch crate** does the final write of an op already located, about 10 of 439 lines; diff (it matches 11
  of 31 diff vectors), keyed lists, lenient apply, order and the stale and conflict rules are custom. Its object
  remove reorders keys under `preserve_order`. It is approach 4 with a dependency that adds nothing.
- **The custom Rust core** ports all eight functions in 492 lines (core 439 + JSON entry point 53), plus 36 lines of
  bindings and 93 of glue (Python 58, TypeScript 35; a 3-line index mapping is the only logic in both): 621 against
  595 today, so it saves no lines, but the logic exists once. JSON text crosses
  both boundaries: pythonize is 28% faster but loses integers past 2**64; serde-wasm-bindgen is slower and turns a
  `__proto__` key into the object's prototype. It is not faster: copying the document across the boundary and
  scanning keyed lists cost what the interpreters spent (native: parse 2.4, clone 1.5, apply of 50 ops 5.3 ms).
  Clean build: wheel 22 s, wasm 27 s + wasm-opt 15 s; wasm-pack's binaryen needs `--enable-bulk-memory` and friends
  for rustc 1.95 output.

**Wired into the app** (`spike/merge-core`): `live_ops.py` is a 12-line re-export of the extension and
`live-ops.ts` a 35-line wrapper; both copies are gone (frozen references stay in the harness). One fixture: pytest,
vitest and `cargo test` all run `merge-core-bench/vectors/op_level.json`. The UI initialises the wasm before the
first render (`init` with vite's `?url`) and vitest with `initSync` in a setup file. One deliberate change: the
conflict rule's index guard takes ASCII digits only, as apply's index rule does (Python's `isdigit` also took `١`).
The Opus review read the core against the Python
reference and ran a differential fuzz (21,700 cases over the real trees incl. the 521-rule sets, 17 calls each,
compared as exact JSON; 12,000 replayed in wasm). It found one silent edit loss, fixed and pinned by two vectors: an
integer was compared with a float after rounding to f64, so past 2**53 `diff` missed an int-to-float change and the
offline same-value rule merged a real conflict. After the fix: 0 mismatches. On the merged tip: Python 449/449,
wasm 445 + 4 xfail (JavaScript cannot hold those numbers), backend 237 passed, vitest 366 passed, the two-node
live e2e (`test_live_collab.py`, `test_live_offline.py`) 18 passed (140 s, once) and the smoke set with live off 21 passed. Open: if the wasm fails to load, the
app stays blank, and the flat-rule editor needs it even with live editing off (it diffs per rule on load: 14.5 ms in
wasm vs 11.0 ms in TypeScript on 521 rules); lone surrogates and nesting past 128 levels raise (Postgres rejects the
first, real trees nest 9 deep).
- Browser: +87.6 KB gzip of wasm, JS unchanged; the real app's first render 2.5 ms later (66.8 to 69.3 ms, N=10).
- Wheel: `maturin build` here gives `cp310-abi3-manylinux_2_34` (273 KB), which pip refuses on glibc 2.31;
  `maturin build --zig --compatibility manylinux_2_28` (ziglang from PyPI, no sudo) gives a 282 KB wheel that
  imports on both. abi3: one wheel per platform for every Python from 3.10.
- Image: a `rust:1.95-bookworm` stage installs maturin and wasm-pack and builds the wheel and `pkg/`: image +1.2 MB,
  cold build 14.5 to 23 min (the Rust stage about 8 min; most of the rest is pip), warm rebuild after a Rust change
  16 s.
- Contributors: `pkg/` is built by `pnpm build:wasm`, not committed. Without wasm-pack the UI build, tsc and vitest
  fail (`Could not resolve '../../../merge-core/pkg/merge_core.js'`); without a C compiler the backend's
  `--with ../merge-core` fails (maturin even fetches Rust itself, then `linker cc not found`). So today every
  contributor needs rustup, the wasm32 target, wasm-pack and maturin. The cheap fix is a CI job that builds the
  wheel and `pkg/` and publishes them (a find-links directory or an index, and an npm artefact or a committed
  `pkg/` once wasm-opt output is shown reproducible); then only someone changing the core needs Rust.
- CI: `pr.yml` runs only the root pytest (the tree editor is not tested in CI at all); `release.yml` builds the
  image with a layer cache. The core adds `cargo test`, `maturin build` and `wasm-pack build` (22 s, 27 s + 15 s
  clean here); not timed in a clean CI container.

## Recommendation

| option | logic written | vectors (449; Python / wasm) | speed vs today | ships | toolchain and CI | verdict |
|---|---|---|---|---|---|---|
| Automerge | ~100 lines glue per language + ~90 server | 317 (5 to 16 fail: LWW by actor, delete vs write) | 17 to 64x slower | 2.4 MB wheel per CPython (pre-release); 1.1 MB gz wasm; binary stored beside the JSON | none | **no**: wrong LWW, new storage and wire format |
| Yjs / Yrs | ~118 lines glue per language + ~90 server | 307 to 309 (12 to 31 fail: LWW by client id, order placement, scalar lists) | 4 to 11x slower | 1.1 MB wheel per CPython; 29 KB gz yjs; Y state stored beside the JSON | none | **no**: no reject, wrong LWW, two implementations of one protocol |
| json-patch crate | 439 Rust, crate used in ~10 | 446 / 444 (of 446) | equal | 292 KB wheel, 98 KB gz wasm | Rust | **no**: the custom core plus a dependency that adds nothing |
| custom Rust core | 492 Rust + 129 bindings and glue, once | 449 / 445 (4 = JS number limits) | equal (most of a call is copying the document across the boundary) | 282 KB abi3 wheel, 88 KB gz wasm, +2.5 ms first render | Rust for every contributor until CI publishes wheel and `pkg/`; image cold build +8 min | **yes, if live sync ships** |
| keep two copies | 311 Python + 284 TS (TS can drop ~90 test-only lines) | 448 / 436 (6 known disagreements) | today | nothing | none | fallback: run the shared harness in CI for both |

Take the Rust core when step 1 (live sync) of `tree-editor-live-collab.md` merges, together with a CI job that
builds and publishes the wheel (`--zig`, manylinux_2_28) and `pkg/`, and a catch around the wasm load; until
then nobody pays for it. It buys one definition of the semantics (the harness found six places where the two copies had already drifted), not speed or size. If adding a Rust
toolchain to the tree editor is not wanted, keep the two copies but run the harness vectors for both in CI.
