# merge-core

The live-sync ops of `backend/dsp_decikit/services/live_ops.py` (diff, apply, stale_ops, conflicts, rebase_orders,
is_related, touched, reconcile) written once in Rust on `serde_json::Value` (`preserve_order`), wrapped for Python
(PyO3, abi3-py310, maturin) and for the browser (wasm-bindgen, `wasm-pack --target web`). It is the app's only implementation: the backend's `services/live_ops.py` re-exports it and the UI's `src/lib/live-ops.ts` wraps the wasm.

| path | what |
|---|---|
| `src/lib.rs` | the core (429 lines) |
| `src/wire.rs` | one entry point, `call(name, args_json) -> result_json`, shared by both bindings (53) |
| `src/python.rs`, `src/wasm.rs` | the bindings |
| `python/merge_core/__init__.py`, `js/merge-core.mjs` | glue: one wrapper per function, JSON in and out (46, 23) |
| `tests/vectors.rs` | `cargo test` runs the 124 op-level vectors of `../merge-core-bench` |
| `examples/prof.rs` | native timings |

Build: `maturin build --release` (wheel), `wasm-pack build --target web --release --no-pack -- --features wasm` (`pkg/`, not committed; `pnpm build:wasm` in `ui/tree-editor/ui` runs it).
`apply` and `stale_ops` return indices into the ops passed, the glue maps them back to the caller's own op objects.

## Boundary: JSON text for both

The direct variants below were measured with `bench/boundary.*` and `direct-python`/`direct-wasm` cargo features, since removed (they cost 3 dependencies and about 50 lines for a rejected route; the numbers stay here). Measured on the 521-rule document (`bench/boundary.*`, N=30, medians, ms, one `apply` of 50 ops, shared box):

| | Python | node |
|---|---|---|
| encode args (`json.dumps` / `JSON.stringify`) | 2.1 | 0.9 |
| native parse + clone + serialize, no ops | 3.3 | 5.3 (wasm) |
| native apply of 50 ops on text | 7.3 | 10.2 |
| decode result | 2.4 | 1.7 |
| text route total | 12.2 | 13.0 |
| direct route total (pythonize / serde-wasm-bindgen) | 8.8 | 22.0 |

- pythonize is 28% faster, but needs a build without serde_json `arbitrary_precision` (which the direct
  conversions cannot serialize) and then fails the 3 integer-past-2**64 vectors ("JSON number out of range").
- serde-wasm-bindgen is slower and unsafe: a `__proto__` key sets the object's prototype instead of being data
  (4 `proto-key` vectors fail; `Object.keys(doc)` is empty).
- Decision: JSON text for both. One code path, big integers kept, `__proto__` stays data (JSON.parse defines it).

## Vectors

`run_vectors.py adapters/rust_py.py` (wheel): 446 passed. `run_vectors.mjs adapters/rust_wasm.mjs` (wasm): 444 passed,
2 xfail that JavaScript itself cannot hold (integers past 2**53 that differ by one: JSON.parse rounds them; an
integer-like key added last is listed first by the engine), as in the TypeScript reference. `cargo test`: 124 op-level
cases passed. Direct variants: Python 443 passed, 3 large-int failed; JS 440 passed, 4 proto-key failed, 2 xfail.

Deviations from the Python twin (none pinned by a vector):
- `conflicts` takes a key segment as an index only when it is ASCII digits, as `apply` does (the TypeScript copy
  already did); Python's `str.isdigit` also took other scripts' digits (`١`). The vectors pin the ASCII rule and the
  frozen Python reference is `XFAIL` on the `١` case.
- `touched`/`stale_ops` raise ValueError for a non-string path (Python returned it as is, then failed in `split`);
  `rebase_orders` raises TypeError for a non-object op (Python raised AttributeError).
- NaN/Infinity (Python's `json.dumps` writes them) are a parse error; JSONB cannot store them either.

## Measurements

Interleaved with the reference adapters in one `heavy.sh` run (`bench.py`/`bench.mjs`, N=30, median ms, shared box;
rust_py vs pure_py, rust_wasm vs pure_ts):

| document | task | Rust (wheel) | Python | Rust (wasm) | TypeScript |
|---|---|---|---|---|---|
| generated-521 | roundtrip one edit | 20.4 | 17.3 | 19.8 | 18.2 |
| | roundtrip 50 edits | 24.6 | 24.5 | 24.6 | 22.5 |
| | offline 50 vs 50 | 51.2 | 36.8 | 50.2 | 52.6 |
| | online 1 vs 1 | 40.6 | 34.6 | 39.1 | 36.7 |
| fraud-live | roundtrip one edit | 24.2 | 21.8 | 23.9 | 23.3 |
| | roundtrip 50 edits | 28.6 | 29.1 | 28.8 | 27.2 |
| | offline 50 vs 50 | 59.8 | 44.1 | 59.2 | 66.8 |
| | online 1 vs 1 | 48.5 | 43.7 | 48.0 | 46.6 |
| campaign-scale | roundtrip one edit | 17.6 | 15.5 | 16.5 | 14.1 |
| | roundtrip 50 edits | 20.6 | 21.8 | 20.2 | 17.8 |
| | offline 50 vs 50 | 44.1 | 32.8 | 41.6 | 41.8 |
| | online 1 vs 1 | 35.2 | 31.1 | 33.1 | 28.8 |

No speed-up: the cost is the key scan (every op re-reads the keys of a 521-item list; the Python copy does the same)
and the copy of the document across the boundary, not interpreter overhead. Native (`examples/prof.rs`, release,
134 KB document): parse 2.4, clone 1.5, serialize 1.0, diff of 50 edits 2.3, apply of 50 ops 5.3, conflicts 18.5 ms.

Cold load, N=20, median: node 31.4 ms (reference plain `.js`: 7.7); headless Chromium page ready 35.1 / 35.6 ms in
two rounds (reference 26.3 / 28.1).

## Sizes (raw / gzip -9, bytes)

| artefact | raw | gzip |
|---|---|---|
| wheel (`cp310-abi3-manylinux_2_34_x86_64`) | 272,721 | 271,688 |
| `.wasm` before wasm-opt | 216,168 | 87,760 |
| `.wasm` after wasm-opt -O3 | 203,751 | 87,269 |
| `.wasm` at opt-level z (+ wasm-opt) | 167,310 | 78,490 |
| JS glue (`merge_core.js`) | 8,136 | 2,317 |
| npm tarball of `pkg/` | 90,837 | 90,312 |
| the TypeScript copy as plain JS, for scale | 11,609 | 4,005 |

Dependencies: serde_json 1.0.151 (indexmap 2.14.2, hashbrown, memchr, itoa, zmij), pyo3 0.25.1 (python), wasm-bindgen
0.2.128 (wasm), optional pythonize 0.25 and serde-wasm-bindgen 0.6.

## Build times (this box, queued behind other work; cargo's own "Finished" time)

Clean release wheel 22 s; clean wasm build 27 s (incl. installing wasm-bindgen) + wasm-opt about 15 s; incremental
(touch lib.rs) wheel 10.7 s, wasm 15 s (wasm-opt dominates).

## In the app

- Bundle (`vite build`, 50 JS chunks, before = `spike/merge-core-rust`): JS 1,343,497 B before, 1,343,704 B after
  (+207 B: the copy of the ops, 11.6 KB, is replaced by the glue and wrapper; 394,253 vs 394,217 B gzipped as one
  stream), plus the wasm asset `merge_core_bg-*.wasm`, 202,800 B (87.58 kB gzip as vite reports it), fetched
  alongside the entry chunk before the first render.
- Cold load of the real app, headless Chromium, fresh context per run, N=10, medians in ms from navigation start,
  both builds served by a backend (`merge-core-bench/coldload_app.py`, builds interleaved, two rounds): #root has
  content at 66.8 / 68.2 before, 69.3 / 70.2 after; the load event 45.0 / 46.5 before, 47.0 / 47.0 after; the wasm is
  instantiated at 58.2 / 58.2. About +2.5 ms on a cold local load.
- Two-node live e2e on this branch (`E2E_NODES=2 DSP_LIVE_COLLAB=true`, `test_live_collab.py test_live_offline.py`):
  18 passed in 125 s.

## Packaging

Measured on the shared dev box (Fedora, glibc 2.43, rustc 1.95, maturin 1.15.0); times include queueing and a slow network.

- `maturin build --release` here gives `merge_core-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl` (272,060 B; newest
  symbol `GLIBC_2.34`). It imports in `python:3.13-slim` (glibc 2.41) and pip refuses it on `python:3.11-slim-bullseye`
  (glibc 2.31).
- Portable wheel without sudo: `uv run --with ziglang --with maturin maturin build --release --zig --target
  x86_64-unknown-linux-gnu --compatibility manylinux_2_28` gives `...-manylinux_2_28_x86_64.whl` (282,464 B, newest
  symbol `GLIBC_2.28`), 41 s of CPU from a warm cache. It imports in both containers above. Not tried on a real 2.28 host.
- The image (`ui/tree-editor/Dockerfile`) has a `rust:1.95-bookworm` stage that installs maturin and wasm-pack with
  `cargo install --locked`, builds the wheel and `pkg/`, and hands the wheel to the runtime stage and `pkg/` to the
  node stage; nothing prebuilt is needed. Image size 774,021,459 B before, 775,254,912 B after (+1.2 MB). A cold
  build (`--no-cache`) took 14 min 30 s before and 23 min 9 s after; most of both is the network-bound `pip install`
  of the backend requirements (about 10 min here), so the Rust stage adds roughly 8 min cold, and it is a cached
  layer afterwards. The image starts against Postgres, `/api/health` answers, `merge_core` imports and the wasm is in
  `build/assets/` served as `application/wasm`.
- A contributor without Rust: `pnpm build:skip-check`, vitest and `tsc` fail at once with `Could not resolve
  '../../../merge-core/pkg/merge_core.js'` (and `TS2307` from tsc), `pnpm build:wasm` with `wasm-pack: command not
  found`. The backend extension is built by uv through maturin, which installs Rust itself when it finds none (a
  rustup download into `~/.cache/puccinialin`), then fails on a machine without a C compiler (`error: linker \`cc\` not
  found`, checked in `ghcr.io/astral-sh/uv:python3.13-bookworm-slim`); with a compiler it works, slowly the first time.
  The cheapest fixes, in order: publish the wheel (a `--find-links` directory or an index) and keep building `pkg/` in
  CI; or commit `pkg/` (200 KB binary per change, and wasm-opt output must be shown to be reproducible first).
- wasm-pack's bundled binaryen is older than what rustc 1.95 emits (bulk memory, nontrapping float-to-int...), so
  `Cargo.toml` passes `--enable-*` flags to wasm-opt; without them `wasm-pack build` fails.
- Not verified: the dev compose file (`docker-compose.yml`, a new `merge-core` one-shot service builds the wheel and
  `pkg/` into the checkout) was written but not run; CI is untouched (`pr.yml` runs only the root `uv run pytest`,
  `release.yml` builds the tree-editor image with this Dockerfile and a GitHub Actions layer cache) and no cold/warm
  timings of cargo test, maturin and wasm-pack were taken in a clean container.
