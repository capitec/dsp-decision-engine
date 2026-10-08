# Serving: the request path, transports and the Arrow boundary

Read this before changing how a request is parsed, scored or transported, or how frames reach kernels; the walkthrough is `docs/architecture/serving.md` and `docs/architecture/engine/boundary.md`.

## One record goes through `score(dict)`, never a frame

- `process_fn` branches on the parsed body: a JSON object calls `Executable.score`, anything else (array, JSONL, CSV, Parquet, Arrow IPC) calls `run` on a frame.
- Why: `run(one-row frame)` took 150 us against `score()`'s 30.4 us (flagship).
- See: [serving latency budget](../experiments/serving-latency-budget.md), `decider/serving/_handler.py`

## The handler hooks are synchronous, and `score` is the in-process entry

- `init_fn`, `input_fn`, `process_fn` and `shutdown_fn` are plain methods; a subclass that defines one with `async def` gets a `TypeError` when the class is created. `score(record_or_frame)` is `process_fn` without the bytes. `score` is not a hook: `process_fn` calls `RequestHandler.score` itself, so a subclass's own `score` (a replay service's, say) never changes what a request does.
- Why: nothing in the default path awaits, a coroutine cost a frame and an event loop per in-process call, and in-process code should not go through the HTTP layer's bytes machinery.
- The Starlette and Sanic endpoints stay `async def` and call the hooks directly, with no thread hop: single-record latency comes first.
- See: `decider/serving/_handler.py`

## JSON is decoded with `pydantic_core.from_json`

- It answers the same values as `json.loads` about 4x faster (0.90 us against 4.08 us on flagship; `process_fn` 49.9 -> 39.4 us).
- One visible change: a lone surrogate (`"\ud800"`) is rejected with a 400.
- See: `decider/serving/_parse.py`

## Arrow IPC is accepted for batches, not recommended for one record

- `application/vnd.apache.arrow.stream` / `.file` are input handlers and the stream is an output formatter, through polars with no pyarrow dependency.
- Why: for one record `read_ipc_stream` costs 17 us and `write_ipc_stream` 40 us against 0.90 us and 1.16 us for JSON; Arrow pays off from about 4096 rows.
- See: [Arrow codec and transports](../experiments/arrow-codec-and-transports.md), `decider/serving/_format.py`

## No Arrow-native, Flight or socket server, and no pyarrow dependency

- Not: a specialised server. The codec is 2-6% of a request, so a free codec saves about 2 us; the long-lived pyarrow record-batch stream measured slowest of every transport except HTTP.
- Not: pyarrow (146.7 MiB installed against polars' 7.9, a second Arrow runtime beside the shim). A co-located caller should `import decider` and call `score()`; a remote caller's network dwarfs the rest.
- See: [Arrow codec and transports](../experiments/arrow-codec-and-transports.md)

## HTTP framing is the largest transport term, and the one lever is deployment

- A no-op route costs 207 us of HTTP + ASGI framing against 18 us over a unix socket and 3.3 us over shared memory; httptools + uvloop did not reliably beat h11 + asyncio on a loaded box.
- Any faster transport keeps HTTP and the `RequestHandler` hooks; nothing in the CLI serves a unix socket yet (untested idea: `uvicorn --uds`).
- See: [Arrow codec and transports](../experiments/arrow-codec-and-transports.md), `docs/architecture/serving.md`

## Not landed: the in-process engine floor for `score()`

- `score()` still builds a `State`, per-dtype arrays and output buffers each call: 39.4 us for a whole flagship request against about 7 us with everything pre-bound (tree: 77.2 against about 18).
- A fix must key a per-(plan, thread) cache on plan identity, cache only `n == 1`, follow the per-thread `FrameView` pool, and never hand a cached `State` to a `Session`.
- See: [serving latency budget](../experiments/serving-latency-budget.md), `docs/architecture/engine/concurrency.md`

## Strings for one record are described from their own bytes

- Up to 32 values (`score()`), `spans()` points at each value's own `bytes` object; above that it exports the frame's column.
- Why: building spans through a frame cost 71.7 us for one 3-byte string (gated tree `score()` 136 us, now 82 us, against 63 us ungated).
- See: [string representations](../experiments/string-representations.md), `decider/_engine/run/representations.py`, `docs/architecture/engine/boundary.md`

## A frame is read through one `__arrow_c_stream__` export and one C call

- `sm_columns` reads every declared column of the batch into one buffer via a small C shim over vendored nanoarrow 0.9.0; polars rechunks on export, so addresses taken before binding are stale.
- Why: the per-row gather it replaced was O(rows x columns): fused `run()` on flagship took 97.6 ms at 1M rows against 3.1 ms from `to_numpy()` columns.
- See: `decider/_engine/boundary/_arrow/c/shim.c`, `docs/architecture/engine/boundary.md`

## Columns of 4096 rows or more with no nulls are borrowed, not copied

- A float64 / int64 / int32 column the kernel reads as it is becomes a read-only view of the Arrow buffer, owned by a moved `ArrowArray` (`_Export`); below `_BORROW_ROWS = 4096` a copy is cheaper.
- Lifetime rule: anything that hands a raw address to a kernel must stay reachable from a live Python object for the whole call, and nothing may rechunk or mutate the owner meanwhile.
- A socket read buffer reused across requests with `nogil=True` kernels in a thread pool is a use-after-free (a segfault, not a wrong answer): one buffer per in-flight request.
- See: `decider/_engine/boundary/_arrow/view.py`, `docs/architecture/engine/boundary.md`

## Nullable columns are still copied; zero-copy validity is not adopted

- Validity becomes a byte mask per row. Reading `(bitmap, offset)` in the kernel made null-heavy wide frames 6-7x faster at 1M rows but touched about 17 boundary tests and the kernel ABI, and was left behind a flag that is no longer in the code.
- `score(dict)` never goes through polars/Arrow (126 us against 30 us on flagship); it builds the same `(values, validity)` arrays from the dict.
- Cheap batch wins left: `Enum.gather` for `Literal` outputs (about -40 ms per 1M tree rows) and `Series.set(mask, None)` for nullable outputs.
- See: [zero-copy Arrow gather spike](../experiments/zero-copy-arrow-gather-spike.md)

## The shim is built on first import with `$CC`; there is no pure-Python path

- A plain shared library loaded with `ctypes`, compiled into `$XDG_CACHE_HOME/decider/arrow-shim/shim-<hash>.so`, where the hash covers sources, flags, compiler and machine; the write is atomic, and nanoarrow is namespaced (`DeciderArrow`) so pyarrow or adbc cannot interpose.
- Why: hatchling has no C-extension support; building on import needs only a compiler, not CPython headers.
- A machine with no compiler gets one `ImportError` naming platform and cause (`_arrow.diagnose()` reports it without raising). Open: a read-only home or no compiler needs the `.so` built at image build time or a prebuilt wheel; Windows is unsupported.
- See: `decider/_engine/boundary/_arrow/_build.py`, `docs/architecture/engine/boundary.md`

## Kernels take shim addresses as arguments, never captured constants

- The gather, string and `load_*` functions are called through `uint64` address arguments.
- Why: a captured symbol address makes a `cache=True` kernel uncacheable; `tests/boundary/test_shim.py` checks the disk cache with a second process.
- See: `decider/_engine/boundary/_arrow/intrinsics.py`
