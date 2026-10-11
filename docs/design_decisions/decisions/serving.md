# Serving: the request path, transports and the Arrow boundary

Read this before changing how a request is parsed, scored or transported, or how frames reach kernels; the walkthrough is `docs/architecture/serving.md` and `docs/architecture/engine/boundary.md`.

## One record goes through `run(dict)`, never a frame

- `process_fn` branches on the parsed body: a JSON object calls `Executable.run` on a dict, anything else (array, JSONL, CSV, Parquet, Arrow IPC) calls `run` on a frame.
- Why: `run(one-row frame)` took 150 us against `run(dict)`'s 30.4 us (flagship).
- See: [serving latency budget](../experiments/serving-latency-budget.md), `decikit/serving/_handler.py`

## Serving binds the engine its settings name, and builds it before warm-up

- `DECIKIT_API__ENGINE__TYPE` (default `fused`) names the engine and `DECIKIT_API__ENGINE__<FIELD>` sets its own fields (`STRICT_COMPILE`, `PARAMS_VALIDATION`); `decikit serve --engine <tag>` sets the type. `RequestHandler(engine=...)` takes an `Engine` or a tag, resolved once. Serving never reads the `decikit.engine(...)` default. `serve` validates the engine document before it starts workers. Serving takes a `NumbaEngine` only: the polars engine is batch-only, so `RequestHandler` and `serve` refuse it with an `EngineError` naming the numba tags. `APISettings` forbids unknown fields, so the retired `DECIKIT_API__MODE` fails loading instead of being ignored.
- `stage()` builds the kernels (`exe.fallbacks()`) before `warm_fn`, so a `strict_compile` refusal raises as `EngineError` unwrapped (a kernel that splits only at launch is refused during warm-up; `warm_fn` re-raises that `StrictCompileError` unwrapped too); a warm-up failure keeps the sample-request hint.
- Why: `strict_compile` and `params_validation` could not be set for a served project before, and a worker that cannot build its engine dies and is restarted forever. Warming first blamed a strict refusal on the sample request.
- See: [engine](engine.md) ("An engine is a registry document"), `decikit/serving/_handler.py`, `decikit/_tooling/cli/__init__.py`

## A declared output fixes the batch response's shape

- A batch response (a JSON array, JSONL, Parquet, Arrow) is a frame, so each row of a declared `TypedDict` or `list[TypedDict]` output carries every declared key, `null` where the step left an optional key out, in the declared order; a one-record response returns the value as the step built it. Decided by the owner, 2026-10-10.
- Why: before, a batch's struct fields were whichever keys some row in that batch happened to hold, so the same request answered differently depending on its neighbours, and the polars engine could not equal the numba engines. Cost: consolidation's served batch gained null keys (`unpriced_reason`, unused rejection codes) when its outputs were declared; no per-record response changed.
- See: [engine](engine.md) (the `each` bullet), [flexible engine comparison](../experiments/flexible-engine-comparison.md)

## A `NotRequired` key a JSON array leaves out is absent for the step, as in a record

- For an input declared with `NotRequired` keys, `coerce_frame` validates the frame with the nulls polars filled in removed, and `State.from_frame` hands the step the item without the key; a null for a required key is a 400 naming the row and field.
- Why: a record that leaves the key out gave the step an absent key, but the same item inside a JSON array reached it as `None`, so one request answered differently as a record and as a batch.
- Not: treating null as absent for every key (it would hide a null in a required field).
- See: `tests/serving/test_frame_requests.py`, [serving](../../architecture/serving.md) (Dates from JSON)

## The handler hooks are synchronous, and `score` is the in-process entry

- `init_fn`, `input_fn`, `process_fn` and `shutdown_fn` are plain methods; a subclass that defines one with `async def` gets a `TypeError` when the class is created. `score(record_or_frame)` is `process_fn` without the bytes. `score` is not a hook: `process_fn` calls `RequestHandler.score` itself, so a subclass's own `score` (a replay service's, say) never changes what a request does.
- Why: nothing in the default path awaits, a coroutine cost a frame and an event loop per in-process call, and in-process code should not go through the HTTP layer's bytes machinery.
- The Starlette and Sanic endpoints stay `async def` and call the hooks directly, with no thread hop: single-record latency comes first.
- See: `decikit/serving/_handler.py`

## JSON is decoded with `pydantic_core.from_json`

- It answers the same values as `json.loads` about 4x faster (0.90 us against 4.08 us on flagship; `process_fn` 49.9 -> 39.4 us).
- One visible change: a lone surrogate (`"\ud800"`) is rejected with a 400.
- See: `decikit/serving/_parse.py`

## Arrow IPC is accepted for batches, not recommended for one record

- `application/vnd.apache.arrow.stream` / `.file` are input handlers and the stream is an output formatter, through polars with no pyarrow dependency.
- Why: for one record `read_ipc_stream` costs 17 us and `write_ipc_stream` 40 us against 0.90 us and 1.16 us for JSON; Arrow pays off from about 4096 rows.
- See: [Arrow codec and transports](../experiments/arrow-codec-and-transports.md), `decikit/serving/_format.py`

## No Arrow-native, Flight or socket server, and no pyarrow dependency

- Not: a specialised server. The codec is 2-6% of a request, so a free codec saves about 2 us; the long-lived pyarrow record-batch stream measured slowest of every transport except HTTP.
- Not: pyarrow (146.7 MiB installed against polars' 7.9, a second Arrow runtime beside the shim). A co-located caller should `import decikit` and call `run()` with a dict; a remote caller's network dwarfs the rest.
- See: [Arrow codec and transports](../experiments/arrow-codec-and-transports.md)

## HTTP framing is the largest transport term, and the one lever is deployment

- A no-op route costs 207 us of HTTP + ASGI framing against 18 us over a unix socket and 3.3 us over shared memory; httptools + uvloop did not reliably beat h11 + asyncio on a loaded box.
- Any faster transport keeps HTTP and the `RequestHandler` hooks; nothing in the CLI serves a unix socket yet (untested idea: `uvicorn --uds`).
- See: [Arrow codec and transports](../experiments/arrow-codec-and-transports.md), `docs/architecture/serving.md`

## A version is staged off the request path, swapped by pointer, read once per request

- `stage()` builds and warms a version, `activate()` swaps under a lock, `rollback()` pops the previous one; `module_fn` reads the active version once per request, so a request never mixes two versions.
- Why: 0 of 1,564 batches straddled over 11.6k swaps (a control re-reading per chunk straddled 1,576 of 1,578); a swap costs 0.18 µs, the first call after it 7.3 µs against 3.5.
- Not: a background compile thread. numba compiles one function at a time (two threads: 1.04x) and a `nogil` kernel then loses 5.2 ms per call to the GIL hand-off, 26% of throughput. Run `decikit build` in its own process first so `stage()` loads cached user steps.
- See: [experiment](../experiments/stage-a-version-while-serving.md), `decikit/serving/_handler.py`

## Not landed: the in-process engine floor for `run(record)`

- `run(record)` still builds a `State`, per-dtype arrays and output buffers each call: 39.4 us for a whole flagship request against about 7 us with everything pre-bound (tree: 77.2 against about 18).
- A fix must key a per-(plan, thread) cache on plan identity, cache only `n == 1`, follow the per-thread `FrameView` pool, and never hand a cached `State` to a `Session`.
- See: [serving latency budget](../experiments/serving-latency-budget.md), `docs/architecture/engine/concurrency.md`

## Strings for one record are described from their own bytes

- Up to 32 values (`run(record)`), `spans()` points at each value's own `bytes` object; above that it exports the frame's column.
- Why: building spans through a frame cost 71.7 us for one 3-byte string (gated tree `run(record)` 136 us, now 82 us, against 63 us ungated).
- See: [string representations](../experiments/string-representations.md), `decikit/_engine/run/representations.py`, `docs/architecture/engine/boundary.md`

## A frame is read through one `__arrow_c_stream__` export and one C call

- `sm_columns` reads every declared column of the batch into one buffer via a small C shim over vendored nanoarrow 0.9.0; polars rechunks on export, so addresses taken before binding are stale.
- Why: the per-row gather it replaced was O(rows x columns): fused `run()` on flagship took 97.6 ms at 1M rows against 3.1 ms from `to_numpy()` columns.
- Not: one nanoarrow call per element (59-90 µs for 17 columns at one row against 43 µs with descriptors resolved once), nor the hand-decoded numba reader; nanoarrow is adopted for the layout code, not for safety.
- Open: `sm_get_string` uses `ArrowArrayViewGetStringUnsafe` and nanoarrow 0.9.0 does not validate string-view elements, so a corrupt buffer index or offset is a segfault, not an error. Every frame is trusted to come from polars.
- See: [reading a polars frame in a kernel](../experiments/reading-a-polars-frame-in-a-kernel.md), `decikit/_engine/boundary/_arrow/c/shim.c`, `docs/architecture/engine/boundary.md`

## Columns of 4096 rows or more with no nulls are borrowed, not copied

- A float64 / int64 / int32 column the kernel reads as it is becomes a read-only view of the Arrow buffer, owned by a moved `ArrowArray` (`_Export`); below `_BORROW_ROWS = 4096` a copy is cheaper.
- Lifetime rule: anything that hands a raw address to a kernel must stay reachable from a live Python object for the whole call, and nothing may rechunk or mutate the owner meanwhile.
- A socket read buffer reused across requests with `nogil=True` kernels in a thread pool is a use-after-free (a segfault, not a wrong answer): one buffer per in-flight request.
- See: `decikit/_engine/boundary/_arrow/view.py`, `docs/architecture/engine/boundary.md`

## Nullable columns are still copied; zero-copy validity is not adopted

- Validity becomes a byte mask per row. Reading `(bitmap, offset)` in the kernel made null-heavy wide frames 6-7x faster at 1M rows but touched about 17 boundary tests and the kernel ABI, and was left behind a flag that is no longer in the code.
- `run(dict)` never goes through polars/Arrow (126 us against 30 us on flagship); it builds the same `(values, validity)` arrays from the dict.
- Cheap batch wins left: `Enum.gather` for `Literal` outputs (about -40 ms per 1M tree rows) and `Series.set(mask, None)` for nullable outputs.
- See: [zero-copy Arrow gather spike](../experiments/zero-copy-arrow-gather-spike.md)

## The shim is built on first import with `$CC`; there is no pure-Python path

- A plain shared library loaded with `ctypes`, compiled into `$XDG_CACHE_HOME/decikit/arrow-shim/shim-<hash>.so`, where the hash covers sources, flags, compiler and machine; the write is atomic, and nanoarrow is namespaced (`DecikitArrow`) so pyarrow or adbc cannot interpose.
- Why: hatchling has no C-extension support; building on import needs only a compiler, not CPython headers.
- A machine with no compiler gets one `ImportError` naming platform and cause (`_arrow.diagnose()` reports it without raising). Open: a read-only home or no compiler needs the `.so` built at image build time or a prebuilt wheel; Windows is unsupported.
- Not yet: the shipped alternative, a `cp310-abi3` extension wheel (one per platform) built in a manylinux container with `optional` off. A wheel built on a current dev box links glibc 2.38 and cannot be tagged `manylinux_2_28`; `optional = true` hides a missing binary; compile on install also needs `Python.h`. See [shipping the Arrow shim](../experiments/shipping-the-arrow-shim.md).
- See: `decikit/_engine/boundary/_arrow/_build.py`, `docs/architecture/engine/boundary.md`

## Outputs are one column array per name

- `State` holds a numpy array per produced name and each becomes a polars series; there is no record array and no row-major output block.
- Why: for 633 outputs at 100k rows, record-out write-back was 776 ms of 1,212 ms against 5.7 ms column-major, and compiling a record type took 11.3 s against 0.27 s. A wide batch is bounded by memory (11.7 KB/row at 400 in / 633 out), so callers slice at about 100k rows; the engine does not chunk.
- See: [output write-back and batch size](../experiments/output-writeback-and-batch-size.md), `decikit/_engine/run/state.py`

## Kernels take shim addresses as arguments, never captured constants

- The gather, string and `load_*` functions are called through `uint64` address arguments.
- Why: a captured symbol address makes a `cache=True` kernel uncacheable; `tests/boundary/test_shim.py` checks the disk cache with a second process.
- See: `decikit/_engine/boundary/_arrow/intrinsics.py`

## The image is built in stages, and a project is baked in only when asked

- A builder stage (full Python image, has a compiler) installs the repository with the `serve-sanic` extra into a venv and compiles the Arrow shim; the runtime stage (slim, non-root `app`, no compiler) copies the venv and the shim. The default build needs no project and no pre-built wheel; a mounted project at `/opt/ml/model/code` is the old contract.
- The `project` target adds a project three ways, all ending in `bake` (the project's `requirements.txt`, then `decikit build`; a no-op when nothing is baked): `--build-context app=<dir>`, `--build-arg PROJECT=examples/<name>` (the sibling example packages go on `sys.path` through a `.pth` file), or the user's own `FROM <runtime>` Dockerfile.
- Why a prebuilt shim works: the shim's cache key covers sources, flags, `cc` as a string and the machine, not the compiler binary, so the one built in the builder loads in the runtime image (checked: no `cc` there, `_arrow.diagnose()` says `loaded`). Why `VERSION` is an argument: hatch-vcs reads git, and the build context has no `.git`.
- The 12 example images share the 675 MB runtime layers; each adds about 50 MB. `DECIKIT_API__ROOT_PATH` and `DECIKIT_EXT__EXTENSION_PATH` of the old image are no longer settings, so the image sets `DECIKIT_API__CODE_PATH` and `DECIKIT_CONFIG__BASEPATH`.
- Rootless podman maps the container's uid 1000 to a subuid, so a mounted folder needs read access for "others" or `--userns=keep-id` (tested: a mode 700 folder fails without it); add `:Z` for SELinux.
- See: `docker/Dockerfile`, `docker/smoke.sh`, `decikit/_engine/boundary/_arrow/_build.py`

## Baking caches the user steps, not the fused kernels

- Measured on one box (28 CPUs, other jobs running, one worker, container start to first `/ping` 200, then 300 requests on one connection):

  | Image | `loan-granting-pricing` | `transaction-fraud` |
  |---|---|---|
  | baked cache | 18.7, 15.4, 16.0 s | 44.4, 37.9, 39.1 s |
  | `NUMBA_CACHE_DIR` pointed at an empty folder | 32.3, 37.8 s | 119.8, 118.7 s |
  | runtime image, project mounted, never built | 30.9 s | not measured |
  | built with `CPU_TARGET=generic` | 15.6, 13.6 s | 33.8, 33.3 s |
  | first request / median request, baked | 12 to 16 / 12 ms | 23 to 26 / 24 ms |

- What is left after the cache hits (about 15 s and 38 s) is the imports, the fused kernels built from LLVM IR in every worker, and the warm-up. Nothing writes those to disk (see the compile cache page), so a worker is not ready until it has done them and the image's `HEALTHCHECK` polls `/ping` (only honoured by `podman build --format docker`).
- Every example answered 200 with its README figures. First requests of `consolidation` (413 ms), `governance-replay` (872 ms), `retail-credit-e2e` (214 ms) and `business-credit-e2e` (115 ms) are slower than the others because they stage something lazily; the rest took 5 to 34 ms.
- The cache is keyed by source path and CPU, so it hits because the project sits at `/opt/ml/model/code` when built and when served, and only on the CPU that built it. Simulating another CPU (`NUMBA_CPU_NAME=generic` at run time on a host-built image) printed the "`decikit build` ran for CPU 'raptorlake' but this host is 'generic'" warning and brought the start back to the empty-cache figures (39 s and 123 s). No warning appears when the image runs on the CPU that built it.
- `CPU_TARGET=generic` is the default: the runtime and project images build and serve with `NUMBA_CPU_NAME=generic` (baseline x86-64, no AVX), so a baked cache hits on any x86-64 CPU. `CPU_TARGET=host` compiles for the build machine.
- Why generic: it costs nothing measurable per record, and host code costs a 2 to 3 times slower start on any other CPU model. Per record (`Executable.run` on a dict, p50, 3 alternating processes, outside a container): generic against host -1.3% to +0.5% on five examples (affordability, shared-credit-core, loan-granting-pricing, collections, transaction-fraud; geometric mean -0.2%, best of 3). In containers (one worker, 2 CPUs, 300 requests over HTTP, 3 alternating starts): HTTP p50 generic against host -1.1% (loan-granting-pricing, 8.2 ms), +0.4% (transaction-fraud, 17.4 ms), -0.7% (collections, 13.1 ms); start to `/ping` the same (12.2, 31.1, 9.7 s). The host image under `NUMBA_CPU_NAME=generic` (another CPU, simulated) warned and took 28.5, 99.3 and 24.9 s. Only this CPU (raptorlake) was tried.
- `NUMBA_CPU_NAME` is chosen by a stage, not an `ENV ${ARG}`: an empty value makes numba name an empty CPU but keep the host features.
- Not verified: the docker daemon, a second CPU model, an internal index through `PIP_CONF`.

