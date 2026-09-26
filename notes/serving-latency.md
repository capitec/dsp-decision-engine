# Serving latency: where a single request's microseconds go, and whether Arrow can take them back

**Question.** If we really cared about very low latency, could the serving layer
take an Arrow payload directly and run it with as little encoding and decoding
as possible — even if that needed its own specialised server?

**Answer.** It could, and it should not. The codec is **2–6% of a request**, so
a codec that cost literally zero would save ~2 µs of 39.4. Arrow IPC is a
*worse* codec than JSON for one record by 3–20x whichever library reads it,
and the long-lived record-batch stream that was supposed to fix that measured
**slowest of every transport except HTTP**. Meanwhile the two terms that do
cost something are the HTTP server (**207 µs** of framing for a no-op route)
and the engine's per-call setup (**30–137 µs against a 7–18 µs floor**), plus
one string representation builder that costs **71.7 µs to describe a 3-byte
string**. None of those needs a new protocol. Details below, then a ranked
recommendation, then a brief for the engine work.

To be precise about what a perfect Arrow path would buy, since this will be
proposed again: a persistent pyarrow `RecordBatchStreamReader` *does* give real
schema-once semantics (polars cannot), and it *is* the fastest Arrow decode
measured, at 2.9–6.5 µs. It is still 3–6x `pydantic_core.from_json`'s
0.94 µs, the values then cost another 11.6–20.4 µs to get out of the batch, and
the transport it needs is 3–10x a 4-byte length prefix. **Even a zero-cost
codec is worth ~2 µs; this one is worth −15 µs.** That is the argument, and it
does not depend on any library limitation.

## How this was measured

- `benchmarks/serving_latency.py` — the in-process budget and the floor.
- `benchmarks/serving_transports.py` — every transport from the caller's side.
- `benchmarks/arrow_codec.py` — what each codec costs in-process
  (`uv run --with pyarrow python benchmarks/arrow_codec.py`).
- Box: 28 cores, load average ~8.5, swap full, `powersave` governor, other
  agents running the test suite at the same time. Timer overhead p50 0.10 µs.
  Every number is the **lowest p50 and p99 of 3 reps** of 2–20k calls: on a
  contended box the minimum is what the term costs when nothing preempts it.
  Absolute values move ±30% between sessions; the ratios do not, and the
  ratios are the finding. Pinning with `taskset` made things 2.4x *worse*
  (the chosen core was busy), so nothing is pinned.
- Pipelines: **flagship** (5 float inputs, 3 scalar steps, fused),
  **tree** (18 inputs, a 127-node v3 tree document), **gated** (the same tree
  behind one `starts_with` string match). The tree document is the one
  `benchmarks/tree_walk.py` builds, which is now importable.

## Phase 1: the budget

µs, fused mode, one record, no HTTP.

| term | flagship | tree | gated |
|---|---|---|---|
| `json.loads(body)` (what it was) | 4.08 | 12.63 | 13.02 |
| `pydantic_core.from_json(body)` (what it is now) | **0.90** | **2.59** | **2.69** |
| `input_fn` body, sync part | 1.01 | 2.68 | 2.80 |
| `await input_fn` | 1.36 | 3.08 | 3.21 |
| `coerce_record` (no date inputs here) | 0.48 | 0.57 | 0.59 |
| `module_fn` | 0.12 | 0.12 | 0.12 |
| **`score()`** | **30.4** | **62.7** | **136.3** |
| `to_json(result)` | 1.16 | 2.78 | 2.85 |
| `output_fn` | 1.84 | 3.49 | 3.56 |
| `await` of an empty coroutine | 0.18 | 0.18 | 0.18 |
| **`await process_fn`, whole** | **39.4** | **77.2** | **151.8** |
| the same before the `from_json` swap | 49.9 | 93.5 | 170.9 |

`score()` split into stages, each the next difference of a cumulative prefix:

| stage | flagship | tree | gated |
|---|---|---|---|
| `State(plan, _NO_FRAME, 1)` | 1.17 | 1.14 | 1.16 |
| `_load` (dict → one numpy block per dtype) | 6.68 | 19.4 | 22.2 |
| `_params` bundle lookup | 1.54 | 1.64 | 1.13 |
| `runner.iterate` (Python driver + kernels) | 18.9 | 34.9 | **104.1** |
| output dict | 2.28 | 4.63 | 6.94 |

Three things fall out of this table.

- **The codec is a rounding error.** `from_json` + `to_json` is 2.1 µs of 39.4
  on flagship, 5.4 of 77.2 on the tree and 5.5 of 151.8 on the gated tree — 3–5%.
  A codec that cost literally zero would save that much. That is the whole
  answer to the Arrow question, and everything below only confirms it.
- **`process_fn` costs ~8–12 µs more than the sum of its parts.** It is not
  async (an empty `await` is 0.18 µs) and not the collector (`gc.disable()`
  changes nothing): the same work written as one synchronous function costs
  35.7 µs against a ~35 µs sum, so it is allocator churn from the fresh dicts,
  arrays, tuples and `Response` every call.
- **`iterate` is a Python driver around a tiny kernel.** 18.9 µs to run a
  2.1 µs kernel on flagship; 104 µs on the gated tree, almost all of it
  `spans()` (see below).

The same script also prices the Arrow codec at each width, in-process:

| | flagship | tree | gated |
|---|---|---|---|
| `pl.read_ipc_stream` | 17.1 | 30.2 | 32.0 |
| the same with the schema message cached server-side | 17.3 | 31.1 | 32.3 |
| `+ df.row(0, named=True)` | 20.0 | 36.8 | 38.8 |
| `write_ipc_stream` of the result | 40.6 | 78.8 | 81.2 |
| **`exe.run(one-row frame)`** | **150** | **285** | **385** |
| against `score()` | 30.4 | 62.7 | 136.3 |

Body sizes: 1000/3008/3192 bytes of Arrow IPC against 107/453/474 of JSON.

## The floor: same kernels, nothing re-allocated per call

| | flagship | tree |
|---|---|---|
| numba dispatchers alone, buffers pre-bound | 2.13 | 6.19 |
| `unit.run` (kernel + its own per-call allocations) | 5.51 | 16.9 |
| **whole request, nothing allocated**: `from_json` → write into pre-bound arrays → dispatchers → read out → `to_json` | **6.98** | **18.2** |
| today | 39.4 | 77.2 |

So **5.6x (flagship) and 4.2x (tree) is available inside the process with no
change to the transport at all.** That is the largest number in this
investigation.

The `gated` case has no floor row on purpose: a kernel reading a `str`/`bytes`
column is handed a representation the runner rebuilds every call, not a buffer
that can be pre-bound. That is the 104 µs `iterate` line, and it is the
single-record cost `notes/strings.md` already flags.

## Phase 2: the codec, measured every way

Getting flagship's five floats out of a request body and into what `score()`
takes. In-process, same session, µs p50.

| how | µs | body bytes |
|---|---|---|
| `struct.unpack_from("<5d", body)` — fixed layout, schema agreed once | **0.43–0.57** | 40 |
| `np.frombuffer(body, np.float64, 5)` | 0.70–0.85 | 40 |
| `pydantic_core.from_json` (what `parse.py` now does) | **0.94–1.13** | 107 |
| `json.loads` (what it did before) | 4.10 | 107 |
| pyarrow `read_next_batch`, **schema parsed once**, from memory | 2.9–6.5 | 376 |
| pyarrow `open_stream(...).read_next_batch()`, schema per call | 12.5–13.4 | 728 |
| pyarrow over a pipe: `write_batch` + flush + `read_next_batch` | 17.3–17.5 | 376 |
| `pl.read_ipc_stream` (polars, schema per call) | 17.1–18.3 | 952 |

…and then the values still have to come out of the batch:

| how | µs |
|---|---|
| `df.row(0, named=True)` after `pl.read_ipc_stream` | 3.2 |
| `[batch.column(i)[0].as_py() for i in range(5)]` | 11.6–12.4 |
| `np.frombuffer` of each column's data buffer (no copy) | 12.8–18.1 |
| `batch.to_pydict()` | 19.3–20.4 |

Writing the reply is worse again: `write_ipc_stream` of the one-row result is
**40.2 µs** against `to_json`'s **1.16 µs**.

**The schema-once claim is confirmed, and it does not save the day.**
- In **polars** there is no incremental reader at all. Caching the schema
  message server-side and prepending it to each batch measures 17.20 µs
  against 17.13 µs with the schema on the wire — identical, because polars
  re-parses it every call. The only saving is 296 bytes of the 952 on the wire.
- In **pyarrow** a persistent `RecordBatchStreamReader` does give real
  schema-once semantics, and it is worth ~7–10 µs (2.9–6.5 against
  12.5–13.4). That is a polars limit, not an Arrow one.
- But even 3 µs of decode is 3x `from_json`, and extracting the five values
  costs another 11.6–20.4 µs. **A best-case pyarrow path is ~15–25 µs against
  JSON's 2.1 µs.** Arrow's per-batch metadata (a flatbuffer with a node and a
  buffer entry per column, and a length, offset and null count per node) is
  simply more work than five JSON numbers, and none of it amortises over one
  row.
- Reading the same batches through a **socket** rather than memory costs
  17 µs in-process and 55–194 µs over a real socket (see
  `*_pyarrow_stream, echo` below), because pyarrow's reader goes through a
  Python file object and calls back into Python for every message. The
  long-lived Arrow stream — the design that was supposed to win — is the
  **slowest** non-HTTP transport measured.
- Once the schema is fixed and sent once, **an Arrow record batch and a
  hand-rolled fixed binary frame are the same bytes with a different header**.
  The 40-byte `struct.unpack` row above *is* the zero-encoding Arrow path,
  read without Arrow's libraries. Everything Arrow adds at n=1 is header.

Arrow earns its keep from ~4096 rows up, which is exactly where the boundary
already borrows buffers (`_BORROW_ROWS` in `_arrow/view.py`) and where
`notes/zero-copy-arrow-investigation.md` measured 6–7x. Nothing here argues
against Arrow for batches.

### pyarrow as a runtime dependency

Not worth it for this, and expensive if it were: **146.7 MiB installed**
against polars' 7.9. The wheel also brings its own Arrow C++ runtime, a second
Arrow implementation beside the vendored nanoarrow shim, and a second set of
CPU/glibc constraints for the deployed image (see
`notes/packaging-first-binary.md` for why that matters here). polars reads and
writes Arrow IPC with no pyarrow (`pl.read_ipc_stream`, `write_ipc_stream`),
which is what the new content types use, so **pyarrow is still not a
dependency.**

## The transports, from the caller

Same flagship pipeline, same session, one record. `echo` rows score nothing, so
the transport's own cost separates from the handler's. Clients are raw sockets
with keep-alive (no `httpx`), so no client library is in the numbers. The raw
servers are single-connection blocking loops — the floor a specialised server
could reach.

| transport | p50 µs | p99 µs | body B |
|---|---|---|---|
| `http_json` — **today** (h11 + asyncio, what `serve-starlette` installs) | 614.5 | 1225 | 107 |
| `http_json`, httptools + uvloop | 583.6 | 1138 | 107 |
| `http_arrow_ipc` (the new content type) | 1290.5 | 2173 | 1000 |
| `http_ping` — HTTP + ASGI framing only, no handler | 207.3 | 736 | 8 |
| `http_ping`, httptools + uvloop | 216.7 | 749 | 8 |
| `tcp_json` (4-byte length prefix, no HTTP) | 193.5 | 671 | 107 |
| `tcp_arrow_ipc` | 471.8 | 1080 | 1000 |
| `tcp_arrow_ipc` → `exe.run(df)` instead of `score()` | 778.8 | 1360 | 1000 |
| `tcp_arrow_batch` (schema cached server-side, batch only) | 285.0 | 732 | 656 |
| `tcp_values` (schema agreed on connect, 5 raw float64) | 155.4 | 574 | 40 |
| `tcp_pyarrow_stream` (persistent `RecordBatchStreamReader`) | 613.6 | 1084 | 376 |
| `tcp_echo` — transport only | 42.3 | 119 | 8 |
| `uds_json` | 138.6 | 450 | 107 |
| `uds_arrow_ipc` | 318.5 | 889 | 1000 |
| `uds_arrow_batch` (schema cached) | 141.8 | 562 | 656 |
| `uds_values` | 131.3 | 466 | 40 |
| `uds_pyarrow_stream` | 460.1 | 992 | 376 |
| `uds_echo` — transport only | 18.3 | 112 | 8 |
| `uds_values`, client spin-polls instead of blocking | 69.8 | 150 | 40 |
| `uds_echo`, spin — transport only | 12.0 | 21.8 | 8 |
| `shm_values` (shared memory, both sides spin) | **43.9** | **51.5** | 40 |
| `shm_echo` — transport only | **3.3** | **3.7** | 8 |
| `tcp_pyarrow_stream`, echo — transport only | 193.8 | 668 | — |
| `uds_pyarrow_stream`, echo — transport only | 55.1 | 340 | — |

Read this table by the `echo` rows, which are the transport with no scoring:
**3.3 µs shared memory, 12–18 µs a unix socket, 42 µs loopback TCP, 207 µs
HTTP + ASGI, 55–194 µs a pyarrow stream.** The handler delta (scored minus
echo) is 41–113 µs everywhere, which is the engine cost from Phase 1 plus the
server's own read/write. So:

- **HTTP framing is the largest single term in a real request** — 207 µs of
  framing against 30–95 µs of engine and ~2 µs of codec. Dropping HTTP for a
  unix socket is worth ~190 µs; making the codec free is worth ~2.
- **Arrow IPC makes every transport slower**, by +180 µs (uds) to +676 µs
  (http), because of `read_ipc_stream` (17 µs), `write_ipc_stream` (40 µs) and
  a 9x larger body. `uds_arrow_batch` looks much better than `uds_arrow_ipc`
  (141.8 vs 318.5) but that is **not** the cached schema: it also replies with
  40 raw bytes instead of a 1000-byte IPC stream, which is where the saving is.
  In-process, caching the schema measured 17.20 vs 17.13 µs — nil.
- **The long-lived pyarrow record-batch stream is the slowest option except
  HTTP.** It is the design the question was really about, and it loses to a
  4-byte length prefix and `struct.unpack` by 10x.
- httptools + uvloop is inside the noise here. Over five alternating
  `http_ping` pairs it won four (242–292 µs against 294–368 µs) and lost one
  outlier (99 µs) when the box briefly went quiet; that 99 µs is probably the
  real uncontended figure for both. **Worth measuring on a quiet box before
  changing the extra; don't assert a win from these numbers.**

Run-to-run spread on this box, same config: `http_ping` 99–368, `tcp_echo`
26–54, `uds_echo` 12–18, `shm_echo` 3.3–6.7. The table is one session; the
ordering held in every session.


### What each one costs operationally

| transport | verdict |
|---|---|
| **HTTP + JSON** (today) | Works with every client, every load balancer, SageMaker's `/invocations` contract, curl. Served by the existing `RequestHandler` hooks unchanged. |
| **HTTP + Arrow IPC body** | Now supported: `application/vnd.apache.arrow.stream` is registered in `DEFAULT_INPUT_HANDLERS` and `DEFAULT_OUTPUT_FORMATTERS`. No new server, no new dependency. **Slower than JSON for one record** and faster only for batches, which is what it is for. |
| **Arrow Flight (gRPC)** | Not built, and now costed from measurement rather than argument. Flight is a long-lived `RecordBatchStream` over gRPC, so its floor is at least the `*_pyarrow_stream` rows: **55–194 µs of transport alone** before gRPC's HTTP/2 framing, against 12–18 µs for a plain unix socket and 3.3 µs for shared memory. Add pyarrow (146 MiB), a new port, a new client SDK, and no SageMaker `/invocations` contract. Flight is built for streaming many batches between data systems; at one row per call it pays all of its overhead and amortises none of it. |
| **Raw Arrow IPC over a unix socket** | Measured (`uds_arrow_ipc` 318.5, `uds_pyarrow_stream` 460.1). The unix socket helps; Arrow hurts. What is fast here is dropping HTTP, not using Arrow — and it needs its own server, its own framing, a co-located client, and it cannot be load-balanced or health-checked by anything standard. |
| **Fixed binary frame over a unix socket, schema agreed on connect** | The fastest realistic option, and the honest form of "Arrow with the schema sent once". Still needs its own server and a bespoke client. |
| **Shared memory + spin** | The genuine floor. Requires a co-located caller in the same container, burns a core per waiting side, has no timeout, no backpressure and no multi-tenancy, and one stale pointer is a segfault rather than a wrong answer. |

## How far can copying actually be avoided?

Tracing an incoming Arrow buffer to a kernel, in the code as it stands:

1. `pl.read_ipc_stream(body)` → polars owns the result, and **it copies**.
   Reading the *same* `bytes` three times gives three different data
   addresses (`0x7f7d5b8a49c0`, `0x…be540`, `0x…ca240`), so each read
   allocates; polars does not retain or alias the caller's buffer. **The
   caller's buffer is already dead by the time the frame exists** — the first
   copy happens before decider sees anything, and nothing in `decider/` can
   prevent it. (`read_ipc(memory_map=True)` can borrow, but only from a file,
   and a socket is not a file.) So "an incoming Arrow buffer reaching a kernel
   with no copy at all" is not achievable through polars at any row count; it
   would need the IPC message decoded by the shim directly into an
   `ArrowArray`, which is a new ~200-line C path and an nanoarrow_ipc vendor
   bump for a gain of, at n=1, nothing.
2. `State.from_frame` → `extract_frame` → `FrameView.bind(df)` →
   `df.__arrow_c_stream__()`. polars **rechunks in place** on export, so any
   address taken before `bind` is stale after it.
3. `sm_import_frame` + `sm_resolve_all` (nanoarrow) resolve each column, then
   `view.columns()` makes one allocation and one C call. A numeric column
   already in the kernel's dtype, with no nulls, and **`n >= 4096`
   (`_BORROW_ROWS`)** is borrowed: a `_Borrowed` numpy view over the exported
   address, owned by an `_Export` that moved the `ArrowArray` out. Below 4096
   rows it is copied, because at that size a copy is cheaper than wrapping.
4. Validity today becomes a byte mask per row. For a genuinely copy-free
   nullable column the kernel ABI has to take `(bitmap, bit offset)` — built
   and measured in `notes/zero-copy-arrow-investigation.md`, **not adopted**.
5. A string column is `(offsets, data)`; the boundary turns it into an
   `(n, 2)` int64 table of `(address, length)` spans. The string bytes are
   never copied; the spans are (16 bytes a row).

**So at n=1 the answer is: copying can be avoided completely, and it buys
nothing.** One flagship record is 40 bytes of values. `memcpy` of 40 bytes is
single-digit nanoseconds. Every microsecond in the boundary is Python and
per-column bookkeeping — hashing the `Input` tuple for the plan cache, a numpy
view and a NamedTuple per column — which a zero-copy design does not remove.
`notes/zero-copy-arrow-investigation.md` reached the same conclusion from the
other side: `score()` through polars/Arrow costs 126 µs against 30, and its
floor is still ~20 µs above today's path.

### The lifetime rule

> Anything that hands a raw address to a kernel must be reachable from a live
> Python object for the whole kernel call, and nothing may rechunk, cast,
> slice or mutate the owning frame in between.

Concretely, in this code: `_Export` owns the moved `ArrowArray` and
`_Borrowed.owner` holds it; `ExtractedFrame.kernel_frame` holds the frame a
`bytes` input's spans point into; `State._representations` keeps an `alive`
list per representation and drops it whenever the values are written again;
`FrameView.release()` invalidates every span it handed out, which is why
`FrameView`s are pooled per thread and popped while in use.

For an Arrow-over-socket server the rule has a sharp edge: **the buffer read
off the socket may not be reused for the next request while a kernel still
holds a pointer into it.** Serving kernels are `nogil=True`, so with a thread
pool a shared read buffer is a use-after-free — a segfault, not a wrong
answer. Either one buffer per in-flight request, or a strictly synchronous
single-threaded loop. A specialised server that reuses one buffer *and* runs a
thread pool is the exact bug this rule exists to forbid.

## Dictionary-encoded strings: can the encoding step disappear?

Partly, and it is worth less than it looks.

What is already true:
- `str` is `FeatureKind.CODE`, an int32 dictionary code, and
  `_NATIVE[CODE] = (Categorical, Enum)` in `boundary/dtypes.py`: a
  dictionary-encoded column is read **natively**, "nanoarrow reads the frame's
  own buffers". `view.dictionary(name)` exports the dictionary.
- `pl.read_ipc_stream` returns `Categorical` for a dictionary-encoded Arrow
  column, so a caller sending dictionary-encoded strings does hand over the
  codes in the right physical shape.

What blocks it:
1. **The engine never routes a string column that way.** `TYPED = (float, int,
   bool)` in `engine/ir/decls.py`, so `State.from_frame` sends only numeric
   columns through `extract_frame`; every string column goes through
   `from_series` into a Python object array, and the runner rebuilds the
   representation per call (`StringCodes.encode` under a lock for
   `RAW_STRING`, `spans()` for `RAW_BYTES`). `ExtractedColumn.categories` is
   populated and then read by nobody in `decider/`.
2. **Two code tables.** `StringCodes` is process-global and monotonic; a
   `str` **param** is encoded into it (`SteppedRunner._bundle`). An Arrow
   dictionary is **per batch**. A kernel comparing an input code against a
   param code needs both from the same table, so the caller's dictionary has
   to be remapped to the global one: one dict lookup per *distinct category*
   per batch, memoisable on the dictionary's identity — not per row. The
   per-row encoding really does disappear; the per-request work does not go to
   zero.
3. **Trees want bytes, not codes.** A plain `str` input inside a row node is
   `RAW_BYTES`, i.e. spans, because `starts_with` / substring matching needs
   the bytes. A dictionary index is not an address. So dictionary encoding can
   remove the work only for **equality** comparisons; prefix and substring
   matching still needs `spans()`. Equality on a code is the case
   `Raw[str]` already serves.

And the size of the prize at n=1, measured directly:

| per call, one value | µs |
|---|---|
| `StringCodes.encode` (the `Raw[str]` / CODE path) | **9.7** |
| `spans()` (the `Raw[bytes]` / STR path, what trees use) | **71.7** |

Both are far bigger than the whole JSON codec, and both are pure Python
overhead on three bytes of data.

- **9.7 µs for one dictionary code is a real prize, and a caller's dictionary
  could take most of it.** It is also mostly self-inflicted:
  `StringCodes.encode` does `self._codes.update(raw_string_codes())` on every
  call, which copies the entire process-global code dict, under a lock, to
  encode one string. That is worth fixing whatever happens to the transport.
- **71.7 µs for one byte span is the biggest single item in this whole
  investigation after HTTP, and a dictionary cannot touch it**: a dictionary
  gives an index, a tree needs an address. It is ~8 numpy calls plus
  `ctypes.data` to describe one 3-byte string, and it is 2.4x the cost of an
  entire flagship request. It accounts for essentially all of the gated tree's
  104 µs `iterate`.

So the honest answer to "can the string encoding step disappear if the caller
sends dictionary-encoded strings": **for equality comparisons yes, worth up to
~9.7 µs a call once the two code tables are reconciled; for prefix/substring
matching no, and that is the expensive case.** Neither depends on the
transport — the win is in `representations.py`, not in the wire format.

Worth writing down separately: **for a batch, a caller that sends
dictionary-encoded strings is handing over exactly the representation the
boundary wants, and today the engine throws it away and rebuilds it.** That is
a real batch win sitting behind `TYPED`, and it belongs with the string work.

## What a caller would see, and who would use it

Today, unchanged, and already enough for an Arrow-native caller:

```
POST /invocations
Content-Type: application/vnd.apache.arrow.stream
Accept: application/vnd.apache.arrow.stream
<Arrow IPC stream>
```

That needs no new server and no pyarrow on either side (polars writes the
stream). It is the right thing for a **batch** caller — a Spark or polars job
scoring a file — and it is the wrong thing for a single record, where the same
caller should send JSON.

The specialised server, if anyone wanted it, would be: a unix socket, a
handshake that names the input order and dtypes once, then `n*8` bytes in and
`m*8` bytes out per call. Who could actually adopt that? Only a process in the
same container as the model server — a sidecar, or an in-process caller. And a
caller in the same container should not be using a socket at all: it should
`import decider` and call `score()`, which is 30 µs today and 7 µs at the
floor, against 55 µs for the best socket. **Every client that genuinely needs a
socket is remote, and for a remote client the network is 100 µs–10 ms and the
entire question is moot.** That is the reason not to build it, and it does not
depend on any number above.

## Recommendation, ranked

1. **Take the engine floor.** 39.4 → ~7 µs (flagship), 77.2 → ~18 µs (tree),
   no protocol change, no new dependency, no client change. Brief below.
2. **Fix the two string representation builders.** `spans()` costs 71.7 µs and
   `StringCodes.encode` 9.7 µs *per call for one value*, both pure Python
   overhead. Together they are the whole gap between the gated tree (136 µs)
   and the plain tree (63 µs). Bigger than every transport change here.
3. **Done: decode with `pydantic_core.from_json`.** `process_fn` 49.9 → 39.4
   (flagship), 93.5 → 77.2 (tree), 170.9 → 151.8 (gated): −11% to −21% of
   every JSON request, for one import.
4. **Measure HTTP framing on a quiet box, then decide about the server.** It
   is 207 µs of a 615 µs request here, which makes it the largest term in
   anything a real client sees — but httptools + uvloop did not reliably beat
   h11 + asyncio in these conditions, and one quiet-moment sample suggests
   both are ~100 µs uncontended. The cheap experiments, in order: rerun
   `http_ping` unloaded; if uvloop wins, change `serve-starlette` to
   `uvicorn[standard]` (one line); and count the ASGI layers per request,
   since 190 µs above `tcp_echo` for a no-op route is Python, not the kernel.
   (`pyproject.toml` is outside my diff scope; flagged, not changed.)
5. **Serve the existing HTTP app over a unix socket** (`uvicorn --uds`) before
   anyone writes a new protocol: `uds_echo` is 18 µs against `tcp_echo`'s 42
   and HTTP's 207, it is a deployment flag, it keeps HTTP and the
   `RequestHandler` hooks unchanged, and it captures most of what the bespoke
   socket was for.
6. **Done: accept and return Arrow IPC** (`application/vnd.apache.arrow.stream`,
   `application/vnd.apache.arrow.file`), for batch callers, with no pyarrow.
   Do not recommend it for single records.
7. **Do not build an Arrow-native or Flight server, and do not add pyarrow.**
   Best case ~15–25 µs of codec against JSON's 2.1, on a term worth 2 µs, for
   146 MiB, a new port, a new client SDK and a protocol nothing off-the-shelf
   speaks. The long-lived batch stream — the one design that was supposed to
   win — measured **slowest of everything except HTTP**.
8. **Do not route `score()` through a frame.** `exe.run(one-row frame)` is
   153 µs against `score()`'s 30 µs. Now called out in `GUIDE.md`.

### One client-visible change to record

`from_json` rejects a lone surrogate (`{"a": "\ud800"}`) where `json.loads`
accepted it. Every other case checked — NaN, ±Infinity, `1e400`, 24-digit
ints, duplicate keys, `1.0` staying a float, arrays, malformed input — gives
identical results, and both raise a `ValueError` subclass so `input_fn` still
answers 400. The surrogate is stricter and would have failed later at the
Arrow boundary, but it is a behaviour change.

## Brief: the engine floor (for whoever owns `run/`)

The prize is 5x on the 90% workload. Everything below is measured, not
estimated; `benchmarks/serving_latency.py` reproduces it.

**Where the time goes in `Executable.score` (flagship / tree, µs):**

| | flagship | tree | gated |
|---|---|---|---|
| `State(plan, _NO_FRAME, 1)` | 1.17 | 1.14 | 1.16 |
| `_load` | 6.68 | 19.4 | 22.2 |
| `_params` | 1.54 | 1.64 | 1.13 |
| `runner.iterate` | 18.9 | 34.9 | 104.1 |
| output dict | 2.28 | 4.63 | 6.94 |
| kernel dispatchers alone | 2.13 | 6.19 | — |

**What is allocated per call, and what would have to be pre-bound:**

- `State.__init__` builds five fresh dicts (`values`, `valid`, `chains`,
  `_sources`, `_representations`), and `chains` copies every list in
  `plan.chains`. For a fixed plan and `n == 1` this is the same shape every
  call: one reusable `State` per (plan, thread), cleared rather than rebuilt.
- `_load` (`run/engine.py`) builds, per dtype group, a Python list
  comprehension, a null list, `np.array(...)`, sets `writeable = False`, then
  one `block[k:k+1]` view per input. Pre-bound: one writable block per dtype
  per (plan, thread), written in place, with the per-input views made once.
  The benchmark's floor does exactly this and it costs ~0.
- `Kernel.run` (`compile/units.py`) builds per call: a `cols` tuple, an
  `np.ones(n)` when the unit has OPTIONAL inputs, a `valids` tuple, a `params`
  list then tuple, and an `np.empty(n, dtype)` per output and per mask. All of
  these are shape-stable for a fixed plan and `n`; all can be pre-bound and
  reused.
- `runner.iterate` walks `plan.calls` through `_sequence`/`_node`/`_call` with
  a `_Scope` and a generator frame per checkpoint, yielding a `Checkpoint`
  dataclass per node for a debugger nobody is attached to. On flagship that is
  19.4 µs around a 2.1 µs kernel. A non-yielding fast path for
  `score()` — the plan is fixed, `rows is None`, no session — is where most of
  the 17 µs is.
- `output_fn`/the output dict: `values.tolist()[0]` per result allocates a
  list to read one element; `arr[0].item()` does not.

**What invalidates a pre-bound cache** — the list a design has to handle:
- a different `Plan` (`iterate` already re-`_compile`s on `plan is not
  self._plan`); key the cache on plan identity,
- a different `n` (every batch), so cache only `n == 1`, which is the case
  that matters,
- a different thread: serving runs concurrent calls on one `Executable` and
  kernels are `nogil=True`. `FrameView`s are already pooled per thread in
  `extract.py` (`_checkout`/`_checkin`) — follow that pattern exactly, and
  pop while in use so a re-entrant call gets its own,
- a params document change: `_params` already caches by `document_key`; the
  converted bundles in `SteppedRunner._converted` are keyed by
  `(params.key, call_id)` and stay valid,
- a `Session`: `State.restore`, `State.record` and overrides mutate a state
  and rewind past representations. A pre-bound state must not be handed to a
  session; `session()` should keep building fresh ones,
- a string or `Rows` input: `State.representation` rebuilds per call and
  borrows buffers into an `alive` list. Those cannot be pre-bound without
  solving the string question separately — and for `gated` that is 104 µs, the
  biggest single number in the whole budget.

**Reachable:** 6.98 µs flagship and 18.2 µs tree for a whole request,
measured, including `from_json` and `to_json` — 5.6x and 4.2x today. Even
taking half of it is worth more than every transport change in this note
combined.
