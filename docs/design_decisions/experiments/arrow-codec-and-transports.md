# Would an Arrow payload or a specialised server make single requests faster?

Date: 2026-09-26 · Script: `benchmarks/arrow_codec.py` (codecs, in-process), `benchmarks/serving_transports.py` (round trips from the caller)

## Purpose
Test whether taking Arrow directly, over HTTP, a unix socket, a persistent pyarrow stream or shared memory, would beat JSON over HTTP for one flagship record.

## Method
Raw-socket clients with keep-alive (no client library); `echo` rows score nothing, so transport and handler separate. Raw servers are single-connection blocking loops (the floor a specialised server could reach). Lowest p50 of 3 reps on a contended box (load ~8.5); the table is one session, the ordering held in all. A rerun on 2026-10-08 on a quieter box kept it: HTTP + JSON 393.5, HTTP + Arrow 737.3, no-op HTTP route 188.6, TCP + JSON 80.3, unix + JSON 61.1, unix echo 8.4, shared-memory echo 2.4 us; pyarrow rows need `--with pyarrow`.

## Result
| Codec for 5 floats (us) | decode | body B |
|---|---|---|
| `struct.unpack` of a fixed layout | 0.43-0.57 | 40 |
| `pydantic_core.from_json` | 0.94-1.13 | 107 |
| pyarrow, schema parsed once | 2.9-6.5 (+11.6-20.4 to extract values) | 376 |
| `pl.read_ipc_stream` | 17.1-18.3 | 952 |

| Transport (p50 us) | scored | echo only |
|---|---|---|
| HTTP + JSON (h11 + asyncio) | 614.5 | 207.3 (no-op route) |
| TCP + JSON, 4-byte length prefix | 193.5 | 42.3 |
| unix socket + JSON | 138.6 | 18.3 |
| unix socket + Arrow IPC | 318.5 | |
| persistent pyarrow stream (TCP / unix) | 613.6 / 460.1 | 193.8 / 55.1 |
| shared memory, both sides spin | 43.9 | 3.3 |

- Writing the reply: `write_ipc_stream` 40.2 us against `to_json` 1.16 us. Caching the schema message made no difference in polars (17.20 against 17.13 us); it re-parses every call.
- Arrow makes every transport slower, by +180 us (unix) to +676 us (HTTP). Once a schema is agreed, a record batch and a fixed binary frame are the same bytes with a different header.
- The caller's buffer is dead before the decider toolkit sees it: `pl.read_ipc_stream` copies, so a no-copy Arrow request is impossible through polars.

## Decision
No Arrow-native or Flight server and no pyarrow; Arrow IPC stays a batch content type; faster transports mean dropping HTTP, not adding Arrow. See [serving](../decisions/serving.md).
