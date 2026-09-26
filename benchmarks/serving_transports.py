"""Per-request latency of one record over each transport, measured from the caller.

Every row is the same flagship pipeline in the same session. `echo` rows score
nothing, so the transport's own cost is separable from the handler's. The raw
servers are single-connection blocking loops on purpose: that is the floor a
specialised server could reach, with no HTTP and no event loop.

    uv run --with uvicorn python benchmarks/serving_transports.py [--calls 5000]

Needs uvicorn only for the `http_*` rows.
"""
import json
import mmap
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import polars as pl
from pydantic_core import to_json

sys.path.insert(0, "benchmarks")
from serving_latency import FLAGSHIP_ROW, cases, _pct  # noqa: E402

# Per run, so two runs of this script never fight over an address.
TAG = os.environ.setdefault("DECIDER_BENCH_TAG", str(os.getpid()))
PORT = 8000 + int(TAG) % 20000
UDS = f".scratch/bench-{TAG}.sock"   # relative: an absolute path here overruns AF_UNIX's 108 bytes
SHM = f"decider-bench-{TAG}"
NAMES = tuple(FLAGSHIP_ROW)
FRAME = pl.DataFrame([FLAGSHIP_ROW])
CALLS = 5_000
WARM = 500
HEAD = ("POST /invocations HTTP/1.1\r\nHost: b\r\nConnection: keep-alive\r\n"
        "Content-Type: {ct}\r\nAccept: {ac}\r\nContent-Length: {n}\r\n\r\n")


def ipc_bytes(df: pl.DataFrame) -> bytes:
    import io

    buf = io.BytesIO()
    df.write_ipc_stream(buf)
    return buf.getvalue()


BODIES = {"json": json.dumps(FLAGSHIP_ROW).encode(), "arrow": ipc_bytes(FRAME),
          "values": struct.pack(f"<{len(NAMES)}d", *(float(FLAGSHIP_ROW[n]) for n in NAMES)),
          "echo": b"\x00" * 8}
# Codecs that send an existing body under another name.
BODIES["arrow_run"] = BODIES["arrow"]
BODIES["ping"] = BODIES["echo"]


def split_ipc(stream: bytes):
    """(schema message, the rest) of an IPC stream; the schema message carries no body."""
    return stream[:8 + struct.unpack_from("<i", stream, 4)[0]], stream[8 + struct.unpack_from("<i", stream, 4)[0]:]


SCHEMA_MSG, BODIES["arrow_batch"] = split_ipc(BODIES["arrow"])


# ── the scoring side, shared by every server ──────────────────────────────────

def build():
    from decider.config import JsonFileStore
    from decider.serving.handler import RequestHandler

    store = JsonFileStore(basepath=tempfile.mkdtemp(dir=".scratch"))
    store.create_version({"params": {}})
    handler = RequestHandler(store, cases()["flagship"][0], mode="fused")
    handler.stage()
    handler.activate()
    return handler


def codec(kind, handler):
    """bytes -> bytes for one request."""
    if kind == "echo":
        return lambda body: b"\x00" * 8
    live = handler.module_fn()
    exe, params = live.executable, live.params
    keys = tuple(k for k, v in exe.plan.outputs.items() if v.producer is not None)
    pack = struct.Struct(f"<{len(keys)}d").pack
    unpack = struct.Struct(f"<{len(NAMES)}d").unpack_from

    def reply(out):
        return pack(*(float(out[k]) for k in keys))

    if kind == "json":
        return lambda body: to_json(exe.score(json.loads(body), params))
    if kind == "arrow":
        return lambda body: ipc_bytes(pl.DataFrame([exe.score(
            pl.read_ipc_stream(body).row(0, named=True), params)]))
    if kind == "arrow_run":
        return lambda body: ipc_bytes(exe.run(pl.read_ipc_stream(body), params))
    if kind == "arrow_batch":
        # The schema arrived once, on the handshake; polars still needs it in front of every batch.
        return lambda body: reply(exe.score(
            pl.read_ipc_stream(SCHEMA_MSG + body).row(0, named=True), params))
    return lambda body: reply(exe.score(dict(zip(NAMES, unpack(body))), params))


# ── servers ───────────────────────────────────────────────────────────────────

def serve_raw(kind, family):
    handler = None if kind == "echo" else build()
    run = codec(kind, handler)
    if family == socket.AF_UNIX:
        if os.path.exists(UDS):
            os.unlink(UDS)
        s = socket.socket(family, socket.SOCK_STREAM)
        s.bind(UDS)
    else:
        s = socket.socket(family, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", PORT))
    s.listen(1)
    print("ready", flush=True)
    conn, _ = s.accept()
    if family != socket.AF_UNIX:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while True:
        head = _recv(conn, 4)
        if not head:
            return
        body = _recv(conn, struct.unpack("<I", head)[0])
        reply = run(body)
        conn.sendall(struct.pack("<I", len(reply)) + reply)


def serve_pa(kind, family):
    """A long-lived Arrow IPC stream each way: the schema crosses once, then record batches."""
    import pyarrow as pa

    handler = None if kind == "echo" else build()
    if family == socket.AF_UNIX:
        if os.path.exists(UDS):
            os.unlink(UDS)
        s = socket.socket(family, socket.SOCK_STREAM)
        s.bind(UDS)
    else:
        s = socket.socket(family, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", PORT))
    s.listen(1)
    print("ready", flush=True)
    conn, _ = s.accept()
    if family != socket.AF_UNIX:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    rf, wf = conn.makefile("rb"), conn.makefile("wb")
    reader = pa.ipc.open_stream(rf)       # the client primed it with schema + one batch
    reader.read_next_batch()
    if kind == "echo":
        keys, score = ("ok",), None
    else:
        live = handler.module_fn()
        exe, params = live.executable, live.params
        keys = tuple(k for k, v in exe.plan.outputs.items() if v.producer is not None)
        score = lambda r: exe.score(r, params)      # noqa: E731
    out_schema = pa.schema([(k, pa.float64()) for k in keys])
    writer = pa.ipc.new_stream(wf, out_schema)
    writer.write_batch(pa.record_batch([[0.0] for _ in keys], schema=out_schema))
    wf.flush()
    try:
        for batch in reader:
            if score is None:
                reply = pa.record_batch([[0.0]], schema=out_schema)
            else:
                record = {k: v[0] for k, v in batch.to_pydict().items()}
                result = score(record)
                reply = pa.record_batch([[float(result[k])] for k in keys], schema=out_schema)
            writer.write_batch(reply)
            wf.flush()
    except (pa.ArrowInvalid, StopIteration, OSError):
        return


def client_pa(kind, family):
    import pyarrow as pa

    schema = pa.schema([(n, pa.float64()) for n in NAMES])
    batch = pa.record_batch([[float(FLAGSHIP_ROW[n])] for n in NAMES], schema=schema)
    s = socket.socket(family, socket.SOCK_STREAM)
    for _ in range(400):
        try:
            s.connect(UDS if family == socket.AF_UNIX else ("127.0.0.1", PORT))
            break
        except OSError:
            time.sleep(0.05)
    if family != socket.AF_UNIX:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    wf, rf = s.makefile("wb"), s.makefile("rb")
    writer = pa.ipc.new_stream(wf, schema)
    writer.write_batch(batch)            # `open_stream` blocks on a schema message alone
    wf.flush()
    reader = pa.ipc.open_stream(rf)
    reader.read_next_batch()

    def once():
        writer.write_batch(batch)
        wf.flush()
        return reader.read_next_batch()

    return once, s.close


def serve_shm(kind):
    from multiprocessing import shared_memory

    handler = None if kind == "echo" else build()
    run = codec(kind, handler)
    shm = shared_memory.SharedMemory(name=SHM, create=True, size=mmap.PAGESIZE * 2)
    buf = shm.buf
    seq = np.ndarray(2, np.int64, buf)                     # [request, response]
    print("ready", flush=True)
    served = 0
    try:
        while True:
            while seq[0] == served:                        # spin: lowest latency, burns a core
                pass
            served = seq[0]
            if served < 0:
                return
            n = struct.unpack_from("<I", buf, 16)[0]
            reply = run(bytes(buf[20:20 + n]))
            buf[mmap.PAGESIZE + 4:mmap.PAGESIZE + 4 + len(reply)] = reply
            struct.pack_into("<I", buf, mmap.PAGESIZE, len(reply))
            seq[1] = served
    finally:
        shm.close()
        shm.unlink()


def serve_http(kind, fast=False):
    import uvicorn
    from decider.serving.servers.starlette import create_app

    handler = build()
    extra = {"loop": "uvloop", "http": "httptools"} if fast else {}
    if fast:
        import httptools
        import uvloop
        print(f"using uvloop {uvloop.__version__} httptools {httptools.__version__}", file=sys.stderr, flush=True)
    print("ready", flush=True)
    uvicorn.run(create_app(handler), host="127.0.0.1", port=PORT, log_level="error", access_log=False, **extra)


def _recv(conn, n):
    out = b""
    while len(out) < n:
        chunk = conn.recv(n - len(out))
        if not chunk:
            return b""
        out += chunk
    return out


# ── clients ───────────────────────────────────────────────────────────────────

def client_raw(kind, family, spin=False):
    body = BODIES[kind]
    msg = struct.pack("<I", len(body)) + body
    s = socket.socket(family, socket.SOCK_STREAM)
    for _ in range(200):
        try:
            s.connect(UDS if family == socket.AF_UNIX else ("127.0.0.1", PORT))
            break
        except OSError:
            time.sleep(0.05)
    if family != socket.AF_UNIX:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if not spin:
        def once():
            s.sendall(msg)
            return _recv(s, struct.unpack("<I", _recv(s, 4))[0])
        return once, s.close

    # Never block, so the reply costs no scheduler wakeup: what shm_* already does.
    s.setblocking(False)

    def spun(n):
        out = b""
        while len(out) < n:
            try:
                out += s.recv(n - len(out))
            except BlockingIOError:
                pass
        return out

    def once():
        s.sendall(msg)
        return spun(struct.unpack("<I", spun(4))[0])

    return once, s.close


def client_http(kind):
    ct, ac = (("application/vnd.apache.arrow.stream",) * 2 if kind.startswith("arrow")
              else ("application/json", "application/json"))
    body = BODIES[kind]
    if kind == "ping":
        req = b"GET /ping HTTP/1.1\r\nHost: b\r\nConnection: keep-alive\r\n\r\n"
    else:
        req = HEAD.format(ct=ct, ac=ac, n=len(body)).encode() + body
    s = socket.socket()
    for _ in range(400):
        try:
            s.connect(("127.0.0.1", PORT))
            break
        except OSError:
            time.sleep(0.05)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    rest = bytearray()

    def once():
        s.sendall(req)
        while b"\r\n\r\n" not in rest:
            rest.extend(s.recv(65536))
        head, _, tail = bytes(rest).partition(b"\r\n\r\n")
        n = int(dict(l.split(b": ", 1) for l in head.split(b"\r\n")[1:]).get(b"content-length", b"0"))
        while len(tail) < n:
            tail += s.recv(65536)
        del rest[:]
        rest.extend(tail[n:])
        return tail[:n]

    return once, s.close


def client_shm(kind):
    from multiprocessing import shared_memory

    body = BODIES[kind]
    for _ in range(400):
        try:
            shm = shared_memory.SharedMemory(name=SHM)
            break
        except FileNotFoundError:
            time.sleep(0.05)
    buf = shm.buf
    seq = np.ndarray(2, np.int64, buf)
    struct.pack_into("<I", buf, 16, len(body))
    buf[20:20 + len(body)] = body
    sent = 0

    def once():
        nonlocal sent
        sent += 1
        seq[0] = sent
        while seq[1] != sent:
            pass
        n = struct.unpack_from("<I", buf, mmap.PAGESIZE)[0]
        return bytes(buf[mmap.PAGESIZE + 4:mmap.PAGESIZE + 4 + n])

    def stop():
        seq[0] = -1
        shm.close()

    return once, stop


# ── driver ────────────────────────────────────────────────────────────────────

ROWS = [
    ("http_json (today, h11+asyncio)", "http", "json"),
    ("http_arrow_ipc", "http", "arrow"),
    ("http_ping (transport only)", "http", "ping"),
    ("http_json, httptools+uvloop", "http+fast", "json"),
    ("http_ping, httptools+uvloop", "http+fast", "ping"),
    ("tcp_json", "tcp", "json"),
    ("tcp_arrow_ipc", "tcp", "arrow"),
    ("tcp_arrow_ipc -> run(df)", "tcp", "arrow_run"),
    ("tcp_arrow_batch (schema cached)", "tcp", "arrow_batch"),
    ("tcp_values (schema once)", "tcp", "values"),
    ("tcp_echo (transport only)", "tcp", "echo"),
    ("uds_json", "uds", "json"),
    ("uds_arrow_ipc", "uds", "arrow"),
    ("uds_arrow_batch (schema cached)", "uds", "arrow_batch"),
    ("uds_values (schema once)", "uds", "values"),
    ("uds_echo (transport only)", "uds", "echo"),
    ("uds_values, spin", "uds+spin", "values"),
    ("uds_echo, spin", "uds+spin", "echo"),
    ("tcp_pyarrow_stream (schema once)", "pa", "values"),
    ("tcp_pyarrow_stream, echo", "pa", "echo"),
    ("uds_pyarrow_stream (schema once)", "pa+uds", "values"),
    ("uds_pyarrow_stream, echo", "pa+uds", "echo"),
    ("shm_values (schema once, spin)", "shm", "values"),
    ("shm_echo (transport only)", "shm", "echo"),
]
FAMILY = {"tcp": socket.AF_INET, "uds": socket.AF_UNIX}


def measure(label, family, kind, calls, reps):
    base, _, flag = family.partition("+")
    # A transport that deadlocks fails its row instead of stalling the table.
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError(label)))
    signal.alarm(180)
    child = subprocess.Popen([sys.executable, __file__, "--serve", base, kind] + ([f"--{flag}"] if flag else []),
                             stdout=subprocess.PIPE, cwd=os.getcwd())
    try:
        while child.stdout.readline().strip() != b"ready":
            if child.poll() is not None:
                raise RuntimeError(f"{label}: server exited {child.returncode}")
        if base == "http":
            once, close = client_http(kind)
        elif base == "shm":
            once, close = client_shm(kind)
        elif base == "pa":
            once, close = client_pa(kind, FAMILY["uds" if flag == "uds" else "tcp"])
        else:
            once, close = client_raw(kind, FAMILY[base], spin=flag == "spin")
        best = None
        for _ in range(reps):
            for _ in range(WARM):
                once()
            samples = np.empty(calls)
            perf = time.perf_counter
            for k in range(calls):
                t = perf()
                once()
                samples[k] = perf() - t
            p = _pct(samples)
            best = p if best is None else (min(best[0], p[0]), min(best[1], p[1]))
        close()
        return (*best, 0 if base == "pa" else len(BODIES[kind]))
    finally:
        signal.alarm(0)
        try:
            child.wait(3)
        except subprocess.TimeoutExpired:
            child.terminate()
            child.wait(10)


def main(calls, reps, only=""):
    print(f"{'transport':<34}{'p50 µs':>9}{'p99 µs':>9}{'body B':>9}   (min of {reps} reps)")
    for label, family, kind in ROWS:
        if only and only not in label:
            continue
        try:
            p50, p99, nbytes = measure(label, family, kind, calls, reps)
            print(f"{label:<34}{p50:>9.1f}{p99:>9.1f}{nbytes:>9}", flush=True)
        except Exception as e:
            print(f"{label:<34}  skipped: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        family, kind = sys.argv[sys.argv.index("--serve") + 1:][:2]
        if family == "http":
            serve_http(kind, fast="--fast" in sys.argv)
        elif family == "pa":
            serve_pa(kind, FAMILY["uds" if "--uds" in sys.argv else "tcp"])
        elif family == "shm":
            serve_shm(kind)
        else:
            serve_raw(kind, FAMILY[family])
    else:
        n = int(sys.argv[sys.argv.index("--calls") + 1]) if "--calls" in sys.argv else CALLS
        reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""
        main(n, reps, only)
