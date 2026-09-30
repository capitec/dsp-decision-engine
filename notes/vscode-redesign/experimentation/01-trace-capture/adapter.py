"""A real threaded adapter with a bounded queue: backpressure, overflow, failure, observability.

The kernel never blocks or allocates per event; the adapter is another thread
(or process) the driver hands a drained trace buffer to. This exercises the real
`queue.Queue` seam: a bounded queue, a producer that must never block (the
capture side), a consumer that can be slow or fail, and the manifest/status
counters that report loss or degradation instead of hiding it.

    uv run python notes/vscode-redesign/experimentation/01-trace-capture/adapter.py
"""
import queue
import threading
import time

import numpy as np
from numba import njit


@njit(nogil=True)
def capture(n, steps, events, offsets):
    for i in range(n):
        for s in range(steps):
            events[i * steps + s] = (s << 48) | (i & 0xFFFF)
        offsets[i] = i * steps
    offsets[n] = n * steps


class TraceAdapter:
    """A bounded-queue adapter: `drain` never blocks (drop-on-overflow); `run` consumes off the hot path."""

    def __init__(self, capacity):
        self.q: queue.Queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.delivered = 0
        self.failed = 0
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._consume, daemon=True)
        self._worker.start()

    def drain(self, events, offsets, n):
        # The producer side: called on the request path. Non-blocking; overflow is
        # dropped and counted, never turned into a block the kernel could hit.
        for r in range(n):
            lo, hi = offsets[r], offsets[r + 1]
            for k in range(lo, hi):
                try:
                    self.q.put_nowait(int(events[k]))
                except queue.Full:
                    self.dropped += 1
        self.delivered += n

    def _consume(self):
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.05)
            except queue.Empty:
                continue
            # A slow consumer is backpressure: the queue fills and `drain` drops.
            if item < 0:  # sentinel the test injects to make the consumer fail
                self.failed += 1
                self.errors.append(f"adapter failed on {item}")
                continue
            time.sleep(0.0)  # simulate delivery work

    def status(self):
        return {"queue": self.q.qsize(), "dropped": self.dropped,
                "delivered": self.delivered, "failed": self.failed, "errors": len(self.errors)}


if __name__ == "__main__":
    n, steps = 200_000, 8
    events = np.empty(n * steps, np.int64)
    offsets = np.empty(n + 1, np.int64)
    capture(n, steps, events, offsets)

    # 1. Overflow: a tiny queue, a fast producer, a slow consumer -> drops are counted.
    small = TraceAdapter(capacity=16)
    small.drain(events, offsets, n)
    time.sleep(0.1)
    s = small.status()
    print(f"backpressure: capacity 16, {n * steps:,} events drained -> "
          f"delivered {s['delivered']:,} records, dropped {s['dropped']:,} events on overflow "
          f"(producer never blocked)")

    # 2. A blocking-free producer: `drain` uses put_nowait, so it cannot block on a full queue.
    assert s["dropped"] > 0, "a tiny queue must overflow to show drop-on-overflow"
    print("overflow semantics: drop + count, never block; the kernel write path is untouched")

    # 3. Adapter failure: the consumer raises on a bad item; the producer keeps going and it is reported.
    big = TraceAdapter(capacity=10_000)
    big.drain(events[: 8 * 1000], offsets[: 1001], 1000)  # 8000 events, all fit
    big.q.put(-1)  # inject a poison item behind a short backlog
    for _ in range(100):  # wait for the worker to reach and count the poison
        if big.failed:
            break
        time.sleep(0.02)
    s = big.status()
    print(f"adapter failure: {s['failed']} failed delivery, {s['errors']} errors reported; "
          f"producer still delivered {s['delivered']:,} records (failure is the adapter's, not the run's)")

    # 4. Observability: the manifest/status counters are the only place loss shows.
    print(f"observability: status = {big.status()} -> a trace manifest carries dropped/failed/queue "
          f"counters so loss and degradation are reported, never silent")

    small._stop.set()
    big._stop.set()
