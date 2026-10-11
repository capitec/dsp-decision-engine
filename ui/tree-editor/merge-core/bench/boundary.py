"""Cost of crossing into the Rust core as JSON text against pythonize, on the bench documents (needs the direct wheel).

    python boundary.py ../merge-core-bench/out/bench_inputs.json [-n 30]
"""

import json
import statistics
import sys
import time

from merge_core import _native

inputs = json.load(open(sys.argv[1]))
n = int(sys.argv[sys.argv.index("-n") + 1]) if "-n" in sys.argv else 30


def med(f):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        times.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(times), 3)


for name, d in inputs.items():
    base, ops = d["base"], json.loads(_native.call("diff", json.dumps([d["base"], d["batch50"]])))
    args = [base, ops]
    text = json.dumps(args)
    empty = json.dumps([base, []])
    out = _native.call("apply", text)
    print(name, {
        "native parse + clone + serialize only (no ops)": med(lambda: _native.call("apply", empty)),
        "json.dumps of args": med(lambda: json.dumps(args)),
        "native apply on text (parse + apply + serialize)": med(lambda: _native.call("apply", text)),
        "json.loads of result": med(lambda: json.loads(out)),
        "text route total": med(lambda: json.loads(_native.call("apply", json.dumps(args)))),
        "pythonize route total": med(lambda: _native.call_direct("apply", args)),
    })
