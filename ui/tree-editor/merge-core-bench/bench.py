"""Time a Python adapter against the reference one, interleaved in one process, on out/bench_inputs.json.

    python bench.py adapters/<approach>.py [--ref adapters/pure_py.py] [-n 30] [--json out.json]

Per document (generated-521 and the largest other real tree): load, export, roundtrip of a one-rule edit, roundtrip
of a 50-edit batch, offline merge of 50 edits against 50 others, online of one edit against one concurrent edit.
Prints and writes median and p95 in ms per adapter.
"""

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def _load(path):
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tasks(m, d):
    state = m.load(d["base"])
    return {
        "load": lambda: m.load(d["base"]),
        "export": lambda: m.export(state),
        "roundtrip_one_edit": lambda: m.roundtrip(d["base"], d["one_edit"]),
        "roundtrip_50_edits": lambda: m.roundtrip(d["base"], d["batch50"]),
        "offline_50_vs_50": lambda: m.offline(d["base"], d["mine50"], d["theirs50"]),
        "online_1_vs_1": lambda: m.online(d["base"], d["first"], d["second"]),
    }


def summary(times):
    ordered = sorted(times)
    return {"median_ms": round(statistics.median(ordered) * 1e3, 3),
            "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1] * 1e3, 3), "n": len(ordered)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter")
    parser.add_argument("--ref", default=str(HERE / "adapters/pure_py.py"))
    parser.add_argument("-n", type=int, default=30)
    parser.add_argument("--json")
    args = parser.parse_args()
    inputs = json.loads((HERE / "out/bench_inputs.json").read_text())
    names = list(dict.fromkeys([args.adapter, args.ref]))  # the reference alone when it is the adapter
    adapters = {name: _load(name) for name in names}
    results = {}
    for doc_id, d in inputs.items():
        runs = {name: tasks(m, d) for name, m in adapters.items()}
        times = {name: {task: [] for task in runs[name]} for name in names}
        for rep in range(args.n + 2):  # the first two are warm-up
            for task in runs[names[0]]:
                # Alternate which adapter goes first, so neither always runs on a warmer cache.
                for name in names if rep % 2 else names[::-1]:
                    if times[name][task] is None:
                        continue
                    started = time.perf_counter()
                    try:
                        runs[name][task]()
                    except NotImplementedError:
                        times[name][task] = None
                        continue
                    if rep >= 2:
                        times[name][task].append(time.perf_counter() - started)
        for name in names:
            results.setdefault(Path(name).stem, {})[doc_id] = {
                t: summary(v) if v else "not implemented" for t, v in times[name].items()}
    for adapter, docs in results.items():
        for doc_id, tasks_ in docs.items():
            for task, s in tasks_.items():
                line = f"median {s['median_ms']:9.3f} ms  p95 {s['p95_ms']:9.3f} ms" if isinstance(s, dict) else s
                print(f"{adapter:14} {doc_id:28} {task:20} {line}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
