"""Time ONE `build_driver` call against a PERSISTENT build_dir, for either
`pipeline_module` (the C-ABI step) or `control_module` (an otherwise
identical pipeline, tree walk done in plain numba, no ctypes, no global
arrays referenced from inside the njit'd step). Run twice per module, in
two separate fresh processes, against the SAME `--build-dir`, to see
whether numba's on-disk cache (decider2.compile.cache's whole point, doc
05 §4.1/§4.2) survives the second process — the control isolates whether
`pipeline_module`'s cold-every-time cost (see RESULTS.md) is really the
ctypes reference, not just this script deleting its own cache between
runs (a tempdir would do that regardless of ctypes; a persistent
`--build-dir` would not).

Usage:
    python time_one_build.py <pipeline_module|control_module> <build_dir>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from decider2.compile.driver import build_driver  # noqa: E402
from decider2.graph.step import make_step  # noqa: E402


def main() -> None:
    module_name, build_dir = sys.argv[1], sys.argv[2]
    mod = __import__(module_name)

    steps = [make_step(mod.scale_features), make_step(mod.tree_decision), make_step(mod.apply_bonus)]
    group_ids = [0, 0, 0]

    t0 = time.perf_counter()
    driver = build_driver(steps, group_ids, build_dir=build_dir, terminal_names=frozenset({"apply_bonus"}))
    dt = time.perf_counter() - t0

    seg = driver.segments[0]
    print(f"module={module_name} build_dir={build_dir} segment_kind={seg.kind} build_s={dt:.4f}")


if __name__ == "__main__":
    main()
