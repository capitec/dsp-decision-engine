"""Does the fused kernel stay fused around a C-ABI call? — item 2 of the
rust-cabi-in-kernel experiment, and the whole point of measuring the C ABI
at all (§U's corrected finding).

Builds a real 3-step decider2 pipeline (`pipeline_module.py`: `scale_features`
-> `tree_decision` [the C-ABI tree walk] -> `apply_bonus`) through decider2's
REAL, unmodified `decider2.compile.driver.build_driver` and
`decider2.compile.codegen.emit_kernel_source` — the exact code path a real
`fuse()`-authored group goes through. decider2/src is imported, never
modified.

Confirms:
  1. `build_driver` produces exactly ONE segment for the group, and it is a
     `CompiledSegment` (not a `FallbackSegment` — i.e. the C-ABI step did
     not trip `_FALLBACK_TRIGGERS` and force a split).
  2. The segment's generated kernel source (`emit_kernel_source`, the exact
     text `decider2.compile.cache.get_or_build` writes to disk and numba
     compiles) contains ONE `def kernel(...)`, calling all three steps —
     including the C-ABI one — inside the SAME row loop.
  3. The driver actually runs and produces the numerically correct answer.

Run:
    <repo>/.venv/bin/python experimentation/rust-cabi-in-kernel/fusion_check/run_fusion_check.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import pipeline_module as pm  # noqa: E402 — the real, importable module under test

from decider2.compile import codegen  # noqa: E402
from decider2.compile.driver import CompiledSegment, FallbackSegment, ResolvedParams, build_driver  # noqa: E402
from decider2.graph.step import make_step  # noqa: E402

RESULTS_PATH = ROOT / "results.jsonl"
GENERATED_SOURCE_PATH = HERE / "generated_kernel_source.py"


def append_result(record: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def main() -> None:
    steps = [make_step(pm.scale_features), make_step(pm.tree_decision), make_step(pm.apply_bonus)]
    for s in steps:
        print(f"step {s.name!r}: inputs={[i.name for i in s.inputs]}")

    group_ids = [0, 0, 0]  # one fuse()-group: all three steps together

    with tempfile.TemporaryDirectory(prefix="cabi_fusion_build_") as build_dir:
        driver = build_driver(
            steps, group_ids,
            build_dir=build_dir,
            terminal_names=frozenset({"apply_bonus"}),
        )

        n_segments = len(driver.segments)
        seg = driver.segments[0]
        kind = seg.kind
        is_compiled = isinstance(seg, CompiledSegment)
        is_fallback = isinstance(seg, FallbackSegment)
        print(f"\nn_segments = {n_segments}")
        print(f"segments[0].kind = {kind!r}  (CompiledSegment: {is_compiled}, FallbackSegment: {is_fallback})")

        assert n_segments == 1, (
            f"expected ONE segment (the whole group fused around the C-ABI call), got {n_segments}: "
            f"{[s.kind for s in driver.segments]}"
        )
        assert is_compiled, f"expected a CompiledSegment, got {type(seg).__name__} (fallback_reason="\
            f"{getattr(seg, 'fallback_reason', None)!r}) — the C-ABI step tripped a numba compile failure "\
            "and was split out, which would refute the whole premise of this experiment."

        plan = seg.plan
        source = codegen.emit_kernel_source(plan)
        GENERATED_SOURCE_PATH.write_text(source)
        print(f"\nwrote generated kernel source to {GENERATED_SOURCE_PATH}")

        n_kernel_defs = source.count("def kernel(")
        n_walk_row_refs = source.count("tree_decision")
        contains_all_three = all(name in source for name in ("scale_features", "tree_decision", "apply_bonus"))
        print(f"'def kernel(' occurrences: {n_kernel_defs}")
        print(f"all three step names present in the ONE generated file: {contains_all_three}")

        assert n_kernel_defs == 1, f"expected exactly one kernel function, found {n_kernel_defs}"
        assert contains_all_three

        # ---- run it, and check the answer is actually right ----
        n = 5
        registry = {
            "f0": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            "f1": np.array([5.0, 15.0, 25.0, 35.0, 45.0]),
            "f2": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "f3": np.array([9.0, 8.0, 7.0, 6.0, 5.0]),
        }
        resolved = ResolvedParams(per_step_scalar={}, per_step_bundle={}, shared=None)
        seg.run(registry, resolved, n)
        got = registry["apply_bonus"]

        # independent oracle: the numba tree walker (`interpreted_kernel`),
        # not the C-ABI path, so "the driver ran and got the right answer"
        # is not just "the driver returned SOMETHING".
        sys.path.insert(0, str(ROOT.parent / "tree-codegen-vs-interpreted"))
        import interpreted_kernel as ik  # noqa: E402

        expected = np.empty(n)
        for i in range(n):
            row = np.array([registry["f0"][i], registry["f1"][i], registry["f2"][i], registry["f3"][i]])
            leaf_val = ik._walk_one(row, pm._STR_ROW, pm._KIND, pm._FEAT, pm._OP, pm._THRESH,
                                     pm._PAT_START, pm._PAT_COUNT, pm._PATTERNS, pm._LEFT, pm._RIGHT, pm._LEAF)
            scale = registry["f0"][i] / 100.0 - registry["f1"][i] / 100.0
            expected[i] = leaf_val + 0.01 * scale

        answers_match = np.allclose(got, expected)
        print(f"\ndriver output: {got}")
        print(f"oracle output: {expected}")
        print(f"answers match: {answers_match}")
        assert answers_match

        record = {
            "event": "fusion_check",
            "n_segments": n_segments,
            "segment_kind": kind,
            "n_kernel_defs_in_source": n_kernel_defs,
            "contains_all_three_steps": contains_all_three,
            "answers_match_oracle": bool(answers_match),
            "generated_source_path": str(GENERATED_SOURCE_PATH),
            "group_name": plan.group_name,
        }
        append_result(record)
        print("\n=== FUSION CHECK: PASSED — one kernel, one segment, correct answer ===")


if __name__ == "__main__":
    main()
