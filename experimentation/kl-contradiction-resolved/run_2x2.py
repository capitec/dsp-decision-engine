"""K/L contradiction, resolved by a clean 2x2 -- real subprocesses (matching
K's methodology and doc 08 SS4's actual deployment shape: compile happens in
one process, is loaded in another), module-name import (doc 05 SS4.1 J2 fix,
not spec_from_file_location), L's dedup-safe driver shape (one occurrence of
the threshold literal, so the edit CAN leave co_code unchanged).

Four arms: both caches live / only .pyc cleared / only numba .nbi+.nbc
cleared / both cleared. Each arm starts from a completely fresh directory
(no cross-arm contamination) and re-establishes the baseline compile itself.

Writes results incrementally to results.jsonl, flushed after each line
(mandatory rule 3). Estimated memory: a handful of KB (one scalar njit
function, no arrays) -- run in the foreground, no tmux/systemd-run needed.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen import gen_source  # noqa: E402

PYTHON = sys.executable
CHILD = os.path.join(HERE, "child.py")

OLD_THRESHOLD = "0.0405"
NEW_THRESHOLD = "0.5950"
assert len(OLD_THRESHOLD) == len(NEW_THRESHOLD)

RESULTS_PATH = os.path.join(HERE, "results.jsonl")


def run_child(workdir):
    proc = subprocess.run([PYTHON, CHILD, workdir], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return {"ok": False, "stderr": proc.stderr[-3000:], "stdout": proc.stdout[-1000:]}
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


def emit(row):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row, indent=2))


def clear_pyc(workdir):
    pc = os.path.join(workdir, "__pycache__")
    if os.path.isdir(pc):
        for fn in os.listdir(pc):
            if fn.endswith(".pyc"):
                os.remove(os.path.join(pc, fn))


def clear_numba_cache(workdir):
    pc = os.path.join(workdir, "__pycache__")
    if os.path.isdir(pc):
        for fn in os.listdir(pc):
            if fn.endswith(".nbi") or fn.endswith(".nbc"):
                os.remove(os.path.join(pc, fn))


def run_arm(name, clear_pyc_flag, clear_numba_flag):
    workdir = os.path.join(HERE, f"_arm_{name}")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    drv_path = os.path.join(workdir, "drv.py")

    # --- baseline (generation A) ---
    with open(drv_path, "w") as f:
        f.write(gen_source(OLD_THRESHOLD))
    before = run_child(workdir)
    st0 = os.stat(drv_path)

    # --- edit in place, same byte length, preserve mtime ---
    with open(drv_path) as f:
        body = f.read()
    assert body.count(OLD_THRESHOLD) == 1, "threshold not unique in source -- trick invalid"
    body2 = body.replace(OLD_THRESHOLD, NEW_THRESHOLD, 1)
    with open(drv_path, "w") as f:
        f.write(body2)
    os.utime(drv_path, (st0.st_atime, st0.st_mtime))
    st1 = os.stat(drv_path)
    assert st1.st_size == st0.st_size, "edit changed file size -- trick invalid"
    assert st1.st_mtime == st0.st_mtime, "mtime not preserved"

    # --- arm-specific cache clearing ---
    if clear_pyc_flag:
        clear_pyc(workdir)
    if clear_numba_flag:
        clear_numba_cache(workdir)

    after = run_child(workdir)

    # --- ground truth: brand new directory, edited source only, never
    #     shared a cache entry with anything above ---
    truth_dir = os.path.join(HERE, f"_arm_{name}_truth")
    shutil.rmtree(truth_dir, ignore_errors=True)
    os.makedirs(truth_dir)
    with open(os.path.join(truth_dir, "drv.py"), "w") as f:
        f.write(gen_source(NEW_THRESHOLD))
    truth = run_child(truth_dir)

    row = {
        "arm": name,
        "pyc_cleared": clear_pyc_flag,
        "numba_cache_cleared": clear_numba_flag,
        "value_before_edit": before.get("value"),
        "value_served_after_edit": after.get("value"),
        "value_ground_truth": truth.get("value"),
        "co_consts_before": before.get("co_consts"),
        "co_consts_after_served": after.get("co_consts"),
        "pycache_listing_after": after.get("pycache_after"),
        "served_stale": (
            after.get("value") == before.get("value")
            and after.get("value") != truth.get("value")
        ),
        "before_ok": before.get("ok"),
        "after_ok": after.get("ok"),
        "truth_ok": truth.get("ok"),
    }
    if not before.get("ok"):
        row["before_error"] = before.get("stderr")
    if not after.get("ok"):
        row["after_error"] = after.get("stderr")
    return row


def main():
    t0 = time.time()
    arms = [
        ("both_live", False, False),
        ("pyc_only_cleared", True, False),
        ("numba_only_cleared", False, True),
        ("both_cleared", True, True),
    ]
    for name, cp, cn in arms:
        print(f"=== arm: {name} (pyc_cleared={cp}, numba_cache_cleared={cn}) ===")
        row = run_arm(name, cp, cn)
        emit(row)
    print(f"\ntotal wall clock: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
