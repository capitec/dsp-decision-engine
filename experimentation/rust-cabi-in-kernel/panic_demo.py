"""Panic safety across the C-ABI boundary — item 5.

Each mode of `panic_demo_worker.py` runs in its OWN subprocess: the
`_unprotected` modes are EXPECTED to kill the process (that is what they
demonstrate), so they must not be able to take this measurement run down
with them.

rustc 1.95.0 here is well past 1.71, the release that made "an `extern
"C"` fn panicking without unwinding across the boundary" a DEFINED abort
rather than undefined behaviour (RFC 2945 / `extern "C-unwind"`). This
script confirms empirically what that guarantee, and this crate's explicit
`catch_unwind` wrapping, actually produce — for both a synthetic
`panic!()` and a realistic trigger (a corrupted tree walked by the exact
function `bench_perf.py` times).

Run:
    <repo>/.venv/bin/python experimentation/rust-cabi-in-kernel/panic_demo.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.jsonl"


def append_result(record: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def run_mode(mode: str) -> dict:
    print(f"\n=== mode: {mode} ===")
    proc = subprocess.run(
        [sys.executable, str(HERE / "panic_demo_worker.py"), mode],
        capture_output=True, text=True, timeout=30,
    )
    print(f"returncode: {proc.returncode}")
    if proc.returncode < 0:
        import signal
        signum = -proc.returncode
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"signal {signum}"
        print(f"  process was KILLED by {signame}")
    print("stdout:")
    for line in proc.stdout.splitlines():
        print(f"  {line}")
    if proc.stderr.strip():
        print("stderr (last 15 lines):")
        for line in proc.stderr.splitlines()[-15:]:
            print(f"  {line}")
    return {
        "mode": mode,
        "returncode": proc.returncode,
        "killed_by_signal": proc.returncode < 0,
        "signal": (-proc.returncode) if proc.returncode < 0 else None,
        "stdout_tail": proc.stdout.splitlines()[-5:],
        "stderr_tail": proc.stderr.splitlines()[-5:],
    }


def main() -> None:
    modes = ["trivial_unprotected", "trivial_protected", "tree_unprotected", "tree_protected"]
    results = {m: run_mode(m) for m in modes}

    print("\n\n=== SUMMARY ===")
    for m in modes:
        r = results[m]
        status = f"KILLED (signal {r['signal']})" if r["killed_by_signal"] else f"exited {r['returncode']}"
        print(f"{m:24s}: {status}")

    append_result({"event": "panic_safety_demo", "results": results})
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
