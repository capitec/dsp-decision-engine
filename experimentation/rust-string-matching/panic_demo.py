"""Panic safety, demonstrated not asserted. Each mode runs in its OWN
subprocess because the unprotected modes are expected to kill it.

  oob_protected     match_one with idx = 10**9 from INSIDE njit -> -1, then
                    a good call in the same process still answers.
  oob_unprotected   same hostile idx through match_one_unprotected -> abort.
  short_protected   values_len deliberately shorter than the offsets claim
                    (a lying length) -> slice panic, caught, -1.
  short_unprotected same, unprotected -> abort.
  synthetic_*       the plain panic!() pair, through the pointer table.
"""
from __future__ import annotations
import json, signal, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ["oob_protected", "oob_unprotected", "short_protected", "short_unprotected",
         "synthetic_protected", "synthetic_unprotected"]


def run(mode):
    p = subprocess.run([sys.executable, str(HERE / "panic_demo_worker.py"), mode],
                       capture_output=True, text=True, timeout=120)
    sig = signal.Signals(-p.returncode).name if p.returncode < 0 else None
    print(f"\n=== {mode}: returncode={p.returncode}" + (f" KILLED by {sig}" if sig else ""))
    for l in p.stdout.splitlines(): print("  " + l)
    tail = [l for l in p.stderr.splitlines() if l.strip()][-3:]
    for l in tail: print("  stderr| " + l)
    return {"mode": mode, "returncode": p.returncode, "signal": sig,
            "stdout": p.stdout.splitlines(), "stderr_tail": tail}


if __name__ == "__main__":
    res = [run(m) for m in MODES]
    print("\n=== SUMMARY ===")
    for r in res:
        print(f"{r['mode']:22s} {('KILLED ' + r['signal']) if r['signal'] else 'exited ' + str(r['returncode'])}")
    with open(HERE / "results.jsonl", "a") as f:
        f.write(json.dumps({"experiment": "rust-string-matching", "item": "panic_demo", "results": res}) + "\n")
