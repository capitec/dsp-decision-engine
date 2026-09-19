"""
Does the tail get WORSE under heavier external CPU contention, or does it
stay bounded? Directly answers the brief's "is it fixable/worth fixing --
19.2% max that stays 19.2% vs. one that becomes 120% under a different
regime" question, for the contention axis (as opposed to alloc_sweep.py's
allocation-size axis).

Launches N_STRESS bounded (timeout-capped, self-terminating) busy-spin
shell loops to create real, brief CPU contention on this shared box, then
reruns the SAME instrumented score() measurement (imports tail_cause.py's
helpers directly, config D = "unpinned, contended") concurrently with that
load. Stressors are killed in a `finally` block and hard-capped at 15s by
`timeout` regardless, to bound impact on this shared, non-idle machine.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("tail_cause", str(HERE / "tail_cause.py"))
tc = importlib.util.module_from_spec(_spec)
sys.modules["tail_cause"] = tc
_spec.loader.exec_module(tc)  # runs n1_overhead import + kernel compile; does NOT run main()

N_STRESS = 14  # half of nproc=28 -- deliberately not saturating the whole box
N_CALLS = 8000
STRESS_TIMEOUT_S = 15  # hard self-terminating cap


def call_score():
    tc.n1.score(tc.n1.REQUEST_DICT, tc.n1.PARAMS_RAW)


def main():
    tc.log(f"launching {N_STRESS} bounded busy-spin stressors (self-terminate after {STRESS_TIMEOUT_S}s)")
    procs = [
        subprocess.Popen(["timeout", str(STRESS_TIMEOUT_S), "bash", "-c", "while :; do :; done"])
        for _ in range(N_STRESS)
    ]
    try:
        time.sleep(1.0)  # let contention ramp up before measuring
        result = tc.run_config(call_score, N_CALLS, 100, "D_contended_unpinned", pin_core=None)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        tc.log("stressors terminated")

    summary_path = HERE / "results_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["D_contended_unpinned"] = result
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    tc.log("DONE")


if __name__ == "__main__":
    main()
