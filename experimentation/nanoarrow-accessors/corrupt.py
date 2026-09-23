"""The corruption matrix: every (case, approach) in its own subprocess.
Classification:
  refused      -- Python exception before any row was read (import/validate/format check)
  error_leaf   -- ran, wrote ERR_LEAF (-1) for the corrupted row, other rows right
  correct      -- ran and matched the clean reference (only meaningful for the legit cases)
  wrong_answer -- ran to completion and returned something else (read out of bounds silently)
  crash        -- the process died (signal), e.g. SIGSEGV
"""
from __future__ import annotations

import os
import subprocess
import sys

from results_io import record

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = ["clean", "sliced_offset", "bad_buffer_index", "bad_offset", "bad_length", "bad_length_scan",
         "null_validity_nonzero_null_count", "n_buffers_truncated", "lying_sizes_buffer"]
APPROACHES = ["A", "B", "B-checked", "C-full", "C-trusting"]
CLEAN = [1, 1, 0, 0, 1, 1]           # contains "dog" over corrupt_child.STRINGS
ERR_AT_ROW1 = [1, -1, 0, 0, 0, 0]    # ERR_LEAF stops the chunk at the bad row (rows after are unwritten -> 0 by np.empty? no: undefined)


LEGIT = ("clean", "sliced_offset")


def classify(case, rc, stdout, stderr):
    res = [ln for ln in stdout.splitlines() if ln.startswith("RESULT")]
    if rc < 0 or rc >= 128:
        return "crash", f"signal {-rc if rc < 0 else rc - 128}"
    if rc != 0:
        last = [ln for ln in stderr.strip().splitlines() if ln.strip()][-1:] or ["?"]
        return "refused", last[0][:160]
    got = eval(res[0][len("RESULT "):]) if res else None
    clean = [0] * len(CLEAN) if case == "bad_length_scan" else CLEAN
    if got is not None and len(got) == len(CLEAN) and -1 in got:
        return "error_leaf", str(got)
    if got == clean:
        if case in LEGIT:
            return "correct", str(got)
        if case == "null_validity_nonzero_null_count":
            return "answered (null_count ignored)", str(got)
        if case == "bad_length":
            return "answered by luck", f"{got}: match found before the 1 GB overrun was walked"
        return "wrong_answer", f"returned the clean answer {got} on corrupt data (read out of bounds silently)"
    return "wrong_answer", str(got)


def main():
    env = dict(os.environ, PYTHONHASHSEED="0")
    rows = []
    for case in CASES:
        for ap in APPROACHES:
            p = subprocess.run([sys.executable, os.path.join(HERE, "corrupt_child.py"), case, ap],
                               capture_output=True, text=True, cwd=HERE, env=env, timeout=120)
            kind, detail = classify(case, p.returncode, p.stdout, p.stderr)
            rec = record(probe="corrupt", case=case, approach=ap, returncode=p.returncode, outcome=kind, detail=detail)
            rows.append(rec)
            print(f"{case:<34} {ap:<11} rc={p.returncode:>4}  {kind:<13} {detail}", flush=True)


if __name__ == "__main__":
    main()
