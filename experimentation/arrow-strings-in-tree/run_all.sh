#!/usr/bin/env bash
# Reproduce every number in RESULTS.md (about 10 minutes on a quiet box).
# Needs >= 8 GB free; polars 1.41.2, numba 0.67, numpy 2.4 -- no pyarrow.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
rm -f results.jsonl
$PY -m pytest test_kernel.py -q            # correctness: 38 tests
$PY probe_zero_copy.py                     # Q1: addresses + O(1) scaling
$PY ablate.py                              # where the first kernel's time went
$PY ablate2.py                             # helper-argument refcount cost
$PY bench.py                               # Q3: 1M rows, low + high cardinality
$PY bench.py                               # second run, for the noise band
$PY cache_check.py                         # Q4: two processes, saved/loaded counts
echo "done: results.jsonl"
