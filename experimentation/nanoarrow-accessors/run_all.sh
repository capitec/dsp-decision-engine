#!/usr/bin/env bash
# Reproduce every number in RESULTS.md (about 6 minutes on a quiet box).
# Needs gcc, >= 8 GB free; polars 1.41.2, numba 0.67, numpy 2.4 -- no pyarrow.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
./c/build.sh                                 # shim + vendored nanoarrow -> c/libnashim.so
rm -f results.jsonl
$PY -m pytest test_all.py -q                 # correctness gate: 40 tests, all approaches
$PY bench.py 1 10 100 1000 1000000           # per-call cost by batch size, staged
$PY validate_cost.py                         # ArrowArrayViewValidate vs length
$PY ablate_import.py                         # where the batch-1 import cost goes
$PY bench_lean.py 1; $PY bench_lean.py 1000  # equal-effort floors (pooled structs, hoisted tables)
$PY corrupt.py                               # corruption matrix (one subprocess each)
$PY spec_change.py                           # 'u' / 'U' / Null / unknown formats
$PY cache_check.py                           # numba disk cache: cold / warm / warm2
./build_cost.sh                              # cold build time + object size
$PY count_layout_lines.py                    # auditable line count
$PY bench.py 1 1000000                       # second pass, for the noise band
echo "done: results.jsonl"
