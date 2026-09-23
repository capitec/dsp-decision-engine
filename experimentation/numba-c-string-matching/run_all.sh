#!/usr/bin/env bash
# Reproduce every number in RESULTS.md, in order. ~8 minutes single-threaded.
# Measurements append to results.jsonl as they happen.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
./c/build.sh
$PY smoke_test.py
$PY bench_regex.py
$PY ablate_refcount.py
$PY bench_typed_walk.py
NUMBA_BOUNDSCHECK=1 $PY bench_typed_walk.py
$PY bench_selectivity.py
$PY safety_check.py
$PY cache_check.py
