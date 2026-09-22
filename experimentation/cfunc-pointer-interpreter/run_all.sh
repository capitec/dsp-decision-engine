#!/usr/bin/env bash
# Runs the whole cfunc-pointer-interpreter experiment in order, writing
# every measurement to results.jsonl incrementally. Machine discipline:
# 2GB address-space cap per process (the shared-box rule this experiment
# was built under), `free -h` printed first.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

echo "=== free -h ==="
free -h
echo

echo "=== item 1: call overhead baseline ==="
ulimit -v 2000000 && "$PY" bench_call_overhead.py

echo
echo "=== item 1b: signature/JIT-vs-AOT sensitivity matrix ==="
ulimit -v 2000000 && "$PY" bench_signature_matrix.py

echo
echo "=== item 2: standalone shape coverage (straight-line, Branch, Loop w/ early exit) ==="
ulimit -v 2000000 && "$PY" test_shapes.py

echo
echo "=== item 5 (cache): cold, against a fresh __pycache__ ==="
rm -rf __pycache__
ulimit -v 2000000 && "$PY" cache_check.py cold

echo
echo "=== item 5 (cache): warm, fresh process, same __pycache__ ==="
ulimit -v 2000000 && "$PY" cache_check.py warm

echo
echo "=== item 4: per-shape build cost, codegen vs interpreter ==="
ulimit -v 2000000 && "$PY" build_cost.py

echo
echo "=== items 3+5+6: identical work vs decider2's real codegen, 5 independent process reruns ==="
for i in 1 2 3 4 5; do
  echo "--- run $i ---"
  ulimit -v 2000000 && "$PY" compare_vs_codegen.py
done

echo
echo "=== done. results.jsonl has $(wc -l < results.jsonl) records. ==="
