#!/usr/bin/env bash
# Full reproduction. From the repo root:
#   bash experimentation/rust-string-matching/run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../.venv/bin/python"
(cd "$HERE/rust_string_match" && cargo build --release 2>&1 | tail -1)
rm -f "$HERE/results.jsonl"
"$PY" "$HERE/smoke_test.py"
echo "=== cache check: cold (fresh __pycache__), then warm in a SEPARATE process ==="
rm -rf "$HERE/__pycache__"
NUMBA_DEBUG_CACHE=1 "$PY" "$HERE/cache_check.py" cold 2>&1 | tee "$HERE/cache_cold.log" | grep -E "^\[cache\]|^\[cold\]" | sed 's/^/  /'
NUMBA_DEBUG_CACHE=1 "$PY" "$HERE/cache_check.py" warm 2>&1 | tee "$HERE/cache_warm.log" | grep -E "^\[cache\]|^\[warm\]" | sed 's/^/  /'
echo "=== bench ==="
"$PY" "$HERE/bench.py" low  | tee "$HERE/bench_low.log"
"$PY" "$HERE/bench.py" high | tee "$HERE/bench_high.log"
echo "=== panic demo ==="
"$PY" "$HERE/panic_demo.py" | tee "$HERE/panic_demo.log"
