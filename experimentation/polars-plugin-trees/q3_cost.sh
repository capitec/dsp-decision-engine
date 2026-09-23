#!/bin/bash
# Q3: what it costs to own. Run LAST (cargo clean forces a full rebuild).
set -u
cd "$(dirname "$0")/decider_trees"
PY=/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python
OUT=../q3_cost.log; : > $OUT
echo "== cargo tree" | tee -a $OUT
N_ALL=$(cargo tree --edges normal --prefix none | sort -u | wc -l)
N_POLARS=$(cargo tree --edges normal --prefix none | sort -u | grep -c '^polars')
N_NOCAT=$(cargo tree --edges normal --prefix none --no-default-features --features 'pyo3-polars/derive' 2>/dev/null | sort -u | wc -l)
echo "unique crates (with dtype-categorical): $N_ALL ; of which polars-*: $N_POLARS" | tee -a $OUT
cargo tree --edges normal --prefix none | sort -u | awk '{print $1}' | sort | uniq -c | sort -rn | awk '$1>1' | head -5 | tee -a $OUT
echo "== pins" | tee -a $OUT
grep -A1 '^name = "polars"$\|^name = "pyo3"$\|^name = "pyo3-polars"$\|^name = "polars-ffi"$' Cargo.lock | tee -a $OUT
echo "== incremental rebuild (touch one line)" | tee -a $OUT
sed -i 's|/// Zero-work expression: the floor for "one plugin call".|/// Zero-work expression: the floor for "one plugin call" (touched).|' src/lib.rs
/usr/bin/time -f "incremental: %e s wall, %U s user, %M KB maxrss" uvx maturin build --release -i $PY -o dist 2>&1 | grep -E "incremental:|Built wheel" | tee -a $OUT
sed -i 's| (touched).|.|' src/lib.rs
/usr/bin/time -f "incremental2: %e s wall, %U s user, %M KB maxrss" uvx maturin build --release -i $PY -o dist 2>&1 | grep -E "incremental2:|Built wheel" | tee -a $OUT
echo "== wheel" | tee -a $OUT
ls -l dist/*.whl | tee -a $OUT
unzip -l dist/*.whl | tail -4 | tee -a $OUT
echo "== cold build, crates already downloaded (cargo clean)" | tee -a $OUT
cargo clean
/usr/bin/time -f "cold(cached crates): %e s wall, %U s user, %S s sys, %P CPU, %M KB maxrss" uvx maturin build --release -i $PY -o dist 2>&1 | grep -E "cold\(|Built wheel" | tee -a $OUT
du -sh target | tee -a $OUT
unzip -o -q dist/*.whl 'decider_trees/*' -d .
echo done | tee -a $OUT
