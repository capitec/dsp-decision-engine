#!/usr/bin/env bash
# Cold build cost of libnashim.so (shim + vendored nanoarrow), three times.
set -euo pipefail
cd "$(dirname "$0")/c"
PY=../../../.venv/bin/python
for i in 1 2 3; do
  rm -f libnashim.so
  t0=$(date +%s.%N); gcc -O2 -fPIC -shared -Wall -Wextra -I../vendor -o libnashim.so nashim.c ../vendor/nanoarrow.c; t1=$(date +%s.%N)
  secs=$(echo "$t1 - $t0" | bc); size=$(stat -c %s libnashim.so)
  strip -o libnashim.stripped.so libnashim.so; ssize=$(stat -c %s libnashim.stripped.so); rm libnashim.stripped.so
  $PY -c "import sys; sys.path.insert(0,'..'); from results_io import record; record(probe='build', run=$i, seconds=float('$secs'), bytes=$size, stripped_bytes=$ssize, compiler='$(gcc --version | head -1)')"
  echo "build $i: ${secs}s, ${size} bytes (${ssize} stripped)"
done
# shim-only compile time (nanoarrow.c dominates the build)
t0=$(date +%s.%N); gcc -O2 -fPIC -c -I../vendor -o /dev/null nashim.c; t1=$(date +%s.%N); echo "shim only: $(echo "$t1 - $t0" | bc)s"
t0=$(date +%s.%N); gcc -O2 -fPIC -c -I../vendor -o /dev/null ../vendor/nanoarrow.c; t1=$(date +%s.%N); echo "nanoarrow.c only: $(echo "$t1 - $t0" | bc)s"
