#!/usr/bin/env bash
# Build libnashim.so from the shim + the vendored nanoarrow amalgamation.
# Prints the wall time and the object size; no other dependencies.
set -euo pipefail
cd "$(dirname "$0")"
t0=$(date +%s.%N)
gcc -O2 -fPIC -shared -Wall -Wextra -I../vendor -o libnashim.so nashim.c ../vendor/nanoarrow.c
t1=$(date +%s.%N)
echo "built $(pwd)/libnashim.so in $(echo "$t1 - $t0" | bc) s, $(stat -c %s libnashim.so) bytes"
