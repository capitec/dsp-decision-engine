#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
V=../../nanoarrow-accessors/vendor
t0=$(date +%s.%N)
gcc -O2 -fPIC -shared -Wall -I$V -o librowshim.so rowshim.c $V/nanoarrow.c -lm
t1=$(date +%s.%N)
echo "built librowshim.so in $(echo "$t1 - $t0" | bc) s, $(stat -c %s librowshim.so) bytes"
