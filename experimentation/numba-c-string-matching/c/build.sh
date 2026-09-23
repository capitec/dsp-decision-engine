#!/usr/bin/env bash
# Build libstrmatch.so against the system libpcre2-8 (runtime lib only; no
# headers needed, prototypes are declared in strmatch.c). ~0.2 s.
set -euo pipefail
cd "$(dirname "$0")"
gcc -O2 -fPIC -shared -Wall -Wextra -o libstrmatch.so strmatch.c -l:libpcre2-8.so.0
echo "built $(pwd)/libstrmatch.so"
