#!/bin/bash
# Raw and gzip -9 bytes of each file given (a wheel, a .wasm, its JS glue, an npm tarball).
#   ./sizes.sh <file>...
set -euo pipefail
printf '%12s %12s  %s\n' raw gzip-9 file
for f in "$@"; do
  printf '%12d %12d  %s\n' "$(stat -c %s "$f")" "$(gzip -9 -c "$f" | wc -c)" "$f"
done
