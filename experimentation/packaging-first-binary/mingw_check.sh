#!/usr/bin/env bash
# Partial Windows evidence: cross-compile nashim.c + nanoarrow.c for Windows x86_64 with
# mingw-w64 (gcc targeting Windows: exercises the _WIN32 paths and Windows headers, NOT
# MSVC's front end). The PyInit file is left out: no Windows Python headers here.
set -euo pipefail
cd "$(dirname "$0")"
podman run --rm -v "$(pwd)/proto/src/d2shim:/src:ro,z" registry.fedoraproject.org/fedora:latest bash -c '
  dnf -y -q install mingw64-gcc mingw64-binutils >/dev/null 2>&1 && echo "toolchain: $(x86_64-w64-mingw32-gcc --version | head -1)"
  t0=$(date +%s)
  x86_64-w64-mingw32-gcc -O2 -shared -Wall -Wextra -I/src/vendor -o /tmp/nashim.dll /src/c/nashim.c /src/vendor/nanoarrow.c 2>&1 | head -20
  echo "mingw-w64 cross build: $(( $(date +%s) - t0 )) s (integer), $(stat -c %s /tmp/nashim.dll) bytes; warnings above if any"
  echo "compiler diagnostics with -Wall -Wextra -pedantic (MSVC-adjacent strictness):"
  x86_64-w64-mingw32-gcc -O2 -c -Wall -Wextra -pedantic -std=c99 -I/src/vendor -o /dev/null /src/c/nashim.c 2>&1 | grep -c warning | sed "s/^/  nashim.c warnings: /"
  x86_64-w64-mingw32-gcc -O2 -c -Wall -Wextra -pedantic -std=c99 -I/src/vendor -o /dev/null /src/vendor/nanoarrow.c 2>&1 | grep -c warning | sed "s/^/  nanoarrow.c warnings: /"
  echo "exported symbols (mingw exports all non-static by default; MSVC would export NONE without __declspec/.def/PyInit):"
  x86_64-w64-mingw32-objdump -p /tmp/nashim.dll | grep -E "\] sm_" | wc -l | sed "s/^/  sm_* exports: /"
  x86_64-w64-mingw32-objdump -p /tmp/nashim.dll | grep -E "\] sm_" | head -2
  echo "DLL imports (runtime dependencies):"; x86_64-w64-mingw32-objdump -p /tmp/nashim.dll | grep -i "DLL Name" | sed "s/^/  /"
'
