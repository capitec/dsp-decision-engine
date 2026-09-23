#!/usr/bin/env bash
# Can a reviewer regenerate the vendored amalgamation from the upstream release and get
# byte-identical files? Downloads the tarball to the scratchpad (never into the repo),
# checks its sha512 against vendor/VERSION, runs upstream's bundle.py, diffs.
set -euo pipefail
cd "$(dirname "$0")"
V=proto/src/d2shim/vendor
SCRATCH=${SCRATCH:-/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/e4ec032f-4ae9-45ee-be8a-e8c600b03a03/scratchpad}
TARBALL=$SCRATCH/apache-arrow-nanoarrow-0.9.0.tar.gz
URL=https://archive.apache.org/dist/arrow/apache-arrow-nanoarrow-0.9.0/apache-arrow-nanoarrow-0.9.0.tar.gz
[ -f "$TARBALL" ] || curl -sSL -o "$TARBALL" "$URL"
echo "1. tarball sha512 vs the pin in $V/VERSION:"
PIN=$(sed -n 's/^sha512 //p' $V/VERSION); GOT=$(sha512sum "$TARBALL" | cut -d' ' -f1)
[ "$PIN" = "$GOT" ] && echo "   MATCH ($GOT)" || { echo "   MISMATCH pin=$PIN got=$GOT"; exit 1; }
echo "2. upstream signature material available for the same release (what a reviewer would also verify):"
for ext in asc sha512; do curl -sS -o /dev/null -w "   $URL.$ext -> HTTP %{http_code}\n" "$URL.$ext"; done
echo "3. regenerate the amalgamation with upstream's bundle.py and diff against the vendored copy:"
rm -rf "$SCRATCH/nano-src" "$SCRATCH/nano-out"; mkdir -p "$SCRATCH/nano-src"
tar xzf "$TARBALL" -C "$SCRATCH/nano-src"
SRC=$(ls -d "$SCRATCH"/nano-src/apache-arrow-nanoarrow-*)
(cd "$SRC" && python3 ci/scripts/bundle.py --output-dir "$SCRATCH/nano-out" > /dev/null)
find "$SCRATCH/nano-out" -type f | sed "s|$SCRATCH/nano-out/|   generated: |"
for f in nanoarrow.c nanoarrow/nanoarrow.h; do
  if cmp -s "$SCRATCH/nano-out/src/$f" "$V/$f" 2>/dev/null || cmp -s "$SCRATCH/nano-out/include/$f" "$V/$f" 2>/dev/null || cmp -s "$(find "$SCRATCH/nano-out" -path "*$f" | head -1)" "$V/$f"; then
    echo "   $f: byte-identical to the regenerated file"
  else echo "   $f: DIFFERS"; diff <(find "$SCRATCH/nano-out" -path "*$f" | head -1 | xargs cat) "$V/$f" | head -5; fi
done
for f in LICENSE.txt NOTICE.txt; do cmp -s "$SRC/$f" "$V/$f" && echo "   $f: byte-identical to the tarball's" || echo "   $f: DIFFERS from the tarball's"; done
echo "4. in-tree manifest:"; (cd $V && sha512sum -c SHA512SUMS | sed 's/^/   /')
