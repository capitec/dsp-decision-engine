#!/usr/bin/env bash
# What a user sees on a platform with NO wheel, i.e. pip falls back to the sdist.
#   A. sdist, no C compiler on PATH            -> install FAILS (default pyproject)
#   B. sdist, no C compiler, optional = true   -> install SUCCEEDS, pure backend + one warning
#   C. sdist, compiler present                 -> compiled backend (compile-on-install)
#   D. runtime: wheel installed but extension unloadable -> fallback warning / forced error
set -uo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
PY314=$(uv python find 3.14 --managed-python)
SDIST=$(ls .tmp/dist/d2shim-0.0.1.tar.gz)
# a PATH with python and uv only: no gcc, no cc
rm -rf .tmp/nocc-bin; mkdir -p .tmp/nocc-bin; ln -s "$(command -v uv)" .tmp/nocc-bin/uv; ln -s "$PY314" .tmp/nocc-bin/python3.14
NOCC_PATH="$HERE/.tmp/nocc-bin"

echo "== A. sdist install with no compiler on PATH (default config: the extension is required)"
rm -rf .tmp/venv-A; uv venv -q -p "$PY314" .tmp/venv-A
env -i HOME="$HOME" PATH="$NOCC_PATH" uv pip install -q -p .tmp/venv-A/bin/python "$SDIST" > .tmp/simA.log 2>&1
echo "  exit code $? ; the error the user sees (last lines):"; grep -vE '^\s*$' .tmp/simA.log | grep -E "error|gcc|cc|Failed|hint" | tail -6 | sed 's/^/  | /'

echo "== A2. sdist install with a compiler but NO Python headers (this box's /usr/bin/python3.14: python3-devel not installed)"
rm -rf .tmp/venv-A2; uv venv -q -p /usr/bin/python3.14 .tmp/venv-A2
uv pip install -q -p .tmp/venv-A2/bin/python "$SDIST" > .tmp/simA2.log 2>&1
echo "  exit code $? ; the error the user sees:"; grep -E "fatal error|Python.h" .tmp/simA2.log | head -2 | sed 's/^/  | /'

echo "== B. same, but with 'optional = true' on the extension"
rm -rf .tmp/proto-optional; cp -r proto .tmp/proto-optional; rm -rf .tmp/proto-optional/build .tmp/proto-optional/src/*.egg-info
sed -i 's|^# optional = true .*|optional = true|' .tmp/proto-optional/pyproject.toml
rm -rf .tmp/dist-optional; uv build --python "$PY314" --sdist --out-dir .tmp/dist-optional .tmp/proto-optional > /dev/null 2>&1
rm -rf .tmp/venv-B; uv venv -q -p "$PY314" .tmp/venv-B
env -i HOME="$HOME" PATH="$NOCC_PATH" uv pip install -q -p .tmp/venv-B/bin/python "$(ls .tmp/dist-optional/*.tar.gz)" > .tmp/simB.log 2>&1
echo "  exit code $?"; grep -iE "warning|optional|skipp" .tmp/simB.log | head -3 | sed 's/^/  | /'
echo "  what got installed:"; ls .tmp/venv-B/lib/python3.14/site-packages/d2shim/ | tr '\n' ' '; echo
echo "  first import in a fresh process:"
.tmp/venv-B/bin/python -W always -c "import d2shim; print('  which() ->', d2shim.which())" 2>&1 | sed 's/^/  | /'
echo "  D2SHIM_BACKEND=compiled (production strictness):"
D2SHIM_BACKEND=compiled .tmp/venv-B/bin/python -c "import d2shim; d2shim.which()" 2>&1 | tail -1 | sed 's/^/  | /'

echo "== C. sdist install WITH a compiler (compile-on-install)"
rm -rf .tmp/venv-C; uv venv -q -p "$PY314" .tmp/venv-C
t0=$(date +%s.%N); uv pip install -q -p .tmp/venv-C/bin/python "$SDIST" > .tmp/simC.log 2>&1; echo "  exit code $? in $(echo "$(date +%s.%N) - $t0" | bc) s (isolated build env: fetch setuptools, compile 3 C files, install)"
.tmp/venv-C/bin/python -c "import d2shim; print('  which() ->', d2shim.which()[0])"

echo "== D. wheel installed, but the extension cannot be loaded at runtime (simulated with sys.modules)"
V=.tmp/venv314/bin/python
echo "  default policy (auto):"
$V -W always -c "import sys; sys.modules['d2shim._nashim']=None; import d2shim; print('  which() ->', d2shim.which())" 2>&1 | sed 's/^/  | /'
echo "  D2SHIM_BACKEND=compiled:"
D2SHIM_BACKEND=compiled $V -c "import sys; sys.modules['d2shim._nashim']=None; import d2shim; d2shim.which()" 2>&1 | tail -1 | sed 's/^/  | /'
echo "  a corrupt/foreign .so in place of the extension (ABI mismatch is caught the same way):"
rm -rf .tmp/venv-D; uv venv -q -p "$PY314" .tmp/venv-D; uv pip install -q -p .tmp/venv-D/bin/python "$(ls .tmp/dist/*.whl)"
printf 'garbage' > .tmp/venv-D/lib/python3.14/site-packages/d2shim/_nashim.abi3.so
.tmp/venv-D/bin/python -W always -c "import d2shim; print('  which() ->', d2shim.which())" 2>&1 | sed 's/^/  | /'
