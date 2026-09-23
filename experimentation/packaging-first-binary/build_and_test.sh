#!/usr/bin/env bash
# Reproduce the prototype build end to end, in throwaway venvs under .tmp/.
#   1. build sdist + wheel with the project's own backend (setuptools, isolated build env)
#   2. show the wheel tag and contents (expect cp310-abi3-linux_x86_64, LICENSE+NOTICE in dist-info)
#   3. auditwheel: what the .so links against, then repair to a manylinux tag
#   4. install the wheel into a fresh 3.14 venv and run the tests
#   5. install the SAME wheel into a 3.12 venv: the abi3 tag means one wheel per platform
# Needs: gcc, uv, a uv-managed CPython (ships Python.h; the Fedora system python here does not).
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
PY314=$(uv python find 3.14 --managed-python)
PY312=$(uv python find 3.12 --managed-python)
mkdir -p .tmp; rm -rf proto/build proto/src/d2shim.egg-info .tmp/dist .tmp/venv314 .tmp/venv312 .tmp/venv-audit .tmp/wheelhouse
echo "== 1. build (uv build: PEP 517 isolated env, setuptools fetched from the pin in pyproject)"
t0=$(date +%s.%N)
uv build --python "$PY314" --out-dir .tmp/dist proto 2>&1 | tail -3
t1=$(date +%s.%N); echo "build wall time: $(echo "$t1 - $t0" | bc) s"
ls -la .tmp/dist
echo "== 2. wheel contents"
WHL=$(ls .tmp/dist/*.whl)
"$PY314" -c "import zipfile,sys; [print(f'  {i.file_size:8d}  {i.filename}') for i in zipfile.ZipFile(sys.argv[1]).infolist()]" "$WHL"
echo "== 3. auditwheel"
uv venv -q -p "$PY314" .tmp/venv-audit; uv pip install -q -p .tmp/venv-audit/bin/python auditwheel patchelf
.tmp/venv-audit/bin/auditwheel show "$WHL" | sed 's/^/  /'
if .tmp/venv-audit/bin/auditwheel repair --plat manylinux_2_28_x86_64 -w .tmp/wheelhouse "$WHL" > .tmp/repair.log 2>&1; then
  MWHL=$(ls .tmp/wheelhouse/*.whl); echo "  repaired: $MWHL"
else
  echo "  auditwheel REFUSED to tag this dev-box build manylinux_2_28 (only consistent with $(grep -o 'manylinux_2_[0-9]*_x86_64' .tmp/repair.log | sort -u | tail -1)):"
  echo "  a manylinux_2_28 wheel has to be built inside the manylinux container (cibuildwheel, ./run_cibuildwheel.sh)."
  MWHL="$WHL"
fi
echo "== 4. install the repaired wheel into a fresh 3.14 venv and test"
uv venv -q -p "$PY314" .tmp/venv314
uv pip install -q -p .tmp/venv314/bin/python "$MWHL" pytest
.tmp/venv314/bin/python -c "import d2shim, json; print(json.dumps(d2shim.diagnose(), indent=1))"
(cd proto && "$HERE/.tmp/venv314/bin/python" -m pytest tests -q 2>&1 | tail -3)
echo "== 5. same wheel, CPython 3.12 (abi3: one wheel per platform)"
uv venv -q -p "$PY312" .tmp/venv312
uv pip install -q -p .tmp/venv312/bin/python "$MWHL" pytest
.tmp/venv312/bin/python -c "import d2shim; print(d2shim.which())"
(cd proto && "$HERE/.tmp/venv312/bin/python" -m pytest tests -q 2>&1 | tail -1)
echo "== done"
