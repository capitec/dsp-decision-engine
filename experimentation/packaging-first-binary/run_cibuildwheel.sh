#!/usr/bin/env bash
# The Linux x86_64 leg of the wheel matrix, exactly as CI would run it, but
# locally in podman: cibuildwheel builds inside quay.io/pypa/manylinux_2_28_x86_64,
# repairs with auditwheel, then installs the wheel in a fresh venv and runs the tests.
# Only this leg can be run on this box; aarch64/macOS/Windows need their own runners.
set -euo pipefail
cd "$(dirname "$0")"
PY314=$(uv python find 3.14 --managed-python)
rm -rf .tmp/venv-cibw .tmp/cibw-wheelhouse
uv venv -q -p "$PY314" .tmp/venv-cibw
uv pip install -q -p .tmp/venv-cibw/bin/python cibuildwheel
t0=$(date +%s)
rm -rf proto/build   # setuptools reuses stale objects in build/ even when flags change (observed); CI must start clean
# run from inside the package dir: `test-sources` in pyproject resolves against the cwd, not the package dir
( cd proto && CIBW_CONTAINER_ENGINE=podman CIBW_BUILD="cp310-manylinux_x86_64" \
  ../.tmp/venv-cibw/bin/cibuildwheel --platform linux --output-dir ../.tmp/cibw-wheelhouse . )
echo "cibuildwheel wall time: $(( $(date +%s) - t0 )) s (includes image already pulled: $(podman images quay.io/pypa/manylinux_2_28_x86_64 --format '{{.Size}}'))"
ls -la .tmp/cibw-wheelhouse
.tmp/venv-cibw/bin/python -m pip --version >/dev/null 2>&1 || true
