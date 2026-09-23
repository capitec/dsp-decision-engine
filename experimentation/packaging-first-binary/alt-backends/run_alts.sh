#!/usr/bin/env bash
# Build each alternative backend's miniature package, print wheel tag, size, build deps and whether it imports.
set -uo pipefail
cd "$(dirname "$0")"
PY314=$(uv python find 3.14 --managed-python)
for alt in skbuild meson hatch; do
  echo "=== $alt"
  rm -rf $alt/dist $alt/build $alt/.venv-test $alt/src/*/*.so
  t0=$(date +%s.%N)
  if uv build --python "$PY314" --wheel --out-dir $alt/dist $alt > $alt/build.log 2>&1; then
    echo "build ok in $(echo "$(date +%s.%N) - $t0" | bc) s"
  else
    echo "BUILD FAILED"; tail -15 $alt/build.log; continue
  fi
  W=$(ls $alt/dist/*.whl); echo "wheel: $(basename $W) ($(stat -c %s $W) bytes)"
  "$PY314" -c "import zipfile,sys; [print(f'  {i.file_size:8d}  {i.filename}') for i in zipfile.ZipFile(sys.argv[1]).infolist() if not i.filename.endswith(('RECORD','METADATA','WHEEL'))]" "$W"
  echo "config lines: $(cat $alt/pyproject.toml $alt/CMakeLists.txt $alt/meson.build $alt/hatch_build.py 2>/dev/null | grep -v '^\s*#' | grep -v '^\s*$' | wc -l) (non-blank, non-comment, pyproject + build file)"
  uv venv -q -p "$PY314" $alt/.venv-test && uv pip install -q -p $alt/.venv-test/bin/python "$W" \
    && $alt/.venv-test/bin/python -c "import importlib, importlib.util; m=importlib.import_module([p for p in ('d2shim_skb','d2shim_mp','d2shim_hatch') if importlib.util.find_spec(p)][0]); print('import ok: nanoarrow', m.NANOARROW_VERSION, m.LIBRARY.split('site-packages/')[-1])"
  echo "build-time packages fetched (isolated env): $(grep -oE 'Installed [0-9]+ packages|[a-z0-9_-]+==[0-9.]+' $alt/build.log | sort -u | tr '\n' ' ' | cut -c1-300)"
done
