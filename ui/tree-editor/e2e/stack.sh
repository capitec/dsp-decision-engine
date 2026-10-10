#!/bin/bash
# A light stack for the e2e suite without docker compose: Postgres in a memory-capped podman (or docker)
# container and the backend serving the built UI, capped by systemd when it is available.
#
#   ./stack.sh up       build the UI if needed, start Postgres and the backend, wait for /api/health
#   ./stack.sh down     stop the backend, remove the container and its volume
#   ./stack.sh logs     the backend log
#   ./stack.sh test [pytest args]   run the suite against this stack (e.g. -m smoke)
#
# E2E_PORT (8600), E2E_PG_PORT (55432), E2E_RUNTIME (podman), E2E_STATE (<repo>/.scratch/ui-e2e-stack),
# E2E_TMPDIR (/var/tmp/dtmp-ui-e2e: temp files and browser profiles on disk, not in a RAM-backed /tmp).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
port=${E2E_PORT:-8600}
pg_port=${E2E_PG_PORT:-55432}
runtime=${E2E_RUNTIME:-podman}
state=${E2E_STATE:-$repo/.scratch/ui-e2e-stack}
pg=tree-editor-e2e-pg-$pg_port
unit=tree-editor-e2e-backend-$port
mkdir -p "$state"

up() {
  [ -f "$here/../ui/build/index.html" ] || (cd "$here/../ui" && pnpm install --frozen-lockfile && pnpm build:skip-check)
  "$runtime" run -d --replace --name "$pg" --memory=512m -e POSTGRES_PASSWORD=postgres \
    -p "127.0.0.1:$pg_port:5432" -v "$pg:/var/lib/postgresql/data" docker.io/library/postgres:17 >/dev/null
  until "$runtime" exec "$pg" pg_isready -U postgres -q; do sleep 1; done

  local env=(DSP_DB__PORT="$pg_port" DSP_UI_BUILD_PATH="$here/../ui/build" SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
             DSP_AUTH__COOKIE__SECURE=false DSP_AUTH__STATUS_COOKIE__SECURE=false PATH="$PATH")
  local cmd=(uv run --no-project --with-requirements requirements.txt --with-editable "$repo"
             sh -c "alembic upgrade head && exec uvicorn dsp_decikit.asgi:app --host 127.0.0.1 --port $port")
  cd "$here/../backend"
  if command -v systemd-run >/dev/null; then
    systemd-run --user --quiet --collect --unit "$unit" -p MemoryMax=2G -p WorkingDirectory="$PWD" \
      "${env[@]/#/--setenv=}" -p StandardOutput="file:$state/backend.log" -p StandardError="file:$state/backend.log" \
      "${cmd[@]}"
  else
    env "${env[@]}" nohup "${cmd[@]}" >"$state/backend.log" 2>&1 &
    echo $! >"$state/backend.pid"
  fi
  for _ in $(seq 180); do
    curl -fs "http://localhost:$port/api/health" >/dev/null && { echo "up on http://localhost:$port"; return; }
    sleep 1
  done
  tail -30 "$state/backend.log"; exit 1
}

down() {
  if command -v systemd-run >/dev/null; then systemctl --user stop "$unit" 2>/dev/null || true
  elif [ -f "$state/backend.pid" ]; then kill "$(cat "$state/backend.pid")" 2>/dev/null || true; fi
  "$runtime" rm -f "$pg" >/dev/null 2>&1 || true
  "$runtime" volume rm "$pg" >/dev/null 2>&1 || true
}

run_tests() {
  TMPDIR=${E2E_TMPDIR:-/var/tmp/dtmp-ui-e2e}; mkdir -p "$TMPDIR"
  cd "$here" && E2E_BASE_URL="http://localhost:$port" TMPDIR="$TMPDIR" SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv run --no-project --with-requirements requirements.txt --with-editable "$repo" pytest "$@"
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  logs) cat "$state/backend.log" ;;
  test) shift; run_tests "$@" ;;
  *) sed -n 2,10p "$0"; exit 2 ;;
esac
