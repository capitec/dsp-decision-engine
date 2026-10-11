#!/bin/bash
# A light stack for the e2e suite without docker compose: Postgres in a memory-capped podman (or docker)
# container and the backend serving the built UI, capped by systemd when it is available.
#
#   ./stack.sh up       build the UI if needed, start Postgres and the backend, wait for /api/health
#   ./stack.sh down     stop the backend, remove the container and its volume
#   ./stack.sh restart  restart the backend (every node), keeping Postgres
#   ./stack.sh logs     the backend log
#   ./stack.sh test [pytest args]   run the suite against this stack (e.g. -m smoke)
#
# E2E_PORT (8600), E2E_PG_PORT (55432), E2E_RUNTIME (podman), E2E_STATE (<repo>/.scratch/ui-e2e-stack),
# E2E_TMPDIR (/var/tmp/dtmp-ui-e2e: temp files and browser profiles on disk, not in a RAM-backed /tmp),
# E2E_SERVE_PORT (8650: the first of the four ports the `decikit serve` tests use),
# E2E_NODES (1; 2 adds a second backend on E2E_PORT+1 sharing Postgres and logins), DSP_LIVE_COLLAB (live editing).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
port=${E2E_PORT:-8600}
pg_port=${E2E_PG_PORT:-55432}
runtime=${E2E_RUNTIME:-podman}
state=${E2E_STATE:-$repo/.scratch/ui-e2e-stack}
nodes=${E2E_NODES:-1}
live=${DSP_LIVE_COLLAB:-false}
pg=tree-editor-e2e-pg-$pg_port
mkdir -p "$state"

# Node n listens on port+n-1 and logs to backend.log (node 1) or backend-<n>.log.
node_port() { echo $((port + $1 - 1)); }
node_log() { [ "$1" = 1 ] && echo "$state/backend.log" || echo "$state/backend-$1.log"; }

start_node() {
  local n=$1 p; p=$(node_port "$1")
  local env=(DSP_DB__PORT="$pg_port" DSP_UI_BUILD_PATH="$here/../ui/build" SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
             DSP_AUTH__COOKIE__SECURE=false DSP_AUTH__STATUS_COOKIE__SECURE=false DSP_LIVE_COLLAB="$live"
             PATH="$PATH")
  # Two nodes need logins both can read: the in-memory cache is per process.
  [ "$nodes" -gt 1 ] && env+=(DSP_CACHE='{"provider": "persisted-cache", "cache": {"provider": "in-memory"}}')
  # merge-core is built from source (cargo and maturin on PATH) and cached until its Rust sources change.
  local cmd=(uv run --no-project --with-requirements requirements.txt --with-editable "$repo" --with "$here/../merge-core"
             sh -c "alembic upgrade head && exec uvicorn dsp_decikit.asgi:app --host 127.0.0.1 --port $p")
  cd "$here/../backend"
  if command -v systemd-run >/dev/null; then
    systemd-run --user --quiet --collect --unit "tree-editor-e2e-backend-$p" -p MemoryMax=2G -p WorkingDirectory="$PWD" \
      "${env[@]/#/--setenv=}" -p StandardOutput="append:$(node_log "$n")" -p StandardError="append:$(node_log "$n")" \
      "${cmd[@]}"
  else
    env "${env[@]}" nohup "${cmd[@]}" >>"$(node_log "$n")" 2>&1 &
    echo $! >"$state/backend-$p.pid"
  fi
}

stop_nodes() {
  local n p
  for n in $(seq "$nodes"); do
    p=$(node_port "$n")
    if command -v systemd-run >/dev/null; then systemctl --user stop "tree-editor-e2e-backend-$p" 2>/dev/null || true
    elif [ -f "$state/backend-$p.pid" ]; then kill "$(cat "$state/backend-$p.pid")" 2>/dev/null || true; fi
  done
}

start_backend() {
  local n p
  # The second node starts after the first has migrated the database.
  for n in $(seq "$nodes"); do
    start_node "$n"
    p=$(node_port "$n")
    for _ in $(seq 180); do
      curl -fs "http://localhost:$p/api/health" >/dev/null && { echo "up on http://localhost:$p"; continue 2; }
      sleep 1
    done
    tail -30 "$(node_log "$n")"; exit 1
  done
}

up() {
  [ -f "$here/../ui/build/index.html" ] || (cd "$here/../ui" && pnpm install --frozen-lockfile && pnpm build:skip-check)
  "$runtime" run -d --replace --name "$pg" --memory=512m -e POSTGRES_PASSWORD=postgres \
    -p "127.0.0.1:$pg_port:5432" -v "$pg:/var/lib/postgresql/data" docker.io/library/postgres:17 >/dev/null
  until "$runtime" exec "$pg" pg_isready -U postgres -q; do sleep 1; done
  start_backend
}

down() {
  stop_nodes
  "$runtime" rm -f "$pg" >/dev/null 2>&1 || true
  "$runtime" volume rm "$pg" >/dev/null 2>&1 || true
}

run_tests() {
  TMPDIR=${E2E_TMPDIR:-/var/tmp/dtmp-ui-e2e}; mkdir -p "$TMPDIR"
  local second=()
  [ "$nodes" -gt 1 ] && second=(E2E_BASE_URL_2="http://localhost:$(node_port 2)")
  cd "$here" && env E2E_BASE_URL="http://localhost:$port" "${second[@]}" E2E_LIVE_COLLAB="$live" TMPDIR="$TMPDIR" \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    uv run --no-project --with-requirements requirements.txt --with-editable "$repo" pytest "$@"
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  restart) stop_nodes; start_backend ;;
  logs) cat "$state"/backend*.log ;;
  test) shift; run_tests "$@" ;;
  *) sed -n 2,12p "$0"; exit 2 ;;
esac
