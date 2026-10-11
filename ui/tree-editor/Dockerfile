# Build from the repo root, after `uv build` has put the decikit wheel in dist/:
#   docker build -f ui/tree-editor/Dockerfile -t decider-tree-editor .

# The merge-core extension (wheel) and its WebAssembly build (pkg/), one toolchain stage for both.
FROM rust:1.95-bookworm AS merge-core

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv
RUN rustup target add wasm32-unknown-unknown && \
    cargo install --locked wasm-pack@0.15.0 && \
    uv tool install 'maturin[zig]==1.15.0'

WORKDIR /src
COPY ui/tree-editor/merge-core/Cargo.toml ui/tree-editor/merge-core/Cargo.lock ui/tree-editor/merge-core/pyproject.toml ./
COPY ui/tree-editor/merge-core/src src
COPY ui/tree-editor/merge-core/python python
# zig links against glibc 2.28, so the abi3 wheel installs on any glibc from 2.28 and every Python from 3.10.
RUN uv tool run --from 'maturin[zig]==1.15.0' maturin build --release --zig --compatibility manylinux_2_28 -o /wheels && \
    wasm-pack build --target web --release --no-pack -- --features wasm

FROM node:24-alpine AS ui-builder

WORKDIR /app/ui

RUN npm install -g pnpm@12.9.1

# Every workspace member's manifest has to be present for a frozen install.
COPY ui/pnpm-workspace.yaml ui/pnpm-lock.yaml ./
COPY ui/lib/package.json lib/
COPY ui/vscode/package.json vscode/
COPY ui/tree-editor/ui tree-editor/ui
COPY --from=merge-core /src/pkg tree-editor/merge-core/pkg

RUN pnpm install --frozen-lockfile --filter dsp-re-ui && \
    cd tree-editor/ui && \
    pnpm build:skip-check && \
    test -d build

FROM python:3.13-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid 1000 appgroup && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" appuser

COPY dist /tmp/decikit/
COPY ui/tree-editor/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir /tmp/decikit/*.whl -r requirements.txt && \
    rm -rf /tmp/decikit

# After the big layer above: a change to merge-core reinstalls only this.
COPY --from=merge-core /wheels /tmp/merge-core/
RUN pip install --no-cache-dir /tmp/merge-core/*.whl && \
    rm -rf /tmp/merge-core

# ponytail: migrations run from this image; split into a job image if it grows
COPY ui/tree-editor/backend/alembic ./alembic
COPY ui/tree-editor/backend/alembic.ini ./alembic.ini
COPY ui/tree-editor/backend/dsp_decikit ./dsp_decikit
COPY --from=ui-builder /app/ui/tree-editor/ui/build ./ui/build

RUN mkdir -p /app/logs /var/cache/dsp-decikit && \
    chown -R appuser:appgroup /app /var/cache/dsp-decikit && \
    chmod -R 755 /var/cache/dsp-decikit

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DSP_UI_BUILD_PATH=/app/ui/build

ENTRYPOINT [ "uvicorn" ]
CMD ["dsp_decikit.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
