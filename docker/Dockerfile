# Build:
#   docker build \
#     --build-arg="PIP_CONF=$(cat $HOME/.config/pip/pip.conf)" \
#     -t decider:latest -f docker/Dockerfile .
#
# Run:
#   docker run -p 8080:8080 \
#     -v $(pwd)/model:/opt/ml/model:ro \
#     decider:latest
#
# Environment variables (all optional):
#   DECIDER_SERVE__HOST          bind address       (default: 0.0.0.0)
#   DECIDER_SERVE__PORT          bind port          (default: 8080)
#   DECIDER_SERVE__WORKERS       worker processes   (default: nproc*2+1)
#   DECIDER_API__ROOT_PATH       project root       (default: /opt/ml/model/code)
#   DECIDER_EXT__EXTENSION_PATH  extensions dir     (default: /opt/ml/model/extensions)

ARG BASE_IMG="python:3.14-rc-slim-bookworm"
FROM ${BASE_IMG}
ARG PIP_CONF=

RUN set -aex && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Drop the pre-built wheel into the image and install with the sanic extra.
# Build the wheel first:  python -m build --wheel -o dist/
COPY dist /tmp/decider/

RUN set -ex && \
    echo "$PIP_CONF" >> /etc/pip.conf && \
    WHEEL=$(ls /tmp/decider/*.whl | head -1) && \
    pip --no-cache-dir install "${WHEEL}[serve-sanic]" && \
    rm /etc/pip.conf && \
    rm -rf /tmp/decider

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# /opt/ml/model — mounted at runtime; mirrors SageMaker & DSP path conventions.
#   model/code/        project root (generate.py, configs/, …)
#   model/extensions/  decider extensions discovered at startup
ENV DECIDER_API__ROOT_PATH="/opt/ml/model/code"
ENV DECIDER_EXT__EXTENSION_PATH="/opt/ml/model/extensions"

# Expose the default port; override with DECIDER_SERVE__PORT + -p at runtime.
EXPOSE 8080

WORKDIR /opt/ml

# `decider serve --engine sanic` picks up host/port/workers from
# DECIDER_SERVE__* env vars; workers falls back to nproc*2+1 when unset.
ENTRYPOINT ["decider", "serve", "--engine", "sanic"]
