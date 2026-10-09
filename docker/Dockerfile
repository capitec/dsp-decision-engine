# Targets (podman commands tested; docker needs BuildKit for --build-context and --mount, not tested):
#   runtime   (default)  decider + sanic only. Mount your project at /opt/ml/model/code; nothing is baked.
#   project              runtime + a project baked in at /opt/ml/model/code, with `decider build` run at
#                        image build time so a bad config or a missing document fails the build.
#
# Run every command from the repository root.
#
# 1. runtime image, project mounted:
#      podman build -t decider:latest -f docker/Dockerfile .
#      podman run --rm -p 8080:8080 -v $PWD/my-project:/opt/ml/model/code:ro,Z \
#          -e DECIDER_API__PIPELINE=my_pkg.pipeline:build decider:latest
#    The project is a folder holding the pipeline package, configs/ and sample_request.json (decider template).
#    :Z relabels the folder for SELinux. The container user (uid 1000) must be able to read it: a folder
#    only you can read needs --userns=keep-id as well. Nothing is built, so the first start compiles everything.
#
# 2. the runtime image as a base for your own tiny Dockerfile (a requirements.txt in the project is pip-installed):
#      FROM decider:latest
#      COPY . /opt/ml/model/code
#      ENV DECIDER_API__PIPELINE=my_pkg.pipeline:build
#      RUN bake
#    `bake` is requirements.txt, then `decider build`; plain `RUN decider build` also works. The entrypoint,
#    port and the variables below are already right; override any with ENV.
#
# 3. a project directory outside this repository, as a named build context:
#      podman build --target project --build-context app=/path/to/my-project \
#          --build-arg PIPELINE=my_pkg.pipeline:build -t my-app -f docker/Dockerfile .
#    Everything the pipeline imports and every config document it reads must be inside that directory.
#
# 4. an example of this repository (the other examples come along as importable packages; the pipeline is
#    the <package>.pipeline:build named in the example's README; docker/smoke.sh does this for all of them):
#      podman build --target project --build-arg PROJECT=examples/loan-granting-pricing \
#          --build-arg PIPELINE=loan_granting.pipeline:build -t loan-granting -f docker/Dockerfile .
#
# Run a baked image, wait for /ping, and ask:
#      podman run --rm -p 8080:8080 -e DECIDER_SERVE__WORKERS=2 my-app
#      curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations
#
# Build arguments:
#   PIP_CONF      pip.conf text for an internal index, for example "$(cat $HOME/.config/pip/pip.conf)"
#   VERSION       version recorded in the wheel (the build context has no .git), default 0.0.0
#   PY            Python minor version of the base images, default 3.14
#   PROJECT       examples/<name> to bake in (target project)
#   PIPELINE      pipeline entry string of the baked project, default pipeline:build
#   CPU_TARGET    host (default) or generic: see the CPU caveat
#
# Environment variables (all optional; override with -e, or ENV in your own Dockerfile):
#   DECIDER_SERVE__HOST        bind address      (default 0.0.0.0)
#   DECIDER_SERVE__PORT        bind port         (default 8080)
#   DECIDER_SERVE__WORKERS     worker processes  (default 2 * CPUs + 1; every worker compiles its kernels
#                                                 at start, so set it)
#   DECIDER_API__CODE_PATH     project root      (default /opt/ml/model/code)
#   DECIDER_API__PIPELINE      pipeline          (default pipeline:build)
#   DECIDER_CONFIG__BASEPATH   config store      (default /opt/ml/model/code/configs)
#   NUMBA_CACHE_DIR            numba's disk cache and decider-cpu-target.json (default /opt/ml/numba-cache)
#
# What is baked: the project, its sibling example packages, its configs, and numba's disk cache for its step
# functions. That cache is keyed by source file path, so it hits because the project is at the same fixed path
# when built and when served. What is not: the fused kernels are built from LLVM IR in every worker at start and
# never cached on disk, so /ping answers 503 until each worker has done that (measured: about 15 s and 40 s
# for two examples with the cache, 35 s and 120 s without). Wait for /ping before sending traffic.
#
# CPU caveat: the baked cache is valid on the CPU that built the image. On another CPU numba misses it,
# `decider serve` warns "ran for CPU ... but this host is ...", and every worker compiles at start (slower,
# still correct). --build-arg CPU_TARGET=generic builds and serves with NUMBA_CPU_NAME=generic, so the cache
# hits anywhere on x86-64, with generic code (no AVX); no slowdown showed on the two pipelines measured.
#
# Readiness: HEALTHCHECK exists only in the docker image format, so build with `podman build --format docker`
# for `podman healthcheck run` to see it; otherwise poll GET /ping.

ARG PY=3.14
ARG CPU_TARGET=host

# Wheel, dependencies, and the Arrow shim (a C file compiled on first import) built where a compiler exists.
FROM docker.io/library/python:${PY}-bookworm AS builder
ARG PIP_CONF=
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
COPY pyproject.toml README.md /src/
COPY decider /src/decider
RUN python -m venv /opt/venv && \
    printf '%s\n' "$PIP_CONF" > /tmp/pip.conf && \
    PIP_CONFIG_FILE=/tmp/pip.conf /opt/venv/bin/pip install --no-cache-dir "/src[serve-sanic]" && \
    rm /tmp/pip.conf && \
    XDG_CACHE_HOME=/opt/ml/cache /opt/venv/bin/python -c "from decider._engine.boundary._arrow import _build; _build.build()"

FROM docker.io/library/python:${PY}-slim-bookworm AS runtime
RUN useradd --create-home --uid 1000 app && \
    mkdir -p /opt/ml/model/code /opt/ml/numba-cache && chown -R app:app /opt/ml
COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/ml/cache /opt/ml/cache
COPY --chmod=755 docker/bake.sh /usr/local/bin/bake
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    XDG_CACHE_HOME=/opt/ml/cache \
    NUMBA_CACHE_DIR=/opt/ml/numba-cache \
    DECIDER_API__CODE_PATH=/opt/ml/model/code \
    DECIDER_CONFIG__BASEPATH=/opt/ml/model/code/configs \
    DECIDER_SERVE__SERVER=sanic
USER app
WORKDIR /opt/ml
EXPOSE 8080
HEALTHCHECK --interval=5s --timeout=3s --start-period=120s \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/ping' % os.environ.get('DECIDER_SERVE__PORT', '8080'))"
ENTRYPOINT ["decider", "serve"]

# Empty unless --build-context app=<dir> is given.
FROM scratch AS app
FROM scratch AS examples
COPY examples /

# NUMBA_CPU_NAME must not be set to an empty string (numba then names an empty CPU but keeps host features),
# so the choice is a stage, not a variable.
FROM runtime AS cpu-host
FROM runtime AS cpu-generic
ENV NUMBA_CPU_NAME=generic

FROM cpu-${CPU_TARGET} AS project
ARG PROJECT=
ARG PIPELINE=
ARG PIP_CONF=
COPY --from=app --chown=app:app / /opt/ml/model/code/
RUN --mount=type=bind,from=examples,target=/examples \
    if [ -n "$PROJECT" ]; then \
        name=$(basename "$PROJECT") && \
        [ -d "/examples/$name" ] || { echo "no example named $name"; exit 1; }; \
        [ -z "$(ls -A /opt/ml/model/code)" ] || { echo "PROJECT and --build-context app both given"; exit 1; }; \
        mkdir -p /opt/ml/examples && \
        cp -r "/examples/$name/." /opt/ml/model/code/ && \
        for d in /examples/*/; do [ "$(basename "$d")" = "$name" ] || cp -r "$d" /opt/ml/examples/; done && \
        printf '%s\n' /opt/ml/examples/*/ > "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/examples.pth"; \
    fi
ENV DECIDER_API__PIPELINE=${PIPELINE:-pipeline:build}
RUN bake

# Last stage, so a build without --target gives the runtime image.
FROM runtime AS default
