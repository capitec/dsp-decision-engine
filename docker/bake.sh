#!/bin/sh
# Install the project's requirements.txt (if it has one), then validate and warm it with `decikit build`.
# Does nothing when no project is baked in, so a plain runtime image builds too.
set -e
code=${DECIKIT_API__CODE_PATH:-/opt/ml/model/code}
if [ -z "$(ls -A "$code" 2>/dev/null)" ]; then
    echo "bake: no project in $code, skipping decikit build"
    exit 0
fi
if [ -f "$code/requirements.txt" ]; then
    if [ -n "$PIP_CONF" ]; then
        printf '%s\n' "$PIP_CONF" > "$HOME/pip.conf"
        export PIP_CONFIG_FILE="$HOME/pip.conf"
    fi
    pip install --no-cache-dir -r "$code/requirements.txt"
    rm -f "$HOME/pip.conf"
fi
exec decikit build
