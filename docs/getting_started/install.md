# Installation

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install with uv

```bash
uv add decider
```

### Optional extras

| Extra | Installs |
|---|---|
| `serve-starlette` | uvicorn + starlette for async HTTP serving |
| `serve-sanic` | sanic for high-throughput HTTP serving |
| `visualise` | streamlit + graphviz for the pipeline visualiser |
| `notebook` | IPython + Jupyter magic |

```bash
# Example: install with starlette serving and notebook support
uv add "decider[serve-starlette,notebook]"
```

## Install with pip

```bash
pip install decider
```

## Install from source

```bash
git clone https://github.com/capitecbankltd/dsp_north-polrs.git
cd dsp_north-polrs
pip install -e ".[all]"
```

## Verify installation

```bash
python -c "import decider; print(decider.__version__)"
```
