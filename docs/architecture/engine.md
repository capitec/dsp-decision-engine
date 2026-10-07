# The engine

!!! note "Draft"
    Outline only.

- `Engine.bind(step)` returns an `Executable`: `run`, `score`, `session`
- Modes: interpreted, stepped, fused
- Compilation: numba kernels, the compile cache keyed by content, what falls back to Python
- Boundary: moving polars and Arrow data in and out of kernels
- Params: how documents are validated, bundled and passed to kernels
- Runners as generators: one code path for running and debugging
- Debug sessions and decision traces
