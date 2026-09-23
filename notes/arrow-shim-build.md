# The Arrow shim: built on first import, no pure-Python path

**Decision:**
- The data boundary reads a polars frame through one `__arrow_c_stream__()`
  export, decoded by vendored nanoarrow 0.9.0 behind a small C shim
  (`engine/boundary/_arrow/c/shim.c`). One C call per row gathers every
  declared column into a typed row.
- The shim is a plain shared library loaded with `ctypes`, not a CPython
  extension. It is compiled with `$CC` (default `cc`) on first import into
  `$XDG_CACHE_HOME/decider/arrow-shim/shim-<hash>.so`. The hash covers the C
  sources, the header, the flags, the compiler and the machine, so an edit
  can't serve a stale build. The write is atomic (`os.replace`).
- nanoarrow is compiled with `NANOARROW_NAMESPACE=DeciderArrow`. Every
  exported symbol is `sm_*` or `DeciderArrow*`, so another nanoarrow in the
  process (pyarrow, adbc) can't interpose.
- There is no pure-Python fallback, as in decider2. A machine without a C
  compiler gets one `ImportError` naming the platform, interpreter and cause;
  `decider.engine.boundary._arrow.diagnose()` reports the same without raising.
  Tests that need the shim skip on such a machine.
- A REQUIRED input with a null (or an absent column) is an error naming the
  input, the step path and the null row count. decider2's refer/decline row
  routing and `NOT_APPLICABLE_AS` are gone (Design D8).

**Why:**
- `decider` builds with hatchling, which has no C-extension support; decider2
  used setuptools `ext-modules`. Building on first import with the fixed C
  source avoids a second build backend and needs only a compiler, not the
  CPython headers (the system Python on the dev box has none).
- Kernels take the shim's function addresses as arguments, never as captured
  constants, so `cache=True` kernels that call it still disk-cache.
- The gather is O(rows × columns), one C call per row (~149 ns/row at 17
  columns in decider2), about 4.6× slower than per-column numpy paths. It was
  accepted for one code path across every mode and dtype.

**Open:** a deployment image with a read-only home or no compiler needs the
`.so` built at image build time (import the boundary once) or a prebuilt wheel.
Windows (MSVC) is not supported by the first-import build.

**Source:** `decider2/src/decider2/_arrow/`, `decider2/pyproject.toml`,
`experimentation/packaging-first-binary/`.
