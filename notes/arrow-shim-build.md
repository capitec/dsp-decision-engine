# The Arrow shim: built on first import, no pure-Python path

**Decision:**
- The data boundary reads a polars frame through one `__arrow_c_stream__()`
  export, decoded by vendored nanoarrow 0.9.0 behind a small C shim
  (`engine/boundary/_arrow/c/shim.c`). One C call (`sm_columns`) reads every
  declared column of the batch, one column at a time, into one buffer.
- A column the kernel can read as it is (float64 as F64, int64 as I64,
  int32 as CODE) with no nulls, in a batch of 4096 rows or more, is not
  copied: its numpy array is a read-only view of the exported Arrow buffer.
  The exported `ArrowArray` is moved (a bitwise move, which the C data
  interface allows) into a small Python owner that every such view holds, so
  the buffer lives exactly as long as the last array reading it, whatever
  happens to the frame. Below 4096 rows a copy is cheaper than wrapping.
- The per-row gather (`sm_gather_row`, `FrameView.gather`/`materialize`) is
  kept as the reference the column path is tested against, and for callers
  that want one row at a time.
- String (`bytes`) inputs reach a kernel as `(address, length)` spans from the
  same call. A compiled run reads the input frame's own column when the state
  still holds the values read from it (`State.source`); an override or a row
  subset is copied into a new Series first.
- Nothing is re-exported from `engine/boundary`, so importing it never builds
  the shim: `extract`, `dtypes.explain_boundary` and `_arrow.view`/`kernels`
  need the compiled shim; the rest of `dtypes`, `nulls`, `_arrow.plan`,
  `_arrow.intrinsics` and `_arrow.diagnose()` don't. The tree walker imports
  the `load_*` intrinsics without loading the shim.
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
  constants, so `cache=True` kernels that call it still disk-cache. The same
  holds for the `load_*` intrinsics: every address is an integer argument.
- The row gather this replaced was O(rows × columns) with one C call per row
  (through a numba loop) plus a copy of every column: fused `run()` on the
  flagship took 97.6 ms at 1M rows against 3.1 ms reading `to_numpy()`
  columns. A one-row call spent most of its time in numpy and ctypes calls
  (about a microsecond each): one allocation for the whole batch, one C call,
  and addresses taken once per plan fixed that. Measured before and after in
  `single-record-path.md`.

**Open:** a deployment image with a read-only home or no compiler needs the
`.so` built at image build time (import the boundary once) or a prebuilt wheel.
Windows (MSVC) is not supported by the first-import build.

**Source:** `decider2/src/decider2/_arrow/`, `decider2/pyproject.toml`,
`experimentation/packaging-first-binary/`.
