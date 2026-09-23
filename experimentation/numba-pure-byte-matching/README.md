# Pure-numba byte matching — ABANDONED, superseded

This strand asked whether `prefix` / `suffix` / `substring` / `exact` string
matching could be done in pure numba over raw `uint8` bytes, with no C and no
Rust. **It never produced a measurement.** It wrote these source files on the
night of 2026-09-22 and then stalled in a retry loop for eleven hours without
writing a single line to `results.jsonl`; it was stopped.

Nothing here has been run or verified. Do not cite any of it.

**The question it was asked was answered elsewhere, properly:**
`experimentation/arrow-strings-in-tree/` built a pure-numba `STR` tree node
doing `exact` / `starts_with` / `ends_with` / `contains` over polars' own
Utf8View bytes, with 38 passing tests covering nulls, empty strings, multi-byte
UTF-8, sliced frames, all-null and zero-row columns, and multi-chunk frames.
Read that instead.

The files are kept only because `matchers.py` may be a useful reference for
byte-compare shapes in numba. They are unmeasured and untested.
