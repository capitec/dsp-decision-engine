"""The data boundary: a polars frame into typed kernel columns in one Arrow export, and outputs back out.

`extract` and `dtypes.explain_boundary` need the compiled Arrow shim (built
with the C compiler on first import); the rest of `dtypes`, `writeback`,
`nulls` and `_arrow.diagnose()` don't, which is why nothing is re-exported here.
"""
