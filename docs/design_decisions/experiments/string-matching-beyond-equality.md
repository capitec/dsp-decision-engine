# Can a kernel match prefix, suffix, contains and regex on strings, and in which language?

Date: 2026-09-22 to 2026-09-23 · Script: spikes, in decider-experimentation/ `arrow-strings-in-tree`, `numba-c-string-matching`, `rust-string-matching` (`numba-pure-byte-matching` stalled and never measured)

## Purpose
Find how a tree node should test a string beyond `==`, whether the test should run lazily at the node (a short-circuiting AND reaches it for few rows) or over the whole column first, and whether C or Rust earns a place for regex.

## Method
Rule `ft1 > c AND ft2 > c AND ft3 ~ /^dog/` over 1M rows, 0/1/50/100% reaching the string node, low cardinality (12 distinct) and high (one per row). Compared: polars `str.contains` over the column, a per-category mask, a pure-numba node reading polars' Utf8View, PCRE2-JIT and the Rust `regex` crate called from the kernel. Answers asserted equal to a Python `re` oracle. Polars 1.41, shared 28-core box (+-20-30%).

## Result
- **Zero-copy**: `Series.__arrow_c_stream__()` hands over polars' own buffers in 1.2 µs from 100k to 4M rows and 0 MB; `_get_buffers()` copies, 17-28 ns/row and about 44 MB per call at 4M rows.
- **A string node in pure numba** reads the 16-byte view (length, then inline bytes up to 12 or buffer index and offset) and does `exact`, `starts_with`, `ends_with`, `contains`. ns/row at 1M rows against casting to Categorical first: low cardinality `==` 56-75 vs 178-185, high 38-55 vs 800-1100, gated at 1% 17-33 vs 62-198. A reached string node costs 25-40 ns more than a numeric one. A prefix is a `memcmp`, 9-14 ns.
- **Regex per call**: PCRE2-JIT 29-40 ns, Rust `regex` about 40 (not the 25.6 first measured), polars 24-46, glibc `regexec` 71-248, Python `re` 622-641. Lazy at the node against the column pass: 15-24 vs 33-62 ms per 1M rows with no row reaching it, level at 50-75%. The whole prize is 20-40 ns/row.
- **Per-category mask** (run the pattern over the distinct values, look up by code): 0.8-1.8 ns/row, but only when the column already arrives dictionary-encoded; encoding it yourself is 47 ns/row and the mask loses to the column pass (50.5 vs 34).
- **Hazards**: PCRE2 backtracks, so `^(a+)+$` costs 211 µs/row with the shim's limit and 42 ms/row at its default; the Rust and polars engines are linear. An unprotected Rust panic aborts the process. A kernel that captures the C symbol as a global cannot be disk-cached; one that takes the address as an argument can.
- Case folding and trimming are a few lines for ASCII only; Unicode folding needs tables.

## Decision
Strings reach the kernel as spans over Arrow views; `exact`, `starts_with`, `ends_with` and `contains` run in numba, and regex (also case folding, trimming) runs in Python with a per-step fallback. No C or Rust regex, no per-category masks. If a profile ever shows a selective regex over a high-cardinality column, pick Rust (linear time) over PCRE2. See [strings](../decisions/strings.md) and [string representations](string-representations.md).
