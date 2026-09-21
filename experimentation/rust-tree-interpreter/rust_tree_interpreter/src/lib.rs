//! One generic tree-walking kernel, called from Python via PyO3.
//!
//! Mirrors, field-for-field, `tree-codegen-vs-interpreted/interpreted_kernel.py`'s
//! `walk_batch` (the numba struct-of-arrays walker): same node encoding
//! (kind/feat_idx/op_code/thresh/pat_start/pat_count/patterns/left/right/
//! leaf_value), same OR-of-exact-dictionary-code string semantics. That is
//! deliberate: the comparison is supposed to isolate the ENGINE (numba JIT
//! vs. a native Rust extension), not the encoding.
//!
//! Per the experiment brief: Rust is called ONCE PER BATCH with arrays
//! (`walk_batch`, GIL released for the walk). `walk_one` exists ONLY to
//! measure single-record FFI/PyO3 marshalling cost against decider2's own
//! realtime floor — never for batch use.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use regex::Regex;
use std::time::Instant;

const KIND_LEAF: i8 = 0;
const KIND_NUM: i8 = 1;
// KIND_STR = 2 is implicit: any non-leaf, non-numeric node is a string
// match node, exactly as interpreted_kernel.py's `else` branch treats it.

/// The kernel. No recursion (tree depth never touches the call stack,
/// same property the numba walker has), no heap allocation, no Python
/// object anywhere in this function.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn walk_one_inner(
    row: &[f64],
    str_row: &[i32],
    kind: &[i8],
    feat_idx: &[i32],
    op_code: &[i8],
    thresh: &[f64],
    pat_start: &[i32],
    pat_count: &[i32],
    patterns: &[i32],
    left: &[i32],
    right: &[i32],
    leaf_value: &[f64],
) -> f64 {
    let mut node: usize = 0;
    while kind[node] != KIND_LEAF {
        let cond = if kind[node] == KIND_NUM {
            let f = row[feat_idx[node] as usize];
            let t = thresh[node];
            match op_code[node] {
                0 => f < t,
                1 => f <= t,
                2 => f == t,
                3 => f > t,
                4 => f >= t,
                _ => f != t,
            }
        } else {
            let code = str_row[feat_idx[node] as usize];
            let start = pat_start[node] as usize;
            let end = start + pat_count[node] as usize;
            patterns[start..end].iter().any(|&p| p == code)
        };
        node = if cond { left[node] as usize } else { right[node] as usize };
    }
    leaf_value[node]
}

#[pyo3::pymodule]
mod rust_tree_interpreter {
    use super::*;
    use pyo3::prelude::*;

    /// Walk every row of `numeric_cols`/`string_cols` against the same
    /// flat tree, writing `out[i]`. Called ONCE for the whole batch; the
    /// GIL is released for the entire walk (`py.allow_threads`), matching
    /// the brief: "crossing into Rust is cheap; crossing through a Python
    /// binding per row is not."
    #[pyfunction]
    #[pyo3(signature = (numeric_cols, string_cols, kind, feat_idx, op_code, thresh,
                         pat_start, pat_count, patterns, left, right, leaf_value, out))]
    #[allow(clippy::too_many_arguments)]
    fn walk_batch<'py>(
        py: Python<'py>,
        numeric_cols: PyReadonlyArray2<'py, f64>,
        string_cols: PyReadonlyArray2<'py, i32>,
        kind: PyReadonlyArray1<'py, i8>,
        feat_idx: PyReadonlyArray1<'py, i32>,
        op_code: PyReadonlyArray1<'py, i8>,
        thresh: PyReadonlyArray1<'py, f64>,
        pat_start: PyReadonlyArray1<'py, i32>,
        pat_count: PyReadonlyArray1<'py, i32>,
        patterns: PyReadonlyArray1<'py, i32>,
        left: PyReadonlyArray1<'py, i32>,
        right: PyReadonlyArray1<'py, i32>,
        leaf_value: PyReadonlyArray1<'py, f64>,
        mut out: PyReadwriteArray1<'py, f64>,
    ) -> PyResult<()> {
        let numeric = numeric_cols.as_array();
        let strings = string_cols.as_array();
        let kind_s = kind.as_slice()?;
        let feat_idx_s = feat_idx.as_slice()?;
        let op_code_s = op_code.as_slice()?;
        let thresh_s = thresh.as_slice()?;
        let pat_start_s = pat_start.as_slice()?;
        let pat_count_s = pat_count.as_slice()?;
        let patterns_s = patterns.as_slice()?;
        let left_s = left.as_slice()?;
        let right_s = right.as_slice()?;
        let leaf_value_s = leaf_value.as_slice()?;
        let out_s = out.as_slice_mut()?;

        let n = numeric.shape()[0];

        // SAFETY-relevant note: everything captured below is a plain slice
        // / ndarray view over memory owned by numpy, valid for 'py — no
        // Python object, no refcounting, so releasing the GIL here is
        // just as safe as it is for the numba kernel it mirrors.
        py.detach(move || {
            for i in 0..n {
                let row = numeric.row(i);
                let str_row = strings.row(i);
                let row_s = row.as_slice().expect("numeric row not contiguous");
                let str_s = str_row.as_slice().expect("string row not contiguous");
                out_s[i] = walk_one_inner(
                    row_s, str_s, kind_s, feat_idx_s, op_code_s, thresh_s,
                    pat_start_s, pat_count_s, patterns_s, left_s, right_s, leaf_value_s,
                );
            }
        });
        Ok(())
    }

    /// ONE row through the FFI — exists only to measure per-call PyO3
    /// binding overhead against decider2's realtime floor (EXPERIMENTS.md
    /// §R: kernel ~1.4 µs inside a ~1.5 ms whole-path budget). Never call
    /// this in a batch loop from Python; that is exactly the per-row
    /// binding cost §R found expensive for ZEN's naive API.
    #[pyfunction]
    #[pyo3(signature = (row, str_row, kind, feat_idx, op_code, thresh,
                         pat_start, pat_count, patterns, left, right, leaf_value))]
    #[allow(clippy::too_many_arguments)]
    fn walk_one<'py>(
        row: PyReadonlyArray1<'py, f64>,
        str_row: PyReadonlyArray1<'py, i32>,
        kind: PyReadonlyArray1<'py, i8>,
        feat_idx: PyReadonlyArray1<'py, i32>,
        op_code: PyReadonlyArray1<'py, i8>,
        thresh: PyReadonlyArray1<'py, f64>,
        pat_start: PyReadonlyArray1<'py, i32>,
        pat_count: PyReadonlyArray1<'py, i32>,
        patterns: PyReadonlyArray1<'py, i32>,
        left: PyReadonlyArray1<'py, i32>,
        right: PyReadonlyArray1<'py, i32>,
        leaf_value: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        Ok(walk_one_inner(
            row.as_slice()?,
            str_row.as_slice()?,
            kind.as_slice()?,
            feat_idx.as_slice()?,
            op_code.as_slice()?,
            thresh.as_slice()?,
            pat_start.as_slice()?,
            pat_count.as_slice()?,
            patterns.as_slice()?,
            left.as_slice()?,
            right.as_slice()?,
            leaf_value.as_slice()?,
        ))
    }

    /// §T's per-category-mask case, using the `regex` crate instead of
    /// numba->libc. `categories` is the small distinct-value list (e.g. 12
    /// employment-status/region codes); timing starts AFTER Python->Rust
    /// argument marshalling (pattern compile + the loop only), so the
    /// returned ns/call is the pure-Rust-regex number, comparable to
    /// EXPERIMENTS.md §T's "0.7 µs of regex" component.
    #[pyfunction]
    fn regex_bench_categories(pattern: &str, categories: Vec<String>) -> PyResult<(Vec<bool>, f64)> {
        let re = Regex::new(pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let t0 = Instant::now();
        let matches: Vec<bool> = categories.iter().map(|c| re.is_match(c)).collect();
        let dt = t0.elapsed().as_secs_f64();
        let n = categories.len().max(1) as f64;
        Ok((matches, dt * 1e9 / n))
    }

    /// §T's per-ROW case (cardinality ~= row count: account numbers, free
    /// text). Same timing convention as `regex_bench_categories` — pure
    /// in-Rust regex-crate cost, comparable to §T's "polars' Rust `regex`
    /// crate" reference point (42.2 ns/call) and to glibc POSIX regex via
    /// numba (60.3 ns/call).
    #[pyfunction]
    fn regex_bench_rows(pattern: &str, haystacks: Vec<String>) -> PyResult<(Vec<bool>, f64)> {
        let re = Regex::new(pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let t0 = Instant::now();
        let matches: Vec<bool> = haystacks.iter().map(|h| re.is_match(h)).collect();
        let dt = t0.elapsed().as_secs_f64();
        let n = haystacks.len().max(1) as f64;
        Ok((matches, dt * 1e9 / n))
    }

    /// Zero-argument round trip — isolates raw PyO3 call overhead (GIL
    /// already held, no argument marshalling at all) from the argument
    /// marshalling cost that `walk_one`'s 12 numpy-array arguments add.
    #[pyfunction]
    fn noop() -> i64 {
        1
    }

    /// The realistic production shape for single-record serving: the tree
    /// is copied into Rust-owned memory ONCE (at load/activate time, off
    /// the request path — mirrors ZEN's own object-based API, doc 08 §4's
    /// "resolve once per generation"), and each request marshals only the
    /// row, not all 12 tree arrays. Contrast this against `walk_one`
    /// (which re-marshals the whole tree every call) to see how much of
    /// `walk_one`'s per-call cost is "PyO3 argument marshalling for 12
    /// arrays" versus "the kernel itself."
    #[pyclass]
    struct PyTree {
        kind: Vec<i8>,
        feat_idx: Vec<i32>,
        op_code: Vec<i8>,
        thresh: Vec<f64>,
        pat_start: Vec<i32>,
        pat_count: Vec<i32>,
        patterns: Vec<i32>,
        left: Vec<i32>,
        right: Vec<i32>,
        leaf_value: Vec<f64>,
    }

    #[pymethods]
    impl PyTree {
        #[new]
        #[pyo3(signature = (kind, feat_idx, op_code, thresh, pat_start, pat_count,
                             patterns, left, right, leaf_value))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            kind: PyReadonlyArray1<'_, i8>,
            feat_idx: PyReadonlyArray1<'_, i32>,
            op_code: PyReadonlyArray1<'_, i8>,
            thresh: PyReadonlyArray1<'_, f64>,
            pat_start: PyReadonlyArray1<'_, i32>,
            pat_count: PyReadonlyArray1<'_, i32>,
            patterns: PyReadonlyArray1<'_, i32>,
            left: PyReadonlyArray1<'_, i32>,
            right: PyReadonlyArray1<'_, i32>,
            leaf_value: PyReadonlyArray1<'_, f64>,
        ) -> PyResult<Self> {
            Ok(Self {
                kind: kind.as_slice()?.to_vec(),
                feat_idx: feat_idx.as_slice()?.to_vec(),
                op_code: op_code.as_slice()?.to_vec(),
                thresh: thresh.as_slice()?.to_vec(),
                pat_start: pat_start.as_slice()?.to_vec(),
                pat_count: pat_count.as_slice()?.to_vec(),
                patterns: patterns.as_slice()?.to_vec(),
                left: left.as_slice()?.to_vec(),
                right: right.as_slice()?.to_vec(),
                leaf_value: leaf_value.as_slice()?.to_vec(),
            })
        }

        /// One row in, one value out — the tree itself costs nothing per
        /// call because it is already resident in `self`.
        fn walk_one(&self, row: PyReadonlyArray1<'_, f64>, str_row: PyReadonlyArray1<'_, i32>) -> PyResult<f64> {
            Ok(walk_one_inner(
                row.as_slice()?, str_row.as_slice()?,
                &self.kind, &self.feat_idx, &self.op_code, &self.thresh,
                &self.pat_start, &self.pat_count, &self.patterns,
                &self.left, &self.right, &self.leaf_value,
            ))
        }

        /// Batch entry point on the same held tree, GIL released for the
        /// walk — the production shape for a server that both batches and
        /// serves single records from one loaded tree.
        fn walk_batch<'py>(
            &self,
            py: Python<'py>,
            numeric_cols: PyReadonlyArray2<'py, f64>,
            string_cols: PyReadonlyArray2<'py, i32>,
            mut out: PyReadwriteArray1<'py, f64>,
        ) -> PyResult<()> {
            let numeric = numeric_cols.as_array();
            let strings = string_cols.as_array();
            let out_s = out.as_slice_mut()?;
            let n = numeric.shape()[0];
            let (kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns, left, right, leaf_value) =
                (&self.kind, &self.feat_idx, &self.op_code, &self.thresh, &self.pat_start,
                 &self.pat_count, &self.patterns, &self.left, &self.right, &self.leaf_value);
            py.detach(move || {
                for i in 0..n {
                    let row = numeric.row(i);
                    let str_row = strings.row(i);
                    let row_s = row.as_slice().expect("numeric row not contiguous");
                    let str_s = str_row.as_slice().expect("string row not contiguous");
                    out_s[i] = walk_one_inner(
                        row_s, str_s, kind, feat_idx, op_code, thresh,
                        pat_start, pat_count, patterns, left, right, leaf_value,
                    );
                }
            });
            Ok(())
        }
    }
}
