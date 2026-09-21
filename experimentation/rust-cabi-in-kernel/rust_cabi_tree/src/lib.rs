//! A Rust `cdylib` exposed through the C ABI, meant to be `ctypes.CDLL`-loaded
//! and called **from inside** a numba `@njit` kernel — not through PyO3.
//!
//! No `#[pymodule]`, no `pyo3`, no `numpy` crate: this .so carries no Python
//! ABI at all (see RESULTS.md's packaging section). Every entry point below
//! is `#[no_mangle] pub unsafe extern "C" fn`, taking raw pointers and
//! lengths — numba's ctypes bridge rejects `c_char_p` and accepts
//! `c_void_p`, so the Python side declares every pointer argument as
//! `ctypes.c_void_p` and every array's length crosses as a separate `i32`/
//! `i64` argument (see `tree_walk.py`).
//!
//! Node encoding mirrors, field-for-field,
//! `tree-codegen-vs-interpreted/interpreted_kernel.py`'s `walk_batch` and
//! `rust-tree-interpreter/rust_tree_interpreter/src/lib.rs`'s
//! `walk_one_inner` — same struct-of-arrays layout, same OR-of-exact-
//! dictionary-code string semantics. `walk_one_inner` itself is lifted
//! unchanged from that crate; only the binding around it is new.

use std::os::raw::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use regex::Regex;

const KIND_LEAF: i8 = 0;
const KIND_NUM: i8 = 1;
// KIND_STR = 2 is implicit: any non-leaf, non-numeric node is a string
// match node (matches interpreted_kernel.py's `else` branch).

/// The kernel. Lifted unchanged from `rust-tree-interpreter`'s
/// `walk_one_inner`: no recursion, no heap allocation, no Python object.
/// Every array access below is a *slice* index (`kind[node]`, not
/// `*kind_ptr.add(node)`), which is load-bearing: slice indexing is
/// bounds-checked and panics (a controlled Rust failure `catch_unwind` can
/// catch) rather than reading out of bounds (real UB `catch_unwind` cannot
/// help with at all). See the panic-safety section of RESULTS.md.
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

/// Build the eleven tree-array slices shared by every walk entry point
/// below, from raw pointers + explicit lengths. `unsafe`: the caller (numba,
/// via ctypes) must pass pointers that really are `n_nodes`/`n_patterns`
/// long — this is the one place that trust is extended; every access past
/// this point is bounds-checked slice indexing.
#[allow(clippy::too_many_arguments)]
unsafe fn tree_slices<'a>(
    kind: *const i8,
    feat_idx: *const i32,
    op_code: *const i8,
    thresh: *const f64,
    pat_start: *const i32,
    pat_count: *const i32,
    patterns: *const i32,
    n_patterns: i32,
    left: *const i32,
    right: *const i32,
    leaf_value: *const f64,
    n_nodes: i32,
) -> (
    &'a [i8], &'a [i32], &'a [i8], &'a [f64], &'a [i32], &'a [i32],
    &'a [i32], &'a [i32], &'a [i32], &'a [f64],
) {
    let n = n_nodes as usize;
    let np = n_patterns as usize;
    (
        slice::from_raw_parts(kind, n),
        slice::from_raw_parts(feat_idx, n),
        slice::from_raw_parts(op_code, n),
        slice::from_raw_parts(thresh, n),
        slice::from_raw_parts(pat_start, n),
        slice::from_raw_parts(pat_count, n),
        slice::from_raw_parts(patterns, np),
        slice::from_raw_parts(left, n),
        slice::from_raw_parts(right, n),
        slice::from_raw_parts(leaf_value, n),
    )
}

// ---------------------------------------------------------------------------
// Per-row calling shape — one `extern "C"` call per row. This is the shape
// that preserves fusion: called from inside an njit row loop, so the fused
// kernel makes the call itself rather than numba handing control back to
// Python between rows.
// ---------------------------------------------------------------------------

/// Production entry point: `catch_unwind`-protected. A panic anywhere in
/// the walk (deliberately, or from a corrupted tree — see RESULTS.md) is
/// caught here and turned into `f64::NAN` plus a clean return, never an
/// unwind crossing into the numba-JIT frame that called this.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn walk_row(
    kind: *const i8, feat_idx: *const i32, op_code: *const i8, thresh: *const f64,
    pat_start: *const i32, pat_count: *const i32, patterns: *const i32, n_patterns: i32,
    left: *const i32, right: *const i32, leaf_value: *const f64, n_nodes: i32,
    row: *const f64, n_numeric: i32,
    str_row: *const i32, n_string: i32,
) -> f64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns, left, right, leaf_value) =
            tree_slices(kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns, n_patterns,
                        left, right, leaf_value, n_nodes);
        let row = slice::from_raw_parts(row, n_numeric as usize);
        let str_row = slice::from_raw_parts(str_row, n_string as usize);
        walk_one_inner(row, str_row, kind, feat_idx, op_code, thresh, pat_start, pat_count,
                       patterns, left, right, leaf_value)
    }));
    result.unwrap_or(f64::NAN)
}

/// Same walk, with **no** `catch_unwind` — used only by the panic-safety
/// experiment to show the contrast against `walk_row` above. Never called
/// from the performance benchmarks.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn walk_row_unprotected(
    kind: *const i8, feat_idx: *const i32, op_code: *const i8, thresh: *const f64,
    pat_start: *const i32, pat_count: *const i32, patterns: *const i32, n_patterns: i32,
    left: *const i32, right: *const i32, leaf_value: *const f64, n_nodes: i32,
    row: *const f64, n_numeric: i32,
    str_row: *const i32, n_string: i32,
) -> f64 {
    let (kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns, left, right, leaf_value) =
        tree_slices(kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns, n_patterns,
                    left, right, leaf_value, n_nodes);
    let row = slice::from_raw_parts(row, n_numeric as usize);
    let str_row = slice::from_raw_parts(str_row, n_string as usize);
    walk_one_inner(row, str_row, kind, feat_idx, op_code, thresh, pat_start, pat_count,
                   patterns, left, right, leaf_value)
}

// ---------------------------------------------------------------------------
// Per-batch calling shape — one `extern "C"` call for the whole batch, loop
// inside Rust. Faster (amortises the call), but reintroduces a boundary:
// nothing between rows is numba-compiled or fusable with a neighbouring
// step, exactly the PyO3 `walk_batch` shape §U already measured, reproduced
// here for the C-ABI binding instead.
// ---------------------------------------------------------------------------

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn walk_batch(
    kind: *const i8, feat_idx: *const i32, op_code: *const i8, thresh: *const f64,
    pat_start: *const i32, pat_count: *const i32, patterns: *const i32, n_patterns: i32,
    left: *const i32, right: *const i32, leaf_value: *const f64, n_nodes: i32,
    numeric: *const f64, n_numeric: i32,
    strings: *const i32, n_string: i32,
    n_rows: i64,
    out: *mut f64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns, left, right, leaf_value) =
            tree_slices(kind, feat_idx, op_code, thresh, pat_start, pat_count, patterns, n_patterns,
                        left, right, leaf_value, n_nodes);
        let n = n_rows as usize;
        let nnum = n_numeric as usize;
        let nstr = n_string as usize;
        let out = slice::from_raw_parts_mut(out, n);
        for i in 0..n {
            let row = slice::from_raw_parts(numeric.add(i * nnum), nnum);
            let str_row = slice::from_raw_parts(strings.add(i * nstr), nstr);
            out[i] = walk_one_inner(row, str_row, kind, feat_idx, op_code, thresh, pat_start,
                                    pat_count, patterns, left, right, leaf_value);
        }
    }));
    match result {
        Ok(()) => 0,
        Err(_) => 1,
    }
}

// ---------------------------------------------------------------------------
// Call-overhead baselines (mirrors t.c's `trivial`/probe, in real Rust).
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn noop() -> i64 {
    1
}

#[no_mangle]
pub extern "C" fn trivial(a: i32, b: i32) -> i32 {
    a + b
}

// ---------------------------------------------------------------------------
// Regex — compile-once, call-many via an opaque handle. This is the
// realistic per-row shape: the pattern is compiled ONCE (off the hot path,
// mirrors decider2's own boundary-time dictionary-encoding contract, §T),
// and every subsequent call crosses only a pointer + length for the text —
// no `Vec<String>` marshalling (§U's naive PyO3 binding cost, ~79 ns/call
// on top of the 25.6 ns/call pure match) and no Python object at all.
// ---------------------------------------------------------------------------

/// Returns a raw, leaked `Box<Regex>` pointer, or null on an invalid
/// pattern or an internal panic (caught here — `regex::Error` is an
/// ordinary `Result`, not a panic path, but the boundary is protected
/// uniformly regardless).
#[no_mangle]
pub unsafe extern "C" fn regex_compile(pattern_ptr: *const u8, pattern_len: i32) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let bytes = slice::from_raw_parts(pattern_ptr, pattern_len as usize);
        let pattern = std::str::from_utf8(bytes).ok()?;
        Regex::new(pattern).ok()
    }));
    match result {
        Ok(Some(re)) => Box::into_raw(Box::new(re)) as *mut c_void,
        _ => std::ptr::null_mut(),
    }
}

/// One call per text. `catch_unwind`-protected; returns -1 on a null
/// handle or a caught panic, else 0/1.
#[no_mangle]
pub unsafe extern "C" fn regex_is_match(handle: *mut c_void, text_ptr: *const u8, text_len: i32) -> i32 {
    if handle.is_null() {
        return -1;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let re = &*(handle as *const Regex);
        let bytes = slice::from_raw_parts(text_ptr, text_len as usize);
        match std::str::from_utf8(bytes) {
            Ok(s) => re.is_match(s),
            Err(_) => false,
        }
    }));
    match result {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn regex_free(handle: *mut c_void) {
    if !handle.is_null() {
        drop(Box::from_raw(handle as *mut Regex));
    }
}

// ---------------------------------------------------------------------------
// Panic safety demonstration. `bad != 0` triggers a deliberate, ordinary
// Rust panic (`panic!()`, a controlled failure — not the tree-corruption
// UB case, which is demonstrated separately via `walk_row`/
// `walk_row_unprotected` with a deliberately out-of-range `left[]` entry;
// see RESULTS.md and panic_demo.py).
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn deliberately_panic_protected(bad: i32) -> f64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if bad != 0 {
            panic!("deliberate panic for FFI safety demo (protected boundary)");
        }
        42.0
    }));
    result.unwrap_or(f64::NAN)
}

#[no_mangle]
pub extern "C" fn deliberately_panic_unprotected(bad: i32) -> f64 {
    if bad != 0 {
        panic!("deliberate panic for FFI safety demo (unprotected boundary)");
    }
    42.0
}
