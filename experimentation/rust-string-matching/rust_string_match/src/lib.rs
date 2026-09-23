//! String matching over **Arrow string buffers**, exposed through the C ABI
//! for a numba `@njit` kernel to call — no PyO3, no Python object, no
//! `Vec<String>`, ever.
//!
//! An Arrow / polars `Utf8` column is two buffers: `values: u8[]` (every
//! string's bytes, concatenated) and `offsets: i64[n + 1]`. String `i` is
//! `values[offsets[i] .. offsets[i + 1]]`. polars hands both out zero-copy
//! (`Series._get_buffers()`), so the kernel passes the two base pointers
//! plus their lengths and every match below is a slice into memory polars
//! already owns.
//!
//! **Patterns are compiled once, keyed by a small integer id** — a
//! lock-free append-only registry (`SLOTS`), so the hot path is one atomic
//! load, not a mutex. Ids are process-lifetime: there is no free, because
//! decider2 binds a pattern at compile/bind time and reuses it for every
//! batch after.
//!
//! **One entry point serves both strategies.** `match_rows` walks whatever
//! string set it is handed: pass the COLUMN's buffers and it matches every
//! row (O(rows)); pass the CATEGORY DICTIONARY's buffers (12 employers) and
//! it produces a per-category mask (O(distinct)) that the kernel then
//! indexes by dictionary code. `match_one` is the lazy, at-the-node shape:
//! one string, one call, only when the walk reaches the node.
//!
//! Regex matching is `regex::bytes::Regex` — polars strings are valid UTF-8 by
//! construction, so a per-call `from_utf8` validation pass (O(len), which
//! `../rust-cabi-in-kernel`'s `regex_is_match` paid) is pure waste here.
//!
//! **Every boundary is `catch_unwind`-wrapped and every buffer access is a
//! bounds-checked slice index**, so a hostile index or a lying length is a
//! Rust panic (caught, returned as -1) and never a read past the buffer.
//! The `_unprotected` twin exists only for the panic demo.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::null_mut;
use std::slice;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use regex::bytes::Regex;

const MAX_PATTERNS: usize = 4096;

/// decider2's `TStringMatchType`, one variant each: only `regex` needs the
/// regex crate at all — the other four are a `memcmp`/`memmem`, which is
/// the honest cost floor for a prefix test like the owner's `^dog`.
enum Matcher {
    Exact(Vec<u8>),
    StartsWith(Vec<u8>),
    Contains(memchr::memmem::Finder<'static>),
    EndsWith(Vec<u8>),
    Regex(Regex),
}

impl Matcher {
    #[inline(always)]
    fn is_match(&self, hay: &[u8]) -> bool {
        match self {
            Matcher::Exact(p) => hay == p.as_slice(),
            Matcher::StartsWith(p) => hay.starts_with(p),
            Matcher::Contains(f) => f.find(hay).is_some(),
            Matcher::EndsWith(p) => hay.ends_with(p),
            Matcher::Regex(re) => re.is_match(hay),
        }
    }
}

pub const KIND_EXACT: i32 = 0;
pub const KIND_STARTS_WITH: i32 = 1;
pub const KIND_CONTAINS: i32 = 2;
pub const KIND_ENDS_WITH: i32 = 3;
pub const KIND_REGEX: i32 = 4;

static SLOTS: [AtomicPtr<Matcher>; MAX_PATTERNS] =
    [const { AtomicPtr::new(null_mut()) }; MAX_PATTERNS];
static NEXT: AtomicUsize = AtomicUsize::new(0);

/// Compile a pattern once. `kind` is one of the `KIND_*` constants
/// (decider2's `TStringMatchType` order). Returns its id (>= 0), or -1 on an
/// invalid pattern / unknown kind / registry full / internal panic.
#[no_mangle]
pub unsafe extern "C" fn pattern_compile(kind: i32, pattern_ptr: *const u8, pattern_len: i64) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let bytes = slice::from_raw_parts(pattern_ptr, pattern_len as usize);
        let m = match kind {
            KIND_EXACT => Matcher::Exact(bytes.to_vec()),
            KIND_STARTS_WITH => Matcher::StartsWith(bytes.to_vec()),
            KIND_CONTAINS => Matcher::Contains(memchr::memmem::Finder::new(bytes).into_owned()),
            KIND_ENDS_WITH => Matcher::EndsWith(bytes.to_vec()),
            KIND_REGEX => Matcher::Regex(Regex::new(std::str::from_utf8(bytes).ok()?).ok()?),
            _ => return None,
        };
        let id = NEXT.fetch_add(1, Ordering::AcqRel);
        if id >= MAX_PATTERNS {
            return None;
        }
        SLOTS[id].store(Box::into_raw(Box::new(m)), Ordering::Release);
        Some(id as i32)
    }));
    match result {
        Ok(Some(id)) => id,
        _ => -1,
    }
}

#[inline(always)]
fn lookup(pattern_id: i32) -> Option<&'static Matcher> {
    if pattern_id < 0 || pattern_id as usize >= MAX_PATTERNS {
        return None;
    }
    let p = SLOTS[pattern_id as usize].load(Ordering::Acquire);
    if p.is_null() {
        None
    } else {
        // SAFETY: slots are write-once from `Box::into_raw` and never freed.
        Some(unsafe { &*p })
    }
}

/// Arrow string-view over `(offsets, values)`. Both are *slices* from here
/// on — every index is bounds-checked, which is what turns a bad `idx` or
/// a lying `values_len` into a catchable panic instead of UB.
struct Strings<'a> {
    offsets: &'a [i64],
    values: &'a [u8],
}

impl<'a> Strings<'a> {
    #[inline(always)]
    fn get(&self, idx: usize) -> &'a [u8] {
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        &self.values[start..end]
    }
}

#[inline(always)]
unsafe fn strings<'a>(
    offsets: *const i64, n_strings: i64, values: *const u8, values_len: i64,
) -> Strings<'a> {
    Strings {
        offsets: slice::from_raw_parts(offsets, (n_strings as usize) + 1),
        values: slice::from_raw_parts(values, values_len as usize),
    }
}

#[inline(always)]
fn match_one_inner(re: &Matcher, s: &Strings, idx: i64) -> i32 {
    if re.is_match(s.get(idx as usize)) { 1 } else { 0 }
}

// ---------------------------------------------------------------------------
// match_one — the lazy shape. Called from INSIDE the kernel, at the node,
// only for rows whose walk actually reaches it.
// ---------------------------------------------------------------------------

/// Returns 1 (match), 0 (no match), or -1 (unknown pattern id, index out
/// of range, or any caught panic).
#[no_mangle]
pub unsafe extern "C" fn match_one(
    pattern_id: i32,
    offsets: *const i64, n_strings: i64,
    values: *const u8, values_len: i64,
    idx: i64,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let re = lookup(pattern_id)?;
        let s = strings(offsets, n_strings, values, values_len);
        Some(match_one_inner(re, &s, idx))
    }));
    match result {
        Ok(Some(v)) => v,
        _ => -1,
    }
}

/// Same body, no `catch_unwind`. PANIC DEMO ONLY — never benchmarked.
#[no_mangle]
pub unsafe extern "C" fn match_one_unprotected(
    pattern_id: i32,
    offsets: *const i64, n_strings: i64,
    values: *const u8, values_len: i64,
    idx: i64,
) -> i32 {
    let re = lookup(pattern_id).expect("unknown pattern id");
    let s = strings(offsets, n_strings, values, values_len);
    match_one_inner(re, &s, idx)
}

// ---------------------------------------------------------------------------
// match_rows — the batch shape. ONE function, two strategies:
//   * hand it the COLUMN   (n_strings = rows, row_codes = null, n = rows)
//     -> out_mask[i] = match(row i)                   O(rows)
//   * hand it the DICTIONARY (n_strings = distinct, row_codes = null,
//     n = distinct) -> out_mask[c] = match(category c) O(distinct); the
//     kernel then does mask[code[i]].
//   * or hand it the DICTIONARY plus row_codes (n = rows)
//     -> out_mask[i] = match(category row_codes[i])   O(rows), no mask.
// ---------------------------------------------------------------------------

/// Returns 0 on success, -1 on unknown pattern id / out-of-range index /
/// caught panic. `out_mask` is written 0/1 per entry.
#[no_mangle]
pub unsafe extern "C" fn match_rows(
    pattern_id: i32,
    offsets: *const i64, n_strings: i64,
    values: *const u8, values_len: i64,
    row_codes: *const i32,
    n: i64,
    out_mask: *mut u8,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let re = lookup(pattern_id)?;
        let s = strings(offsets, n_strings, values, values_len);
        let out = slice::from_raw_parts_mut(out_mask, n as usize);
        if row_codes.is_null() {
            for (i, o) in out.iter_mut().enumerate() {
                *o = match_one_inner(re, &s, i as i64) as u8;
            }
        } else {
            let codes = slice::from_raw_parts(row_codes, n as usize);
            for (o, &c) in out.iter_mut().zip(codes) {
                *o = match_one_inner(re, &s, c as i64) as u8;
            }
        }
        Some(())
    }));
    match result {
        Ok(Some(())) => 0,
        _ => -1,
    }
}

// ---------------------------------------------------------------------------
// Call-overhead baseline and a synthetic panic pair, for the same reasons
// ../rust-cabi-in-kernel has them.
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn trivial_i32(a: i32, b: i32) -> i32 {
    a + b
}

#[no_mangle]
pub extern "C" fn deliberately_panic_protected(bad: i32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if bad != 0 {
            panic!("deliberate panic (protected boundary)");
        }
        42
    }))
    .unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn deliberately_panic_unprotected(bad: i32) -> i32 {
    if bad != 0 {
        panic!("deliberate panic (unprotected boundary)");
    }
    42
}
