"""Per-call cost of each matcher, called from njit through the pointer
table, over Arrow (offsets, values) buffers -- the number that goes on
§T/§U's table next to libc 60.3 / polars 42.2 / Rust regex 25.6.

Also: the call-overhead probe (§V's shape, (int32,int32)->int32) for the
argument-pointer mechanism vs the global-capture mechanism vs no call.
"""
import re, time, sys
import numpy as np
import polars as pl
from numba import njit
import strmatch as SM
import kernels as K
from call_ptr import call_match, call_i32_i32
from results_io import record

REPEATS = 7

def best(fn, *a, repeats=REPEATS):
    fn(*a); t = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(*a); t.append(time.perf_counter() - t0)
    return min(t)

@njit(cache=True)
def match_all(fn_table, handle, offs, vals, out):
    base = np.uint64(vals.ctypes.data)
    for i in range(offs.shape[0] - 1):
        s = offs[i]; e = offs[i + 1]
        out[i] = call_match(fn_table[0], handle, base + np.uint64(s), e - s)

@njit(cache=True)
def sum_lengths(offs, out):
    """Same loop, no call: the loop's own floor."""
    for i in range(offs.shape[0] - 1):
        out[i] = np.int32(offs[i + 1] - offs[i])

def make_strings(n, cardinality, rng):
    base = ["dog", "doge", "cat", "dogma", "hotdog", "bird", "fish", "dogs", "cow", "dodo", "dig", "dögg"]
    if cardinality == "low":
        return pl.Series("s", np.array(base)[rng.integers(0, 12, n)])
    return pl.Series("s", [f"{base[i % 12]}-{i:07d}" for i in range(n)])  # ~unique per row

def main():
    n = 200_000
    rng = np.random.default_rng(1)
    patterns = {
        "^dog": [("pcre2_jit_utf", SM.REGEX, SM.FLAG_UTF), ("pcre2_jit_bytes", SM.REGEX, 0),
                 ("pcre2_nojit", SM.REGEX, SM.FLAG_UTF | SM.FLAG_NO_JIT),
                 ("posix_regexec", SM.POSIX, 0), ("literal_prefix_memcmp", SM.PREFIX, 0)],
        r"^d[o0]g\w*(ma|e|s)$": [("pcre2_jit_utf", SM.REGEX, SM.FLAG_UTF), ("pcre2_jit_bytes", SM.REGEX, 0),
                                  ("pcre2_nojit", SM.REGEX, SM.FLAG_UTF | SM.FLAG_NO_JIT), ("posix_regexec", SM.POSIX, 0)],
        "dog": [("pcre2_jit_utf", SM.REGEX, SM.FLAG_UTF), ("literal_substring_memmem", SM.SUBSTRING, 0)],
    }
    fn_table = np.array([SM.MATCH_ADDR], np.uint64)
    for card in ("low", "high"):
        s = make_strings(n, card, rng)
        offs, vals = K.string_buffers(s)
        out = np.zeros(n, np.int32)
        floor = best(sum_lengths, offs, out)
        record("regex_percall", pattern="(loop floor, no call)", matcher="none", cardinality=card, n=n,
               ns_per_call=floor / n * 1e9)
        py_list = s.to_list()
        for pat, matchers in patterns.items():
            oracle = np.array([1 if re.search(pat, x) else 0 for x in py_list], np.int32)
            # polars, same pattern, same column
            t = best(lambda: s.str.contains(pat))
            got = s.str.contains(pat).to_numpy().astype(np.int32)
            assert (got == oracle).all(), "polars disagrees with re"
            record("regex_percall", pattern=pat, matcher="polars_str_contains", cardinality=card, n=n,
                   ns_per_call=t / n * 1e9, agrees=True)
            for name, kind, flags in matchers:
                # POSIX ERE has no \w ; translate for that control only
                p = pat.replace(r"\w", "[[:alnum:]_]") if kind == SM.POSIX else pat
                if kind in (SM.PREFIX, SM.SUBSTRING):
                    p = pat.lstrip("^")
                h = SM.compile_pattern(p, kind, flags)
                t = best(match_all, fn_table, np.int64(h), offs, vals, out)
                agrees = bool((out == oracle).all())
                assert agrees, (name, pat, card)
                record("regex_percall", pattern=pat, matcher=name, cardinality=card, n=n,
                       ns_per_call=t / n * 1e9, ns_per_call_minus_loop=(t - floor) / n * 1e9,
                       jit=SM.is_jit(h), agrees=agrees)
                if name == "pcre2_jit_utf" and pat == "^dog":
                    # registry indirection cost: same handle via raw pointer
                    raw_tbl = np.array([SM.MATCH_RAW_ADDR], np.uint64)
                    t_raw = best(match_all, raw_tbl, np.uint64(SM.handle_ptr(h)), offs, vals, out)
                    assert (out == oracle).all()
                    record("regex_percall", pattern=pat, matcher="pcre2_jit_utf_RAW_POINTER(no registry check)", cardinality=card, n=n,
                           ns_per_call=t_raw / n * 1e9, ns_per_call_minus_loop=(t_raw - floor) / n * 1e9, jit=True, agrees=True)
                SM.free_pattern(h)
        # python re loop, small n, for scale
        m = 20_000
        t0 = time.perf_counter(); [re.search("^dog", x) for x in py_list[:m]]; t = time.perf_counter() - t0
        record("regex_percall", pattern="^dog", matcher="python_re_loop", cardinality=card, n=m, ns_per_call=t / m * 1e9)

    # ---- call overhead probe: §V's (int32,int32)->int32 shape ----
    @njit(cache=True)
    def via_table(fn_table, reps):
        acc = np.int32(0)
        for i in range(reps):
            acc += call_i32_i32(fn_table[0], np.int32(i), np.int32(1))
        return acc
    triv = SM.sm_trivial_c
    @njit(cache=False)
    def via_global(reps):
        acc = np.int32(0)
        for i in range(reps):
            acc += triv(np.int32(i), np.int32(1))
        return acc
    @njit(cache=True)
    def no_call(reps):
        acc = np.int32(0)
        for i in range(reps):
            acc += np.int32(i) + np.int32(1)
        return acc
    reps = 20_000_000
    tbl = np.array([SM.TRIVIAL_ADDR], np.uint64)
    assert via_table(tbl, 1000) == via_global(1000) == no_call(1000)
    t_tab = best(via_table, tbl, reps); t_glob = best(via_global, reps); t_none = best(no_call, reps)
    record("call_overhead", mechanism="pointer-table argument (@intrinsic inttoptr+call)", ns_per_call=(t_tab - t_none) / reps * 1e9)
    record("call_overhead", mechanism="ctypes symbol captured as global (§V mechanism, uncacheable)", ns_per_call=(t_glob - t_none) / reps * 1e9)
    record("call_overhead", mechanism="no call (loop floor)", ns_per_call=t_none / reps * 1e9)

if __name__ == "__main__":
    main()
