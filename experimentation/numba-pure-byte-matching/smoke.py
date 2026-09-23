import time, numpy as np, polars as pl
import matchers as M

def py_match(kind, h, nd):
    if h is None: return False
    hb = h.encode(); nb = nd.encode()
    if kind == M.EXACT: return hb == nb
    if kind in (M.PREFIX, M.PREFIX_W): return hb.startswith(nb)
    if kind == M.SUFFIX: return hb.endswith(nb)
    if kind in (M.SUBSTRING, M.SUBSTRING_SKIP): return nb in hb
    if kind == M.PREFIX_CI: return hb.lower().startswith(nb.lower())

s = pl.Series("s", ["dog-savings-acct", "Doge-credit", "", None, "hotdog", "d", "dögg-account", "dogma-0001234", "xdog"])
b = s._get_buffers()
print("buffers keys:", list(b.keys()), "validity:", b.get("validity"))
offs, vals, valid = M.string_buffers(s)
print("valid", valid, "offs", offs, "nbytes", len(vals))
tab = M.buffer_table(offs, vals, valid)
n = len(s); out = np.zeros(n, np.uint8)
for nd in ["dog", "", "dögg", "acct", "dog-savings-acct", "a-very-long-needle-longer-than-any-haystack", "4"]:
    for kind in range(7):
        nt = M.NeedleTable([(kind, nd)])
        M.col_ptr(kind, tab, nt.addr_table(0), out)
        exp = np.array([py_match(kind, h, nd) for h in s.to_list()], np.uint8)
        assert (out == exp).all(), (M.KIND_NAMES[kind], nd, out, exp)
        M.col_arr(kind, offs, vals, valid, nt.buf, nt.start[0], nt.length[0], out)
        assert (out == exp).all(), ("arr", M.KIND_NAMES[kind], nd, out, exp)
print("small-set oracle OK (null rows -> no match, since a null row has zero length in offsets)")

# extrapolate: 100k rows
rng = np.random.default_rng(0)
words = ["dog-savings-acct", "doge-credit-card", "cat-home-loan-x", "dogma-vehicle-fin", "hotdog-overdraft", "bird-fixed-deposit",
         "fish-notice-acct", "dogs-cheque-acct", "cow-flexi-savings", "dodo-tax-free-sav", "dig-global-one-x", "dögg-money-market"]
s = pl.Series("s", np.array(words)[rng.integers(0, 12, 100_000)])
offs, vals, valid = M.string_buffers(s); tab = M.buffer_table(offs, vals, valid); out = np.zeros(len(s), np.uint8)
nt = M.NeedleTable([(M.PREFIX, "dog")])
for kind in (M.PREFIX, M.SUBSTRING):
    M.col_ptr(kind, tab, nt.addr_table(0), out)
    t0 = time.perf_counter(); M.col_ptr(kind, tab, nt.addr_table(0), out); t = time.perf_counter() - t0
    print(M.KIND_NAMES[kind], "ptr", f"{t/len(s)*1e9:.1f} ns/row  -> 1M rows ~ {t*10*1e3:.1f} ms")
