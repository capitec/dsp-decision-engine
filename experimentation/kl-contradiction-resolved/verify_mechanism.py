"""Step 3: verify the mechanism directly.

(a) co_code vs co_consts across the edit, for BOTH driver shapes:
    - L-shape: threshold appears exactly once (gen.py)
    - K-shape: threshold shared across branches (mirrors numba_cache_survival/gen_driver.py's "0.5" reused everywhere)
(b) the .nbi index key actually written by numba in the pyc_only_cleared arm,
    read back the same way numba_cache_survival/run_experiment.py:inspect_index does.
"""
import hashlib
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen import gen_source as gen_l  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def strip_njit(src):
    # Pure-CPython co_code/co_consts comparison -- strip the numba decorator
    # (and its import) so this doesn't need a real on-disk file / locator.
    return "\n".join(
        ln for ln in src.splitlines()
        if "@njit" not in ln and "from numba" not in ln
    )


def inspect(label, src_before, src_after):
    ns = {}
    exec(compile(strip_njit(src_before), "drv.py", "exec"), ns)
    code_before = ns["driver"].__code__
    ns2 = {}
    exec(compile(strip_njit(src_after), "drv.py", "exec"), ns2)
    code_after = ns2["driver"].__code__
    print(f"--- {label} ---")
    print(f"  co_consts before: {code_before.co_consts}")
    print(f"  co_consts after : {code_after.co_consts}")
    print(f"  co_code sha256[:16] before: {sha(code_before.co_code)}")
    print(f"  co_code sha256[:16] after : {sha(code_after.co_code)}")
    print(f"  co_code IDENTICAL: {code_before.co_code == code_after.co_code}")
    print()


# --- L-shape: single occurrence ---
inspect("L-shape (threshold appears once)",
        gen_l("0.0405"), gen_l("0.5950"))


# --- K-shape: constant shared across every branch (numba_cache_survival/gen_driver.py's pattern) ---
def gen_k_source(n_args, shared_const):
    a = [f"a{i}" for i in range(n_args)]
    out = ["from numba import njit", "", "", "@njit(cache=True)", f"def driver({', '.join(a)}):", "    acc = 0.0"]
    for i, name in enumerate(a):
        k = 1.0 + (i % 7) / 1000.0
        out.append(f"    if {name} > {shared_const}:")
        out.append(f"        acc += {name} * {k!r}")
        out.append("    else:")
        out.append(f"        acc -= {name} * {k!r} * {shared_const}")
    out.append("    return acc")
    return "\n".join(out)


inspect("K-shape (shared constant '0.5', edited to '0.7' -- same byte length)",
        gen_k_source(8, "0.5"), gen_k_source(8, "0.7"))

# --- .nbi index contents, pyc_only_cleared arm, before vs after the edit ---
print("--- .nbi index, pyc_only_cleared arm ---")
nbi_path = os.path.join(HERE, "_arm_pyc_only_cleared", "__pycache__", "drv.driver-4.py314.nbi")
if os.path.exists(nbi_path):
    with open(nbi_path, "rb") as f:
        version = pickle.load(f)
        data = f.read()
    stamp, overloads = pickle.loads(data)
    print(f"  index file: {os.path.basename(nbi_path)}")
    print(f"  index version: {version}")
    print(f"  source stamp (mtime, size) recorded in index: {stamp}")
    for key, val in overloads.items():
        sig, magic, hashes = key
        print(f"  signature: {sig}")
        print(f"  magic_tuple: {magic}")
        print(f"  code/closure hashes (this is condition 6 -- co_code + pickled closure, NOT co_consts): {hashes}")
        print(f"  -> data file: {val}")
else:
    print(f"  NOT FOUND: {nbi_path}")

# --- Reproduce K's ACTUAL edit exactly: only ONE of many occurrences of the
# shared value "0.5" is changed (the a1 else-branch multiplier), the rest of
# the 0.5 occurrences are left as 0.5 -- this is what
# numba_cache_survival/run_experiment.py:part_stale literally does, unlike
# the template-regeneration above which changed every occurrence at once. ---
print("--- K's REAL edit: only the a1 'acc -= a1 * k * 0.5' line's 0.5 -> 0.7, rest of the 16 occurrences of 0.5 untouched ---")
k_src_before = gen_k_source(8, "0.5")
lines = k_src_before.splitlines()
target = [i for i, ln in enumerate(lines) if ln.strip().startswith("acc -= a1 *")]
assert len(target) == 1
old_line = lines[target[0]]
assert old_line.endswith("* 0.5")
new_line = old_line[:-len("0.5")] + "0.7"
lines2 = list(lines)
lines2[target[0]] = new_line
k_src_after = "\n".join(lines2)
inspect("K-shape, K's real single-occurrence edit (0.5 still needed elsewhere)",
        k_src_before, k_src_after)
