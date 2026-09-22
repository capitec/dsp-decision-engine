"""Does decider2's OWN tree shape pay per-row NRT refcount traffic?

decider2/src/decider2/trees/encode.py builds a per-row closure `path_fn` that
captures the 7 structure arrays as freevars and passes them as ARGUMENTS to the
separately-compiled njit `walk_tree` -- once per row. The numba+C strand's
ablation found that shape costs ~365 ns/row in ITS prototype. This probe asks
the same question of decider2's real `walk_tree`, unmodified.

A: driver loop -> path_fn(args_tuple, params) -> walk_tree(..., 7 arrays, ...)   [decider2 today]
B: same walk body inlined in the driver loop, arrays captured, no per-row call.
"""
import re, sys, time
import numpy as np
sys.path.insert(0, "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/src")
from numba import njit
from decider2.trees.interpreter import walk_tree, compare, LEAF, CMP, IS_TRUE, IS_FALSE

N = 1_000_000
rng = np.random.default_rng(7)
x0 = rng.random(N); x1 = rng.random(N); x2 = rng.random(N)

# depth-3 full binary tree over 3 features: 7 internal + 8 leaves = 15 nodes
kind      = np.array([CMP,CMP,CMP,CMP,CMP,CMP,CMP, LEAF,LEAF,LEAF,LEAF,LEAF,LEAF,LEAF,LEAF], dtype=np.int32)
feat_idx  = np.array([0,1,1,2,2,2,2, 0,0,0,0,0,0,0,0], dtype=np.int32)
op        = np.array([3,3,3,3,3,3,3, 0,0,0,0,0,0,0,0], dtype=np.int32)   # GT
thr_slot  = np.array([0,1,1,2,2,2,2, 0,0,0,0,0,0,0,0], dtype=np.int32)
then_     = np.array([1,3,5,7,9,11,13, 0,0,0,0,0,0,0,0], dtype=np.int32)
else_     = np.array([2,4,6,8,10,12,14, 0,0,0,0,0,0,0,0], dtype=np.int32)
leaf_value= np.array([0,0,0,0,0,0,0, 1,2,3,4,5,6,7,8], dtype=np.int64)
start_pc = 0
literals_t = (0.5, 0.5, 0.5)

# --- A: decider2's real shape, verbatim from encode.py's n_computed == 0 branch
@njit(cache=False)
def path_fn(args, params):
    thresholds = params + literals_t
    return walk_tree(args, thresholds, kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc)

@njit(cache=False)
def driver_A(a0, a1, a2, out):
    for i in range(out.shape[0]):
        out[i] = path_fn((a0[i], a1[i], a2[i]), ())

# --- B: same logic, one loop, no per-row call, arrays captured
@njit(cache=False)
def driver_B(a0, a1, a2, out):
    for i in range(out.shape[0]):
        feats = (a0[i], a1[i], a2[i])
        pc = start_pc
        while True:
            k = kind[pc]
            if k == LEAF:
                out[i] = leaf_value[pc]; break
            elif k == CMP:
                a = feats[feat_idx[pc]]
                b = literals_t[thr_slot[pc]]
                pc = then_[pc] if compare(op[pc], a, b) else else_[pc]
            elif k == IS_TRUE:
                pc = then_[pc] if feats[feat_idx[pc]] != 0.0 else else_[pc]
            else:
                pc = then_[pc] if feats[feat_idx[pc]] == 0.0 else else_[pc]

def bench(fn, reps=5):
    out = np.zeros(N, dtype=np.int64)
    fn(x0, x1, x2, out)                       # compile
    best = min(_time(fn, out) for _ in range(reps))
    return best / N * 1e9, out.copy()

def _time(fn, out):
    t = time.perf_counter(); fn(x0, x1, x2, out); return time.perf_counter() - t

def nrt_counts(dispatcher):
    llvm = "\n".join(dispatcher.inspect_llvm().values())
    return len(re.findall(r"call .*@NRT_incref", llvm)), len(re.findall(r"call .*@NRT_decref", llvm))

ns_a, out_a = bench(driver_A)
ns_b, out_b = bench(driver_B)
assert (out_a == out_b).all(), "answers differ"
print(f"A  decider2 shape (per-row call into walk_tree, 7 array args) : {ns_a:8.2f} ns/row   NRT incref/decref in driver: {nrt_counts(driver_A)}")
print(f"B  same body inlined in one loop, arrays captured             : {ns_b:8.2f} ns/row   NRT incref/decref in driver: {nrt_counts(driver_B)}")
print(f"ratio A/B = {ns_a/ns_b:.2f}x")
print(f"path_fn itself NRT incref/decref: {nrt_counts(path_fn)}")
print(f"walk_tree      NRT incref/decref: {nrt_counts(walk_tree)}")
