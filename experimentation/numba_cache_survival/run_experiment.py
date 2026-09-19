#!/usr/bin/env python
"""EXPERIMENT C -- numba's on-disk cache, and what actually invalidates it.

Tests doc 05 sec 4.1 ("real files, never exec"), doc 05 sec 4.2 ("byte-identical
is the requirement"), doc 05 sec 8 (NUMBA_CPU_NAME=generic), and REVIEW.md sec 7
("byte-identical is necessary but not sufficient -- the cache stamps entries with
(st_mtime, st_size) and indexes by path").

Every scenario runs in a FRESH PROCESS (child.py). numba's in-memory dispatcher
cache would otherwise hide every on-disk effect.

Run:
    .venv/bin/python experimentation/numba_cache_survival/run_experiment.py
    .venv/bin/python experimentation/numba_cache_survival/run_experiment.py --scaling
    .venv/bin/python experimentation/numba_cache_survival/run_experiment.py --n-args 400
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "child.py")
PYTHON = sys.executable

sys.path.insert(0, HERE)
from gen_driver import gen_source  # noqa: E402

RESULTS = []


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_driver(path, n_args, comment="decider2 generated driver"):
    src = gen_source(n_args, comment)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(src)
    return sha256_file(path)


def run_child(py_path, n_args, env_extra=None, register=True, reps=5,
              modname="decider2_gen_driver", expect_ok=True):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cfg = json.dumps({
        "py_path": py_path, "n_args": n_args, "reps": reps,
        "register_in_sys_modules": register, "modname": modname,
    })
    t0 = time.perf_counter()
    proc = subprocess.run([PYTHON, CHILD, cfg], capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    line = ""
    for ln in proc.stdout.splitlines():
        if ln.startswith("{"):
            line = ln
    if not line:
        return {"ok": False, "error": "no json from child",
                "stderr_tail": proc.stderr[-2000:], "process_wall_s": wall}
    out = json.loads(line)
    out["process_wall_s"] = wall
    out["returncode"] = proc.returncode
    if not out.get("ok"):
        out["stderr_tail"] = proc.stderr[-3000:]
    return out


def record(name, question, res, sha=None, extra=None):
    row = {
        "scenario": name,
        "question": question,
        "ok": res.get("ok"),
        "hits": res.get("n_hits"),
        "misses": res.get("n_misses"),
        "first_call_s": res.get("t_first_call_s"),
        "process_wall_s": res.get("process_wall_s"),
        "source_sha256": sha,
        "source_stamp": res.get("source_stamp"),
        "index_path": res.get("index_path"),
        "filename_base": res.get("filename_base"),
        "locator": res.get("locator_class"),
        "cpu_name": res.get("numba_cpu_name"),
        "error": res.get("error"),
    }
    if extra:
        row.update(extra)
    RESULTS.append(row)
    verdict = "ERROR" if not res.get("ok") else (
        "HIT " if res.get("n_hits") else "MISS")
    fc = res.get("t_first_call_s")
    print(f"  {name:<34} {verdict}  first_call={fc if fc is None else f'{fc:8.3f}s'}"
          f"  hits={row['hits']} misses={row['misses']}")
    if not res.get("ok"):
        err = (res.get("error") or "").splitlines()
        print(f"      error: {err[0] if err else res.get('stderr_tail', '')[:200]}")
    return row


# ---------------------------------------------------------------- part 1
def part1_exec(workdir):
    """Does njit(cache=True) on an exec'd function raise the documented error?"""
    print("\n[1] exec'd function with cache=True")
    script = textwrap.dedent("""
        import json, traceback
        from numba import njit
        src = "def f(x):\\n    return x * 2.0\\n"
        ns = {}
        exec(compile(src, "<string>", "exec"), ns)
        out = {"stage": None, "exc_type": None, "message": None}
        try:
            out["stage"] = "decorate"
            g = njit(cache=True)(ns["f"])
            out["stage"] = "call"
            g(1.0)
            out["stage"] = "no error"
        except Exception as exc:
            out["exc_type"] = type(exc).__name__
            out["message"] = str(exc)
        print("RESULT" + json.dumps(out))
    """)
    p = os.path.join(workdir, "exec_probe.py")
    with open(p, "w") as f:
        f.write(script)
    proc = subprocess.run([PYTHON, p], capture_output=True, text=True)
    data = None
    for ln in proc.stdout.splitlines():
        if ln.startswith("RESULT"):
            data = json.loads(ln[len("RESULT"):])
    print(f"  raised at stage : {data and data['stage']}")
    print(f"  exception type  : {data and data['exc_type']}")
    print(f"  message         : {data and data['message']!r}")
    RESULTS.append({"scenario": "exec_cache_true", "exec_probe": data})
    return data


# ---------------------------------------------------------------- part 4
def inspect_index(index_path):
    print("\n[4] what is in the cache index")
    d = os.path.dirname(index_path)
    print(f"  cache dir: {d}")
    for fn in sorted(os.listdir(d)):
        print(f"    {os.path.getsize(os.path.join(d, fn)):>9d}  {fn}")
    with open(index_path, "rb") as f:
        version = pickle.load(f)
        data = f.read()
    stamp, overloads = pickle.loads(data)
    print(f"  index version : {version}")
    print(f"  source stamp  : {stamp}   <- (st_mtime, st_size) of the .py")
    print(f"  entries       : {len(overloads)}")
    for key, val in overloads.items():
        sig, magic, hashes = key
        print(f"    signature   : {sig}")
        print(f"    magic_tuple : {magic}")
        print(f"    code hashes : {hashes}")
        print(f"    -> data file: {val}")
    RESULTS.append({
        "scenario": "index_contents",
        "index_path": index_path,
        "index_version": version,
        "source_stamp": list(stamp),
        "n_entries": len(overloads),
        "entry_keys": [[str(k[0]), list(k[1]), list(k[2])] for k in overloads],
        "entry_values": [str(v) for v in overloads.values()],
        "files": sorted(os.listdir(d)),
    })
    return stamp, overloads


# ---------------------------------------------------------------- stale check
def part_stale(workdir, n, reps):
    """Change one CONSTANT, keep (st_mtime, st_size) identical, and see whether
    numba serves the previously compiled machine code for the new source.

    The edited term must actually affect the result, so it is picked on an
    argument whose value takes the else-branch with a non-zero operand:
    child.py passes args[i] = (i % 3) / 2.0, so a1 == 0.5 and 0.5 > 0.5 is False.
    """
    M = os.path.join(workdir, "build_m")
    shutil.rmtree(M, ignore_errors=True)
    pm = os.path.join(M, "drv.py")
    sha_m0 = write_driver(pm, n)
    m_cold = run_child(pm, n, reps=reps)
    record("M1 cold", "baseline for the stale-constant test", m_cold, sha_m0)

    st = os.stat(pm)
    keep = (st.st_atime, st.st_mtime)
    with open(pm) as f:
        body = f.read()
    target = [ln for ln in body.splitlines() if ln.strip().startswith("acc -= a1 *")]
    assert len(target) == 1, target
    old_line = target[0]
    assert old_line.endswith("* 0.5"), old_line
    new_line = old_line[: -len("0.5")] + "0.7"      # same byte length
    body2 = body.replace(old_line, new_line, 1)
    with open(pm, "w") as f:
        f.write(body2)
    os.utime(pm, keep)
    assert os.stat(pm).st_size == st.st_size, "edit changed the file size"

    m_res = run_child(pm, n, reps=reps)
    shutil.rmtree(os.path.join(M, "__pycache__"), ignore_errors=True)
    m_truth = run_child(pm, n, reps=reps)
    edit_matters = m_truth.get("value") != m_cold.get("value")
    stale = edit_matters and m_res.get("value") == m_cold.get("value")
    record("M2 constant changed, stamp identical",
           "can a forged stamp serve STALE machine code?",
           m_res, sha256_file(pm),
           {"identical_bytes": False,
            "edited_line_from": old_line.strip(),
            "edited_line_to": new_line.strip(),
            "value_before_edit": m_cold.get("value"),
            "value_served": m_res.get("value"),
            "value_after_cache_wipe": m_truth.get("value"),
            "edit_changes_the_result": bool(edit_matters),
            "served_stale_code": bool(stale)})
    print(f"      value before edit = {m_cold.get('value')!r}")
    print(f"      value served      = {m_res.get('value')!r}")
    print(f"      value after wipe  = {m_truth.get('value')!r}")
    print(f"      edit changes result: {edit_matters}   SERVED STALE CODE: {stale}")


# ---------------------------------------------------------------- scaling
def scaling(workdir, sizes, reps):
    print("\n[scaling] cold compile vs cached load, by argument count")
    rows = []
    for n in sizes:
        d = os.path.join(workdir, f"scale_{n}")
        shutil.rmtree(d, ignore_errors=True)
        p = os.path.join(d, "drv.py")
        write_driver(p, n)
        cold = run_child(p, n, reps=reps)
        warms = [run_child(p, n, reps=reps) for _ in range(3)]
        warm_med = statistics.median(w["t_first_call_s"] for w in warms)
        rows.append({
            "n_args": n,
            "cold_s": cold["t_first_call_s"],
            "warm_median_s": warm_med,
            "warm_samples_s": [w["t_first_call_s"] for w in warms],
            "speedup": cold["t_first_call_s"] / warm_med,
            "nbc_bytes": os.path.getsize(cold["index_path"].replace(".nbi", ".1.nbc")),
            "run_median_s": cold["t_run_median_s"],
        })
        r = rows[-1]
        print(f"  n_args={n:<5} cold={r['cold_s']:7.3f}s  warm(median of 3)={r['warm_median_s']:.3f}s"
              f"  speedup={r['speedup']:6.1f}x  nbc={r['nbc_bytes']/1024:.0f} KiB")
    RESULTS.append({"scenario": "scaling", "rows": rows})
    return rows


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-args", type=int, default=200,
                    help="arguments in the generated driver for the invalidation matrix")
    ap.add_argument("--scaling", action="store_true",
                    help="also run the cold-vs-warm sweep over driver sizes")
    ap.add_argument("--scaling-sizes", type=int, nargs="+", default=[200, 400, 800])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--json-out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--only", choices=["all", "stale"], default="all",
                    help="'stale' runs only the forged-stamp / changed-constant check")
    args = ap.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="numba_cache_survival_")
    os.makedirs(workdir, exist_ok=True)
    n = args.n_args
    print(f"python   : {PYTHON}")
    import numba
    print(f"numba    : {numba.__version__}")
    print(f"workdir  : {workdir}")
    print(f"n_args   : {n}")

    if args.only == "stale":
        part_stale(workdir, n, args.reps)
        with open(args.json_out.replace(".json", "_stale.json"), "w") as f:
            json.dump({"numba": numba.__version__, "n_args": n, "results": RESULTS},
                      f, indent=2)
        print(f"\nworkdir kept at {workdir}")
        return

    part1_exec(workdir)

    # ---- [2] cold vs warm in a fresh process, real .py -------------------
    print(f"\n[2] real .py, cache=True, fresh process each time (n_args={n})")
    A = os.path.join(workdir, "build_a")
    pa = os.path.join(A, "drv.py")
    sha_a = write_driver(pa, n)
    print(f"  driver sha256: {sha_a}")
    cold = run_child(pa, n, reps=args.reps)
    record("A1 cold (empty cache)", "does a first build compile?", cold, sha_a)
    warm_runs = [run_child(pa, n, reps=args.reps) for _ in range(3)]
    for i, w in enumerate(warm_runs):
        record(f"A2 warm, file untouched #{i+1}", "does an untouched file hit?", w, sha_a)
    warm_med = statistics.median(w["t_first_call_s"] for w in warm_runs)
    print(f"  cold={cold['t_first_call_s']:.3f}s  warm median of 3={warm_med:.3f}s"
          f"  speedup={cold['t_first_call_s']/warm_med:.1f}x")

    # ---- [3] the invalidation matrix ------------------------------------
    print("\n[3] what invalidates the cache")

    # B: byte-identical rewrite at the same path (mtime moves)
    stamp_before = warm_runs[-1]["source_stamp"]
    time.sleep(0.05)
    sha_b = write_driver(pa, n)
    assert sha_b == sha_a, "regenerated file is not byte-identical"
    record("B byte-identical rewrite, same path",
           "REVIEW sec 7: does a new mtime miss?",
           run_child(pa, n, reps=args.reps), sha_b,
           {"identical_bytes": True, "stamp_before": stamp_before})

    # C: byte-identical rewrite with mtime restored
    st = os.stat(pa)
    keep = (st.st_atime, st.st_mtime)
    time.sleep(0.05)
    sha_c = write_driver(pa, n)
    os.utime(pa, keep)
    assert sha_c == sha_a
    record("C identical rewrite + os.utime restore",
           "does restoring mtime rescue the cache?",
           run_child(pa, n, reps=args.reps), sha_c, {"identical_bytes": True})

    # D: touch only -- content untouched, mtime bumped
    st = os.stat(pa)
    os.utime(pa, (st.st_atime, st.st_mtime + 10.0))
    record("D touch only (bytes unchanged)",
           "is mtime alone enough to invalidate?",
           run_child(pa, n, reps=args.reps), sha256_file(pa), {"identical_bytes": True})

    # E: identical bytes at a different path, no cache copied
    E = os.path.join(workdir, "build_e")
    pe = os.path.join(E, "drv.py")
    sha_e = write_driver(pe, n)
    assert sha_e == sha_a
    record("E identical bytes, different dir",
           "does the cache follow the content?",
           run_child(pe, n, reps=args.reps), sha_e, {"identical_bytes": True})

    # F: whole directory copied with mtimes preserved (the Docker-image case)
    F = os.path.join(workdir, "build_f")
    shutil.rmtree(F, ignore_errors=True)
    shutil.copytree(A, F, copy_function=shutil.copy2)
    record("F dir copied, mtimes preserved",
           "does a relocated build dir keep its cache?",
           run_child(os.path.join(F, "drv.py"), n, reps=args.reps),
           sha256_file(os.path.join(F, "drv.py")), {"identical_bytes": True})

    # G: same content, different file BASENAME, cache dir copied
    G = os.path.join(workdir, "build_g")
    shutil.rmtree(G, ignore_errors=True)
    shutil.copytree(A, G, copy_function=shutil.copy2)
    os.rename(os.path.join(G, "drv.py"), os.path.join(G, "driver_v2.py"))
    record("G same content, renamed file",
           "is the filename part of the key?",
           run_child(os.path.join(G, "driver_v2.py"), n, reps=args.reps),
           sha256_file(os.path.join(G, "driver_v2.py")), {"identical_bytes": True})

    # H: comment changed, same line count -- bytes differ
    H = os.path.join(workdir, "build_h")
    shutil.rmtree(H, ignore_errors=True)
    shutil.copytree(A, H, copy_function=shutil.copy2)
    ph = os.path.join(H, "drv.py")
    sha_h = write_driver(ph, n, comment="decider2 generated driver (rebuild 2)")
    assert sha_h != sha_a
    record("H comment changed (bytes differ)",
           "doc 05 sec 4.2: do differing bytes miss?",
           run_child(ph, n, reps=args.reps), sha_h, {"identical_bytes": False})

    # I: extra comment LINE -- shifts the function's first line number
    I_ = os.path.join(workdir, "build_i")
    shutil.rmtree(I_, ignore_errors=True)
    shutil.copytree(A, I_, copy_function=shutil.copy2)
    pi = os.path.join(I_, "drv.py")
    with open(pi) as f:
        body = f.read()
    with open(pi, "w") as f:
        f.write("# extra header line\n" + body)
    record("I extra comment LINE prepended",
           "does a line shift change the cache FILE, not just the entry?",
           run_child(pi, n, reps=args.reps), sha256_file(pi), {"identical_bytes": False})

    # L: comment changed to a SAME-LENGTH string, mtime restored.
    # Bytes differ; (st_mtime, st_size) does not. Is "byte-identical" the real rule?
    L = os.path.join(workdir, "build_l")
    pl = os.path.join(L, "drv.py")
    sha_l0 = write_driver(pl, n, comment="decider2 generated driver")
    l_cold = run_child(pl, n, reps=args.reps)
    record("L1 cold", "baseline for the same-length comment test", l_cold, sha_l0)
    st = os.stat(pl)
    keep = (st.st_atime, st.st_mtime)
    sha_l1 = write_driver(pl, n, comment="decider2 GENERATED DRIVER")  # same length
    os.utime(pl, keep)
    assert sha_l1 != sha_l0, "comment change did not change the bytes"
    assert os.stat(pl).st_size == st.st_size, "comment change changed the file size"
    l_res = run_child(pl, n, reps=args.reps)
    record("L2 comment differs, stamp identical",
           "doc 05 sec 4.2: is byte-identical actually required?",
           l_res, sha_l1,
           {"identical_bytes": False, "sha_before": sha_l0,
            "value_cold": l_cold.get("value"), "value_after": l_res.get("value")})

    part_stale(workdir, n, args.reps)

    # ---- [4] index contents ---------------------------------------------
    idx = warm_runs[-1]["index_path"]
    inspect_index(idx)

    # ---- [5] NUMBA_CPU_NAME ---------------------------------------------
    print("\n[5] NUMBA_CPU_NAME=generic (doc 05 sec 8)")
    J = os.path.join(workdir, "build_j")
    pj = os.path.join(J, "drv.py")
    sha_j = write_driver(pj, n)
    record("J1 default CPU, cold", "baseline", run_child(pj, n, reps=args.reps), sha_j)
    record("J2 default CPU, warm", "baseline", run_child(pj, n, reps=args.reps), sha_j)
    gen_env = {"NUMBA_CPU_NAME": "generic", "NUMBA_CPU_FEATURES": ""}
    record("J3 generic against native cache",
           "do CPU features change the key?",
           run_child(pj, n, env_extra=gen_env, reps=args.reps), sha_j)
    record("J4 generic, second run", "does generic cache on its own?",
           run_child(pj, n, env_extra=gen_env, reps=args.reps), sha_j)
    record("J5 back to default CPU", "do both entries coexist?",
           run_child(pj, n, reps=args.reps), sha_j)
    idx_j = RESULTS[-1]["index_path"]
    with open(idx_j, "rb") as f:
        pickle.load(f)
        stamp_j, ov_j = pickle.loads(f.read())
    print(f"  index now holds {len(ov_j)} entries; magic_tuples:")
    for k in ov_j:
        print(f"    {k[1]}")
    RESULTS.append({"scenario": "cpu_name_index",
                    "n_entries": len(ov_j),
                    "magic_tuples": [list(k[1]) for k in ov_j]})

    # ---- [6] sys.modules registration ------------------------------------
    print("\n[6] driver imported WITHOUT registering it in sys.modules")
    K = os.path.join(workdir, "build_k")
    pk = os.path.join(K, "drv.py")
    sha_k = write_driver(pk, n)
    record("K1 unregistered, cold",
           "does the build-side compile+save work?",
           run_child(pk, n, register=False, reps=args.reps), sha_k,
           {"registered": False})
    record("K2 unregistered, warm",
           "can the entry it wrote be loaded back?",
           run_child(pk, n, register=False, reps=args.reps), sha_k,
           {"registered": False})
    record("K3 registered, poisoned cache kept",
           "does registering later fix the load?",
           run_child(pk, n, register=True, reps=args.reps), sha_k,
           {"registered": True})
    shutil.rmtree(os.path.join(K, "__pycache__"), ignore_errors=True)
    record("K4 registered, cache wiped, cold",
           "is the reader or the stored entry at fault?",
           run_child(pk, n, register=True, reps=args.reps), sha_k,
           {"registered": True})
    record("K5 registered, warm",
           "does it cache normally once registered?",
           run_child(pk, n, register=True, reps=args.reps), sha_k,
           {"registered": True})

    if args.scaling:
        scaling(workdir, args.scaling_sizes, args.reps)

    with open(args.json_out, "w") as f:
        json.dump({"python": PYTHON, "numba": numba.__version__,
                   "workdir": workdir, "n_args": n, "results": RESULTS}, f, indent=2)
    print(f"\nwrote {args.json_out}")
    print(f"workdir kept at {workdir}")


if __name__ == "__main__":
    main()
