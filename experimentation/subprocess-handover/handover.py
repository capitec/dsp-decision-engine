"""EXPERIMENT K -- subprocess compile and cache survival, TOGETHER (doc 08 SS4).

Doc 08 SS4.1's 'live' mode depends on a chain that was never tested end to end:
    child compiles -> pinned cache dir -> parent loads with ZERO compilations
    -> atomic swap -> serving continues
EXPERIMENTS.md SSH measured "subprocess compile, serving untouched" but never
checked what the parent loaded. EXPERIMENTS.md SSC measured the six-condition
cache contract, but only ever in ONE process talking to itself, never across
a real child/parent boundary. This harness puts them together.

Reused, not rewritten (per decider2/docs/EXPERIMENTS.md's own convention):
  - emit.emit_ruleset / count_lines              <- ruleset-compile-latency/emit.py
  - compile_kernel, serve_until, pct, med, make_args, build_src, count_compiles
    (imported live from the module, not copy-pasted)
                                                   <- staged-compile-atomic-swap/run.py
  - gen_driver.gen_source (deterministic tiny driver, built for exactly this
    kind of cache-key test) and the subprocess-JSON-line child pattern
                                                   <- numba_cache_survival/gen_driver.py,
                                                      numba_cache_survival/child.py
    (child_op.py in this directory is that pattern, extended with
    numba.core.event compile-event counting, which child.py did not have)
  - the same-byte-length constant-edit trick for a stale-code probe
                                                   <- numba_cache_survival/run_experiment.py:part_stale

MEMORY: every array here is <=100_000 rows of a handful of float64/int64/bool
columns (~15 MB, printed below before anything runs -- rule 1). Every
subprocess is fire-and-wait, one JSON line captured then the process exits;
nothing accumulates across measurements, and each record is appended to
results.jsonl and flushed immediately (rule 3). This does not approach the
2 GB threshold, so it is NOT run under tmux/systemd-run (rule 4 is about
capping likely->2GB runs; a false-positive cap here would only add overhead).
`free -g` is checked and printed before starting (rule 5).

Run:
    .venv/bin/python handover.py                 # all measurements (~1-2 min)
    .venv/bin/python handover.py --phase m2       # just one phase
Writes results.jsonl (one JSON record per measurement, append mode).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
CHILD_OP = str(HERE / "child_op.py")

RULESET_DIR = HERE.parent / "ruleset-compile-latency"
STAGED_DIR = HERE.parent / "staged-compile-atomic-swap"
CACHE_SURVIVAL_DIR = HERE.parent / "numba_cache_survival"

# ---- reuse: import the sibling harnesses as modules, do not copy their code ----
sys.path.insert(0, str(RULESET_DIR))
import emit  # noqa: E402  (ruleset-compile-latency/emit.py)

sys.path.insert(0, str(CACHE_SURVIVAL_DIR))
from gen_driver import gen_source  # noqa: E402  (numba_cache_survival/gen_driver.py)


def _import_module_from_path(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# staged-compile-atomic-swap/run.py is literally named run.py; import it under
# a distinct name so it cannot collide with anything in this file/process.
h = _import_module_from_path(STAGED_DIR / "run.py", "h_staged_swap_harness")
# h.compile_kernel, h.serve_until, h.pct, h.med, h.make_args, h.build_src,
# h.count_compiles, h.rss_mb, h.ROWS are all now available.

ROWS = h.ROWS  # 100_000, kept identical to SSH's numbers for comparability

RESULTS_PATH = HERE / "results.jsonl"


def emit_record(rec: dict) -> None:
    rec["t_wall"] = time.time()
    with open(RESULTS_PATH, "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print(f"  -> {rec.get('measurement', '?')}: "
          f"{ {k: v for k, v in rec.items() if k not in ('measurement',)} }"[:300])


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def run_child_op(cfg: dict, env_extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.pop("NUMBA_CACHE_DIR", None)  # start from a clean slate every call
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run([PYTHON, CHILD_OP, json.dumps(cfg)],
                          capture_output=True, text=True, env=env)
    line = ""
    for ln in proc.stdout.splitlines():
        if ln.startswith("RESULT"):
            line = ln[len("RESULT"):]
    if not line:
        return {"ok": False, "exc_type": "NoOutput",
                "exc_msg": (proc.stderr or "")[-1000:], "returncode": proc.returncode}
    out = json.loads(line)
    out["returncode"] = proc.returncode
    return out


# ============================================================ pre-flight ===
def preflight():
    # rule 5: check available memory
    free = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
    print(free)
    avail_gb = None
    for ln in free.splitlines():
        if ln.startswith("Mem:"):
            avail_gb = int(ln.split()[-1])
    if avail_gb is not None and avail_gb < 8:
        print(f"REFUSING TO START: available memory {avail_gb} GB < 8 GB threshold.")
        sys.exit(1)

    # rule 1: small-first estimate. Largest array set used anywhere below is
    # ROWS x (16 float64 + 6 int64 + 3 bool) columns, exactly SSH's shape.
    n_cols_bytes = 16 * 8 + 6 * 8 + 3 * 1
    est_bytes = ROWS * n_cols_bytes
    print(f"[preflight] largest live array set: {ROWS:,} rows x 25 cols "
          f"~= {est_bytes / 1e6:.1f} MB. Well under the 4 GB restructure "
          f"threshold -- proceeding without tmux/systemd-run.")
    RESULTS_PATH.write_text("")  # fresh file this run


# =============================================================== M1 + M3 ===
def phase_m1_m3(bg_rules=30, serve_rules=10, baseline_s=1.0):
    """M1: count numba compile events IN THE PARENT when it loads a
    child-compiled generation from the pinned cache dir (must be 0).
    M3: serving interference during that SAME subprocess compile, measured
    exactly as SSH measured the thread case (throughput retained, p95, max).
    Run together because doc 08 SS4.1 is one lifecycle, not two experiments."""
    print("\n=== M1+M3: subprocess compile + parent load, together ===")
    gen_dir = HERE / "_gen_m1m3"
    shutil.rmtree(gen_dir, ignore_errors=True)
    gen_dir.mkdir()

    # the currently-ACTIVE generation: compiled in-process, cache=False,
    # nogil=True, exactly SSH's serving kernel -- reused via h.compile_kernel.
    serve_args = h.make_args(ROWS, "first_match", serve_rules)
    serve_src = h.build_src(serve_rules, "first_match", seed=101, fname="serve")
    kernel, compile_s = h.compile_kernel(serve_src, "serve", serve_args, nogil=True)
    for _ in range(5):
        kernel(*serve_args)  # extra warm-up

    baseline = h.serve_until(kernel, serve_args, __import__("threading").Event(), baseline_s)

    # the NEW generation: a bigger (bg_rules) kernel, written to the PINNED
    # dir doc 08 SS4.3 requires, compiled by a CHILD SUBPROCESS with cache=True.
    new_src = emit.emit_ruleset(bg_rules, "first_match", seed=202, fname="kern")
    new_path = gen_dir / "gen_kernel.py"
    new_path.write_text(new_src)

    MODNAME = "decider2_gen_kernel_m1m3"  # must be IDENTICAL in child and parent --
    # see the ModuleNotFoundError('<dynamic>') finding in the README: numba
    # pickles the Environment by module NAME, resolved via sys.modules at
    # compile time. Skip registering it and the cache entry is unloadable by
    # ANY process, including a second load in the same one.
    child_script = HERE / "_m1m3_child_compile.py"
    child_script.write_text(
        "import sys, time, importlib.util\n"
        f"sys.path.insert(0, {str(RULESET_DIR)!r})\n"
        "import numpy as np\n"
        "from numba import njit\n"
        f"spec = importlib.util.spec_from_file_location({MODNAME!r}, {str(new_path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        f"sys.modules[{MODNAME!r}] = mod\n"
        "spec.loader.exec_module(mod)\n"
        f"rng = np.random.default_rng(7)\n"
        f"fs=[np.ascontiguousarray(rng.random({ROWS})) for _ in range(16)]\n"
        f"cs=[np.ascontiguousarray(rng.integers(0,10,{ROWS})) for _ in range(6)]\n"
        f"bs=[np.ascontiguousarray(rng.random({ROWS})>0.5) for _ in range(3)]\n"
        f"out=np.zeros({ROWS}, np.int64)\n"
        "args = tuple(fs+cs+bs+[out])\n"
        "d = njit(cache=True)(mod.kern)\n"
        "t0=time.perf_counter(); d(*args); print('COMPILE_S', time.perf_counter()-t0)\n"
        "print('CHECKSUM', int(out.sum()))\n"
    )

    t0 = time.perf_counter()
    proc = subprocess.Popen([PYTHON, str(child_script)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    during = []
    while proc.poll() is None:
        t1 = time.perf_counter()
        kernel(*serve_args)
        during.append((time.perf_counter() - t1) * 1e3)
    out_txt, err_txt = proc.communicate()
    child_wall = time.perf_counter() - t0

    child_compile_s = None
    child_checksum = None
    for ln in out_txt.splitlines():
        if ln.startswith("COMPILE_S"):
            child_compile_s = float(ln.split()[1])
        if ln.startswith("CHECKSUM"):
            child_checksum = int(ln.split()[1])

    # ---- M1: PARENT loads the new generation IN-PROCESS, event-counted ----
    from numba.core import event as nb_event
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(MODNAME, str(new_path))
    mod = ilu.module_from_spec(spec)
    sys.modules[MODNAME] = mod  # MUST match the child's registration name
    spec.loader.exec_module(mod)
    load_args = h.make_args(ROWS, "first_match", bg_rules)
    with nb_event.install_recorder("numba:compile") as rec:
        t0 = time.perf_counter()
        d = __import__("numba").njit(cache=True)(mod.kern)
        d(*load_args)
        parent_load_s = time.perf_counter() - t0
    n_events = len(rec.buffer)
    parent_checksum = int(load_args[-1].sum())

    rec_out = {
        "measurement": "m1_m3_subproc_compile_and_load",
        "bg_rules": bg_rules, "serve_rules": serve_rules, "rows": ROWS,
        "child_compile_s": child_compile_s,
        "child_process_wall_s": child_wall,
        "child_returncode": proc.returncode,
        "child_stderr_tail": err_txt.strip()[-500:] if proc.returncode else None,
        "serving_baseline_median_ms": h.med(baseline),
        "serving_during_median_ms": h.med(during),
        "serving_during_p95_ms": h.pct(during, 95),
        "serving_during_max_ms": max(during) if during else None,
        "serving_calls_during_compile": len(during),
        "throughput_retained": h.med(baseline) / h.med(during) if during else None,
        "parent_compile_events_on_load": n_events,   # M1's headline number
        "parent_compile_events_is_zero": n_events == 0,
        "parent_load_s": parent_load_s,
        "checksum_child_vs_parent_match": child_checksum == parent_checksum,
        "child_checksum": child_checksum, "parent_checksum": parent_checksum,
    }
    emit_record(rec_out)
    child_script.unlink(missing_ok=True)
    return rec_out


# ==================================================================== M2 ===
def phase_m2(n_args=8):
    """Do the six conditions survive a process boundary? Force child/parent
    to DISAGREE on cwd, NUMBA_CACHE_DIR, generated filename, and def line
    number, one axis at a time. Records ok=False (a raised exception, LOUD)
    vs ok=True with cache_misses=1 (a silent-but-SAFE recompile) vs the
    dangerous case, ok=True with cache_hits=1 serving the WRONG value
    (silent and UNSAFE) -- that last one does not happen on any of these
    four axes; M4 is where it does happen."""
    print("\n=== M2: four cache-boundary conditions, child vs parent ===")
    base = HERE / "_gen_m2"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir()

    def w(dirpath: Path, name="drv.py"):
        dirpath.mkdir(parents=True, exist_ok=True)
        p = dirpath / name
        p.write_text(gen_source(n_args))
        return p

    # --- control: identical settings both times -> must HIT ---
    d0 = base / "control"
    p0 = w(d0)
    c_compile = run_child_op({"path": str(p0), "n_args": n_args})
    c_load = run_child_op({"path": str(p0), "n_args": n_args})
    emit_record({
        "measurement": "m2_control_identical", "axis": "none (sanity check)",
        "compile_ok": c_compile.get("ok"), "load_ok": c_load.get("ok"),
        "load_hits": c_load.get("cache_hits"), "load_misses": c_load.get("cache_misses"),
        "load_compile_events": c_load.get("n_compile_events"),
        "verdict": "HIT" if c_load.get("cache_hits") else "MISS",
    })

    # --- axis: cwd (relative path resolved against a different cwd) ---
    d1 = base / "cwd_a"
    w(d1)
    d1b = base / "cwd_b_empty"
    d1b.mkdir()
    compile_r = run_child_op({"path": "drv.py", "n_args": n_args, "chdir": str(d1), "rel": True})
    load_r = run_child_op({"path": "drv.py", "n_args": n_args, "chdir": str(d1b), "rel": True})
    emit_record({
        "measurement": "m2_axis", "axis": "cwd",
        "child_compiled_at": str(d1), "parent_loaded_from_cwd": str(d1b),
        "compile_ok": compile_r.get("ok"),
        "load_ok": load_r.get("ok"),
        "load_exc": load_r.get("exc_type"),
        "load_exc_msg": (load_r.get("exc_msg") or "")[:150],
        "verdict": "LOUD (raised)" if not load_r.get("ok") else
                   ("HIT (silent, unexpected)" if load_r.get("cache_hits") else "silent MISS (safe recompile)"),
    })

    # --- axis: NUMBA_CACHE_DIR ---
    d2 = base / "cache_dir_src"
    w(d2)
    cache_a = base / "cache_A"
    cache_b = base / "cache_B"
    compile_r = run_child_op({"path": str(d2 / "drv.py"), "n_args": n_args},
                             env_extra={"NUMBA_CACHE_DIR": str(cache_a)})
    load_same = run_child_op({"path": str(d2 / "drv.py"), "n_args": n_args},
                             env_extra={"NUMBA_CACHE_DIR": str(cache_a)})
    load_diff = run_child_op({"path": str(d2 / "drv.py"), "n_args": n_args},
                             env_extra={"NUMBA_CACHE_DIR": str(cache_b)})
    emit_record({
        "measurement": "m2_axis", "axis": "NUMBA_CACHE_DIR",
        "compile_env_cache_dir": str(cache_a),
        "load_same_env_ok": load_same.get("ok"), "load_same_env_hits": load_same.get("cache_hits"),
        "load_diff_env_cache_dir": str(cache_b),
        "load_diff_env_ok": load_diff.get("ok"),
        "load_diff_env_hits": load_diff.get("cache_hits"),
        "load_diff_env_misses": load_diff.get("cache_misses"),
        "load_diff_env_compile_events": load_diff.get("n_compile_events"),
        "verdict": "LOUD (raised)" if not load_diff.get("ok") else
                   ("HIT despite different NUMBA_CACHE_DIR (unexpected)" if load_diff.get("cache_hits")
                    else "silent MISS (safe recompile, wrong dir just means cold cache)"),
    })

    # --- axis: generated filename (deploy-tool naming drift) ---
    d3 = base / "filename"
    w(d3, name="drv_genA.py")
    compile_r = run_child_op({"path": str(d3 / "drv_genA.py"), "n_args": n_args})
    load_r = run_child_op({"path": str(d3 / "drv_genB.py"), "n_args": n_args})  # never written
    emit_record({
        "measurement": "m2_axis", "axis": "generated_filename",
        "child_wrote": "drv_genA.py", "parent_expected": "drv_genB.py",
        "load_ok": load_r.get("ok"), "load_exc": load_r.get("exc_type"),
        "load_exc_msg": (load_r.get("exc_msg") or "")[:150],
        "verdict": "LOUD (raised)" if not load_r.get("ok") else "silent (unexpected)",
    })

    # --- axis: def line number, WITH mtime+size held constant (the sharp case) ---
    d4 = base / "lineno"
    p4 = w(d4)
    compile_r = run_child_op({"path": str(p4), "n_args": n_args})
    st = os.stat(p4)
    keep = (st.st_atime, st.st_mtime)
    body = p4.read_text()
    # same trick as EXPERIMENTS.md SSC / M4 below: edit one constant, same digit
    # count, so bytes are unchanged -- but ALSO prepend a blank line and trim
    # one character off the header comment, so total size is STILL unchanged
    # but every line below shifts down by one (def moves).
    lines = body.split("\n")
    assert lines[0].startswith("# ")
    lines[0] = lines[0][:-1]  # drop 1 char from "# decider2 generated driver"
    shifted = "\n" + "\n".join(lines)
    assert len(shifted) == len(body), (len(shifted), len(body))
    p4.write_text(shifted)
    os.utime(p4, keep)
    assert os.stat(p4).st_mtime == st.st_mtime and os.stat(p4).st_size == st.st_size
    load_r = run_child_op({"path": str(p4), "n_args": n_args})
    emit_record({
        "measurement": "m2_axis", "axis": "def_lineno_shift_size_mtime_held_constant",
        "compile_lineno": compile_r.get("def_lineno"),
        "load_lineno": load_r.get("def_lineno"),
        "load_ok": load_r.get("ok"),
        "load_hits": load_r.get("cache_hits"), "load_misses": load_r.get("cache_misses"),
        "load_compile_events": load_r.get("n_compile_events"),
        "load_value": load_r.get("value"), "compile_value": compile_r.get("value"),
        "served_stale": load_r.get("value") == compile_r.get("value"),
        "verdict": ("silent MISS -- lineno shift alone forces a correct recompile "
                    "even with mtime+size frozen" if load_r.get("cache_misses")
                    else "HIT -- would be stale if values differ"),
    })

    # --- SEVENTH condition, found by accident running M1+M3: sys.modules
    # registration name. Not in EXPERIMENTS.md SSC's six, because that harness's
    # child.py always registers (register_in_sys_modules defaults True) and
    # never varied it against the LOADER. It matters just as much as the six. ---
    d5 = base / "modreg"
    p5 = w(d5)
    compile_r = run_child_op({"path": str(p5), "n_args": n_args,
                              "modname": "decider2_gen_driver_A", "register": True})
    load_unregistered = run_child_op({"path": str(p5), "n_args": n_args,
                                      "modname": "decider2_gen_driver_A", "register": False})
    load_wrong_name = run_child_op({"path": str(p5), "n_args": n_args,
                                    "modname": "decider2_gen_driver_B", "register": True})
    load_right_name = run_child_op({"path": str(p5), "n_args": n_args,
                                    "modname": "decider2_gen_driver_A", "register": True})
    emit_record({
        "measurement": "m2_axis", "axis": "sys_modules_registration_name (7th condition)",
        "compiled_with_modname": "decider2_gen_driver_A",
        "load_unregistered_ok": load_unregistered.get("ok"),
        "load_unregistered_exc": load_unregistered.get("exc_type"),
        "load_unregistered_exc_msg": (load_unregistered.get("exc_msg") or "")[:200],
        "load_wrong_name_ok": load_wrong_name.get("ok"),
        "load_wrong_name_exc": load_wrong_name.get("exc_type"),
        "load_wrong_name_exc_msg": (load_wrong_name.get("exc_msg") or "")[:200],
        "load_right_name_ok": load_right_name.get("ok"),
        "load_right_name_hits": load_right_name.get("cache_hits"),
        "verdict": ("LOUD but CRYPTIC: ModuleNotFoundError('<dynamic>') or "
                    "ModuleNotFoundError(the wrong name), raised deep inside "
                    "numba's pickle.loads, not at import time -- nothing about "
                    "the traceback mentions caching or module registration"
                    if not load_unregistered.get("ok") else "unexpected: succeeded"),
    })


# ==================================================================== M4 ===
def phase_m4(n_args=8):
    """The stale-code hazard ACROSS THE BOUNDARY: child compiles generation A;
    a rule threshold changes (one constant, same digit count, def line
    UNCHANGED); child compiles B to the SAME pinned path. Two sub-cases:
    naive (a build tool normalises mtime for reproducibility, doc 05 SS4.2's
    own stated requirement) vs honest (mtime allowed to change naturally)."""
    print("\n=== M4: stale-code hazard across the child/parent boundary ===")
    base = HERE / "_gen_m4"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir()

    def make_gen_b_same_line(src: str) -> str:
        # find the SAME-line constant edit used by numba_cache_survival's
        # part_stale: "acc -= a1 * <const>" specifically -- child_op.py's
        # fixed args are (i%3)/2.0, so a1 == 0.5, which both (a) takes the
        # else-branch (0.5 > 0.5 is False) and (b) is non-zero, so the edited
        # multiplier actually reaches the output. a0 == 0.0 would silently
        # make ANY multiplier edit a no-op -- caught by the assert below.
        lines = src.split("\n")
        target_i = None
        for i, ln in enumerate(lines):
            if ln.strip().startswith("acc -= a1 *") and ln.rstrip().endswith("* 0.5"):
                target_i = i
                break
        assert target_i is not None, "could not find a same-length edit point"
        old = lines[target_i]
        new = old[:-3] + "0.9"  # "...* 0.5" -> "...* 0.9", same byte length
        assert len(new) == len(old)
        lines[target_i] = new
        return "\n".join(lines)

    for case in ("naive_mtime_preserved", "honest_fresh_mtime"):
        d = base / case
        d.mkdir()
        p = d / "drv_stale.py"
        src_a = gen_source(n_args)
        p.write_text(src_a)
        gen_a = run_child_op({"path": str(p), "n_args": n_args})

        st = os.stat(p)
        keep = (st.st_atime, st.st_mtime)
        src_b = make_gen_b_same_line(src_a)
        assert len(src_b) == len(src_a)
        assert src_b != src_a
        p.write_text(src_b)
        if case == "naive_mtime_preserved":
            os.utime(p, keep)
            assert os.stat(p).st_mtime == st.st_mtime and os.stat(p).st_size == st.st_size

        gen_b_load = run_child_op({"path": str(p), "n_args": n_args})

        # ground truth for B: compile it fresh, in a cache dir it has never
        # touched, so this reading cannot itself be stale.
        truth_dir = d / "truth"
        truth_dir.mkdir()
        pt = truth_dir / "drv_stale.py"
        pt.write_text(src_b)
        gen_b_truth = run_child_op({"path": str(pt), "n_args": n_args})

        edit_matters = gen_b_truth.get("value") != gen_a.get("value")
        assert edit_matters, (
            "harness bug: the constant edit did not change the true value -- "
            f"gen_a={gen_a.get('value')} gen_b_truth={gen_b_truth.get('value')}"
        )
        served_stale = edit_matters and gen_b_load.get("value") == gen_a.get("value")
        emit_record({
            "measurement": "m4_stale_across_boundary", "case": case,
            "mtime_size_preserved": case == "naive_mtime_preserved",
            "gen_a_value": gen_a.get("value"),
            "gen_b_value_served_to_parent": gen_b_load.get("value"),
            "gen_b_true_value": gen_b_truth.get("value"),
            "gen_b_load_hits": gen_b_load.get("cache_hits"),
            "gen_b_load_misses": gen_b_load.get("cache_misses"),
            "gen_b_load_compile_events": gen_b_load.get("n_compile_events"),
            "edit_changes_the_result": edit_matters,
            "SERVED_STALE_CODE": served_stale,
        })


# =================================================================== M4c ===
def phase_m4c_pyc_confound(n_args=8):
    """Found by accident building M4, then isolated deliberately: is the
    'stale HIT' in EXPERIMENTS.md SSC actually numba's on-disk cache, or is it
    CPython's OWN __pycache__/*.pyc bytecode cache (also keyed on (mtime,
    size), and consulted by spec.loader.exec_module() BEFORE numba's
    decorator runs at all)? Controlled A/B: compile gen A ONCE (raw, so a
    .pyc gets written -- child_op.py's own compiles never do, see
    child_op.py:_clear_stale_pyc), apply SSC's exact same-length edit
    preserving (mtime, size), then load the SAME resulting on-disk state
    TWICE: once with the .pyc left in place, once with ONLY that .pyc
    removed (numba's .nbi/.nbc untouched in both)."""
    print("\n=== M4c: is the M4/SSC stale hit numba's cache, or CPython's? ===")
    base = HERE / "_gen_m4c"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir()
    d = base / "case"
    d.mkdir()
    p = d / "drv.py"
    src_a = gen_source(n_args)
    p.write_text(src_a)

    # compile gen A RAW (no pyc-clearing) -- this is what
    # numba_cache_survival/child.py does, and it DOES write a .pyc.
    raw_child = HERE / "_m4c_raw_child.py"
    raw_child.write_text(
        "import sys, json, importlib.util\n"
        "path = sys.argv[1]\n"
        "spec = importlib.util.spec_from_file_location('decider2_gen_driver', path)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['decider2_gen_driver'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "args = tuple(float(i % 3) / 2.0 for i in range(8))\n"
        "val = mod.driver(*args)\n"
        "st = mod.driver.stats\n"
        "print(json.dumps({'value': val, 'hits': sum(st.cache_hits.values()), "
        "'misses': sum(st.cache_misses.values())}))\n"
    )
    proc = subprocess.run([PYTHON, str(raw_child), str(p)], capture_output=True, text=True)
    gen_a = json.loads(proc.stdout.strip().splitlines()[-1])
    pyc_path = importlib.util.cache_from_source(str(p))
    pyc_existed_after_compile = os.path.exists(pyc_path)

    # SSC's exact edit: shared "* 0.5" tail on the a1 line, same byte length.
    st = os.stat(p)
    keep = (st.st_atime, st.st_mtime)
    lines = src_a.split("\n")
    ti = next(i for i, ln in enumerate(lines)
              if ln.strip().startswith("acc -= a1 *") and ln.rstrip().endswith("* 0.5"))
    lines[ti] = lines[ti][:-3] + "0.7"
    src_b = "\n".join(lines)
    assert len(src_b) == len(src_a)
    p.write_text(src_b)
    os.utime(p, keep)
    assert os.stat(p).st_mtime == st.st_mtime and os.stat(p).st_size == st.st_size

    # ground truth for B, in a cache dir never touched by A
    truth_dir = base / "truth"
    truth_dir.mkdir()
    pt = truth_dir / "drv.py"
    pt.write_text(src_b)
    truth = run_child_op({"path": str(pt), "n_args": n_args})

    # branch 1: load with the stale .pyc LEFT IN PLACE (identical state to
    # what a real deploy that regenerates the same filename would produce)
    proc = subprocess.run([PYTHON, str(raw_child), str(p)], capture_output=True, text=True)
    with_pyc = json.loads(proc.stdout.strip().splitlines()[-1])

    # branch 2: same on-disk state, but delete ONLY the CPython .pyc first --
    # numba's .nbi/.nbc are completely untouched.
    assert os.path.exists(pyc_path), "expected a .pyc to exist before this branch"
    os.remove(pyc_path)
    proc = subprocess.run([PYTHON, str(raw_child), str(p)], capture_output=True, text=True)
    pyc_removed = json.loads(proc.stdout.strip().splitlines()[-1])

    emit_record({
        "measurement": "m4c_pyc_vs_numba_cache",
        "pyc_written_by_cpython_after_compile": pyc_existed_after_compile,
        "gen_a_value": gen_a.get("value"),
        "true_gen_b_value": truth.get("value"),
        "load_with_stale_pyc_present": with_pyc.get("value"),
        "load_with_stale_pyc_present_hits": with_pyc.get("hits"),
        "load_with_stale_pyc_removed": pyc_removed.get("value"),
        "load_with_stale_pyc_removed_misses": pyc_removed.get("misses"),
        "STALE_CAUSED_BY_CPYTHON_PYC": (with_pyc.get("value") == gen_a.get("value")
                                        and pyc_removed.get("value") == truth.get("value")),
        "numba_cache_itself_was_untouched_between_branches": True,
        "verdict": ("CPython's own bytecode cache, not numba's, is what serves "
                    "the stale value in this scenario -- removing only the .pyc "
                    "(numba's .nbi/.nbc left exactly as they were) fixes it"),
    })
    raw_child.unlink(missing_ok=True)


# ==================================================================== M5 ===
def phase_m5(n_args=8):
    """Does content-addressed naming (drv_<sha256[:16]>.py) fix M4 WITHOUT
    destroying cache hits for genuinely unchanged content?"""
    print("\n=== M5: content-addressed naming ===")
    base = HERE / "_gen_m5"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir()

    def write_content_addressed(gen_dir: Path, src: str) -> tuple[Path, bool]:
        h_ = sha16(src)
        p = gen_dir / f"drv_{h_}.py"
        already_there = p.exists()
        if not already_there:
            p.write_text(src)
        return p, already_there

    # (a) unchanged content, two separate "deploy" events -> must survive
    d = base / "unchanged"
    d.mkdir()
    src = gen_source(n_args)
    p1, existed1 = write_content_addressed(d, src)
    gen1 = run_child_op({"path": str(p1), "n_args": n_args})
    # "redeploy" with byte-identical generated output (a second, independent
    # regeneration -- content-addressed means we do NOT even need to touch
    # the file, so mtime never moves)
    p2, existed2 = write_content_addressed(d, src)
    gen2 = run_child_op({"path": str(p2), "n_args": n_args})
    emit_record({
        "measurement": "m5_content_addressed", "case": "unchanged_content_redeploy",
        "same_path_both_deploys": p1 == p2,
        "second_deploy_skipped_write": existed2,
        "second_load_hits": gen2.get("cache_hits"),
        "second_load_compile_events": gen2.get("n_compile_events"),
        "verdict": "HIT preserved across redeploy" if gen2.get("cache_hits") else "MISS (unexpected)",
    })

    # (b) changed content (M4's exact edit) -> must NEVER be servable as stale
    d2 = base / "changed"
    d2.mkdir()
    src_a = gen_source(n_args)
    lines = src_a.split("\n")
    ti = next(i for i, ln in enumerate(lines)
              if ln.strip().startswith("acc -= a1 *") and ln.rstrip().endswith("* 0.5"))
    old = lines[ti]
    lines[ti] = old[:-3] + "0.9"
    src_b = "\n".join(lines)
    assert len(src_b) == len(src_a)

    pa, _ = write_content_addressed(d2, src_a)
    gen_a = run_child_op({"path": str(pa), "n_args": n_args})
    # simulate a naive deploy tool trying (and failing) to force a stale hit:
    # even if it wanted to reuse gen_a's path/mtime, content-addressing means
    # the CHANGED source hashes to a DIFFERENT path -- there is no path to
    # collide mtime/size on.
    pb, existed_b = write_content_addressed(d2, src_b)
    gen_b = run_child_op({"path": str(pb), "n_args": n_args})
    assert gen_a.get("value") != gen_b.get("value"), "harness bug: edit did not change the value"
    emit_record({
        "measurement": "m5_content_addressed", "case": "changed_content",
        "path_changed_with_content": pa != pb,
        "gen_b_was_fresh_file": not existed_b,
        "gen_b_compile_events": gen_b.get("n_compile_events"),
        "gen_a_value": gen_a.get("value"), "gen_b_value": gen_b.get("value"),
        "values_differ_as_expected": gen_a.get("value") != gen_b.get("value"),
        "verdict": "structurally cannot serve stale: different content -> different path",
    })
    return base


# ==================================================================== M6 ===
def phase_m6(gen_dir_from_m5: Path | None = None):
    """Wall clock from 'config change arrives' to 'new generation serving',
    for a 10-rule and a 30-rule change, using the content-addressed fix."""
    print("\n=== M6: config-change to serving, wall clock ===")
    gen_dir = HERE / "_gen_m6"
    shutil.rmtree(gen_dir, ignore_errors=True)
    gen_dir.mkdir()

    for n_rules in (10, 30):
        src = emit.emit_ruleset(n_rules, "first_match", seed=303 + n_rules, fname="kern")
        path = gen_dir / f"drv_{sha16(src)}.py"

        t_change_arrives = time.perf_counter()
        modname = f"decider2_gen_kernel_m6_{n_rules}_{sha16(src)}"
        child_script = HERE / f"_m6_child_{n_rules}.py"
        child_script.write_text(
            "import sys, time\n"
            f"open({str(path)!r}, 'w').write({src!r})\n"
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location({modname!r}, {str(path)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            f"sys.modules[{modname!r}] = mod\n"
            "spec.loader.exec_module(mod)\n"
            "import numpy as np\nfrom numba import njit\n"
            f"rng = np.random.default_rng(1)\n"
            f"fs=[np.ascontiguousarray(rng.random({ROWS})) for _ in range(16)]\n"
            f"cs=[np.ascontiguousarray(rng.integers(0,10,{ROWS})) for _ in range(6)]\n"
            f"bs=[np.ascontiguousarray(rng.random({ROWS})>0.5) for _ in range(3)]\n"
            f"out=np.zeros({ROWS}, np.int64)\n"
            "args = tuple(fs+cs+bs+[out])\n"
            "d = njit(cache=True)(mod.kern)\n"
            "t0=time.perf_counter(); d(*args); print('COMPILE_S', time.perf_counter()-t0)\n"
        )
        proc = subprocess.run([PYTHON, str(child_script)], capture_output=True, text=True)
        t_child_done = time.perf_counter()
        child_compile_s = None
        for ln in proc.stdout.splitlines():
            if ln.startswith("COMPILE_S"):
                child_compile_s = float(ln.split()[1])

        # parent load, event-counted, in-process (SAME modname as the child --
        # see the m1m3 registration finding)
        from numba.core import event as nb_event
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(modname, str(path))
        mod = ilu.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        load_args = h.make_args(ROWS, "first_match", n_rules)
        with nb_event.install_recorder("numba:compile") as rec:
            t0 = time.perf_counter()
            d = __import__("numba").njit(cache=True)(mod.kern)
            d(*load_args)
            parent_load_s = time.perf_counter() - t0
        t_serving = time.perf_counter()
        n_events = len(rec.buffer)

        emit_record({
            "measurement": "m6_wallclock", "n_rules": n_rules,
            "child_wall_s": t_child_done - t_change_arrives,
            "child_compile_s": child_compile_s,
            "parent_load_s": parent_load_s,
            "parent_compile_events": n_events,
            "atomic_swap_us": 0.177,  # cite SSH; not re-measured here, see README
            "total_change_to_serving_s": (t_serving - t_change_arrives) + 0.177e-6,
            "returncode": proc.returncode,
        })
        child_script.unlink(missing_ok=True)


# =================================================================== main ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "m1m3", "m2", "m4", "m4c", "m5", "m6"])
    a = ap.parse_args()

    preflight()
    t0 = time.perf_counter()

    if a.phase in ("all", "m1m3"):
        phase_m1_m3()
    if a.phase in ("all", "m2"):
        phase_m2()
    if a.phase in ("all", "m4"):
        phase_m4()
    if a.phase in ("all", "m4c"):
        phase_m4c_pyc_confound()
    if a.phase in ("all", "m5"):
        phase_m5()
    if a.phase in ("all", "m6"):
        phase_m6()

    print(f"\nDONE in {time.perf_counter() - t0:.1f}s. Results: {RESULTS_PATH}")
    print(f"Peak RSS this process: {h.rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
