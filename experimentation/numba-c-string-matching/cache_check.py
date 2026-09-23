"""Does the lazy kernel still disk-cache? Two SEPARATE processes, a
persistent NUMBA_CACHE_DIR, NUMBA_DEBUG_CACHE=1. Cold run must `save`,
warm run must `load` and save nothing. Recorded verbatim from numba's own
cache log lines, not inferred from timings."""
import os, re, shutil, subprocess, sys, time, pathlib
from results_io import record

HERE = pathlib.Path(__file__).resolve().parent
CACHE = HERE / ".numba_cache_persist"
PY = sys.executable

def run(tag, extra=()):
    env = dict(os.environ, NUMBA_CACHE_DIR=str(CACHE), NUMBA_DEBUG_CACHE="1")
    t0 = time.perf_counter()
    p = subprocess.run([PY, str(HERE / "cache_worker.py"), *extra], env=env, capture_output=True, text=True, cwd=HERE)
    wall = time.perf_counter() - t0
    log = p.stdout + p.stderr
    saved = len(re.findall(r"\[cache\] data saved", log))
    loaded = len(re.findall(r"\[cache\] data loaded", log))
    idx_saved = len(re.findall(r"\[cache\] index saved", log))
    result = re.search(r"RESULT (.*)", log)
    warn = re.search(r"GLOBAL_CAPTURE_WARNINGS (.*)", log)
    (HERE / f"cache_{tag}.log").write_text(log)
    rec = dict(tag=tag, returncode=p.returncode, wall_s=wall, data_saved=saved, index_saved=idx_saved, data_loaded=loaded,
               worker=result.group(1) if result else None)
    if warn: rec["global_capture_warnings"] = warn.group(1)[:300]
    record("cache", **rec)
    return rec

if __name__ == "__main__":
    shutil.rmtree(CACHE, ignore_errors=True)
    # also drop any __pycache__ numba index for kernels.py so nothing leaks in
    cold = run("cold")
    warm = run("warm")
    warm2 = run("warm_with_global_capture_contrast", ("--with-global-capture",))
    assert cold["returncode"] == 0 and warm["returncode"] == 0
    assert cold["data_saved"] > 0, "cold run saved nothing"
    assert warm["data_loaded"] > 0 and warm["data_saved"] == 0, "warm run did not hit cleanly"
    files = sorted(str(p.relative_to(CACHE)) for p in CACHE.rglob("*") if p.is_file())
    record("cache", tag="summary", verdict="lazy kernel disk-caches across processes",
           cold_saved=cold["data_saved"], warm_loaded=warm["data_loaded"], warm_saved=warm["data_saved"],
           cache_files=files)
    print("CACHE OK")
