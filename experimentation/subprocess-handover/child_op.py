"""Fresh-process worker for the parent/child cache-handover experiment (K).

Adapted from experimentation/numba_cache_survival/child.py: one scenario per
process, because numba's in-memory dispatcher cache would otherwise hide
on-disk cache effects (a HIT in the same process is not a cache HIT, it is
just reusing the live Dispatcher object).

New here (not in child.py): counts numba compile *events* around the call via
numba.core.event.install_recorder, exactly as
experimentation/staged-compile-atomic-swap/run.py:count_compiles does. This
lets a "loader" process assert zero real compilations happened, the same
assertion doc 08 SS4.1 makes for parent load.

Usage: child_op.py '<json config>'
  config keys:
    path      path to the generated driver .py (see gen_driver.gen_source)
    n_args    number of scalar arguments the driver takes
    chdir     if given, os.chdir() here BEFORE importing (tests cwd-relative
              path resolution -- condition 1, "same absolute directory")
    rel       if true, import `path` as-given (possibly relative to chdir)
              instead of os.path.abspath()-ing it first
    modname   sys.modules registration name (default a fixed name so two
              children of the same scenario look identical to numba)
    register  put the module in sys.modules before exec (default True)

Prints exactly one JSON object on stdout, prefixed "RESULT". Everything else
(including the numba import itself) goes to stderr / is discarded.
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True  # do not ADD to the confound below


def _clear_stale_pyc(path: str) -> None:
    """CPython's OWN import machinery caches compiled bytecode for a .py file
    in __pycache__/<stem>.cpython-XY.pyc, validated by the SAME (mtime, size)
    stamp numba uses (see importlib._bootstrap_external). A mtime+size-
    preserving content edit (the exact scenario this experiment tests) fools
    THIS cache too, and it is consulted by spec.loader.exec_module() BEFORE
    numba's decorator ever runs -- so without this, a "did numba serve stale
    code" test can actually be measuring "did Python's loader serve a stale
    *module*", one layer up. Found empirically while building this harness
    (see README: 'the confound almost invalidated M2's def-lineno result').
    Real deployments hit this too: it is not specific to this test.
    """
    try:
        import importlib.util as ilu
        pyc = ilu.cache_from_source(path)
        if os.path.exists(pyc):
            os.remove(pyc)
    except Exception:
        pass


def main():
    cfg = json.loads(sys.argv[1])
    path = cfg["path"]
    n_args = cfg["n_args"]
    chdir = cfg.get("chdir")
    rel = cfg.get("rel", False)
    modname = cfg.get("modname", "decider2_gen_driver")
    register = cfg.get("register", True)

    if chdir:
        os.chdir(chdir)
    if not rel:
        path = os.path.abspath(path)

    import numba  # noqa: F401
    from numba.core import event as nb_event

    import importlib.util

    out = {"ok": True, "path_used": path, "cwd": os.getcwd(),
           "numba_cache_dir_env": os.environ.get("NUMBA_CACHE_DIR", "<unset>")}
    try:
        _clear_stale_pyc(path)
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        if register:
            sys.modules[modname] = mod
        spec.loader.exec_module(mod)

        args = tuple(float(i % 3) / 2.0 for i in range(n_args))

        with nb_event.install_recorder("numba:compile") as rec:
            t0 = time.perf_counter()
            val = mod.driver(*args)
            t_first_call = time.perf_counter() - t0
        n_compile_events = len(rec.buffer)

        st = mod.driver.stats
        cache = mod.driver._cache
        impl = cache._impl
        out.update({
            "t_first_call_s": t_first_call,
            "value": val,
            "n_compile_events": n_compile_events,
            "cache_hits": sum(st.cache_hits.values()),
            "cache_misses": sum(st.cache_misses.values()),
            "cache_path": st.cache_path,
            "index_path": cache._cache_file._index_path,
            "filename_base": impl.filename_base,
            "source_stamp": list(impl.locator.get_source_stamp()),
            "def_lineno": mod.driver.py_func.__code__.co_firstlineno,
        })
    except Exception as exc:  # the point: LOUD failures must show up as ok=False
        out["ok"] = False
        out["exc_type"] = type(exc).__name__
        out["exc_msg"] = str(exc)
    print("RESULT" + json.dumps(out))


if __name__ == "__main__":
    main()
