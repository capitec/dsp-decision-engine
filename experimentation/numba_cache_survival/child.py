"""Fresh-process worker. One scenario per process, because numba's in-memory
dispatcher cache would otherwise hide every on-disk cache effect.

Usage: child.py '<json config>'
  config keys:
    py_path   path to the generated driver .py to import and call
    n_args    number of scalar arguments the driver takes
    reps      number of timed repeats of the already-compiled call (default 5)
    modname   module name to import the driver under (default decider2_gen_driver)
    register_in_sys_modules  put the module in sys.modules before exec (default True)

Prints one JSON object on stdout. Everything else goes to stderr.
"""
import json
import os
import statistics
import sys
import time


def main():
    cfg = json.loads(sys.argv[1])
    py_path = cfg["py_path"]
    n_args = cfg["n_args"]
    reps = cfg.get("reps", 5)
    register = cfg.get("register_in_sys_modules", True)
    modname = cfg.get("modname", "decider2_gen_driver")

    t_imp0 = time.perf_counter()
    import numba  # noqa: F401
    t_import_numba = time.perf_counter() - t_imp0

    import importlib.util

    # Import the generated driver by path, exactly as a decider2 runtime load
    # would import a driver written at build time.
    spec = importlib.util.spec_from_file_location(modname, py_path)
    mod = importlib.util.module_from_spec(spec)
    if register:
        # Without this, inspect.getmodule() returns None inside numba and the
        # cached Environment is pickled under the module name '<dynamic>'.
        sys.modules[modname] = mod
    t0 = time.perf_counter()
    spec.loader.exec_module(mod)
    t_exec_module = time.perf_counter() - t0

    args = tuple(float(i % 3) / 2.0 for i in range(n_args))

    # THE measurement: first call = compile, or load from the on-disk cache.
    t0 = time.perf_counter()
    val = mod.driver(*args)
    t_first_call = time.perf_counter() - t0

    # Already-compiled call, for reference. Warm up once, then time repeats.
    mod.driver(*args)
    run_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mod.driver(*args)
        run_times.append(time.perf_counter() - t0)

    st = mod.driver.stats
    cache = mod.driver._cache
    impl = cache._impl
    result = {
        "ok": True,
        "py_path": py_path,
        "t_import_numba_s": t_import_numba,
        "t_exec_module_s": t_exec_module,
        "t_first_call_s": t_first_call,
        "t_run_median_s": statistics.median(run_times),
        "value": val,
        "cache_hits": {str(k): v for k, v in st.cache_hits.items()},
        "cache_misses": {str(k): v for k, v in st.cache_misses.items()},
        "n_hits": sum(st.cache_hits.values()),
        "n_misses": sum(st.cache_misses.values()),
        "cache_path": st.cache_path,
        "index_path": cache._cache_file._index_path,
        "filename_base": impl.filename_base,
        "locator_class": type(impl.locator).__name__,
        "source_stamp": list(impl.locator.get_source_stamp()),
        "numba_cpu_name": os.environ.get("NUMBA_CPU_NAME", "<unset>"),
        "registered_in_sys_modules": register,
        "modname": modname,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # report, never mask
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        raise
