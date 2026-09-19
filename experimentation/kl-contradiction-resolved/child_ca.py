import json, os, sys
workdir, modname = sys.argv[1], sys.argv[2]
sys.path.insert(0, workdir)
mod = __import__(modname)
print(json.dumps({'ok': True, 'value': mod.driver(0.05), 'co_consts': list(mod.driver.py_func.__code__.co_consts)}))
