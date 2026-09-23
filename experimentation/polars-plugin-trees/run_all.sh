#!/bin/bash
# Reproduce everything in RESULTS.md. Needs: rustc/cargo 1.9x, uv (for `uvx maturin`), the repo venv, network for crates.io.
# Total: ~6 min build + ~15 min of benchmarks on an idle 28-core box.
set -e
cd "$(dirname "$0")"
PY=/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python
(cd decider_trees && uvx maturin build --release -i $PY -o dist && unzip -o -q dist/*.whl 'decider_trees/*' -d .)
free -h
$PY q1_types.py                                   # §1 the int64 / bool / string triple, error messages
POLARS_MAX_THREADS=1 $PY q0_single.py ; $PY q0_single.py   # §2 PRIMARY: single record and batches of 10/100/1000, us per call
POLARS_MAX_THREADS=1 $PY q0_parts.py  ; $PY q0_parts.py    # §2 where the single-record microseconds go
POLARS_MAX_THREADS=1 $PY q0_packed.py ; $PY q0_packed.py   # §2 packed tree wire format vs list-of-dicts
$PY q2_driver.py 10000 100000 1000000 10000000    # §2 the 4-step pipeline grid (each cell its own process)
POLARS_MAX_THREADS=1 $PY q2_isolate.py            # §2 what one row-walk test costs vs polars' own vectorised test
$PY q2_depth.py                                   # §2 depth scaling: row walk vs when/then vs decider2
$PY q4_breaks.py                                  # §4 single record, equivalence, cost of shipping the tree
$PY q4b_retune_panic.py                           # §4 retune compile count, error / abort / panic demos
./q3_cost.sh                                      # §3 incremental + cold build, wheel, cargo tree (runs cargo clean!)
