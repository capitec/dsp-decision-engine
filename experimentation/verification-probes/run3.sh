#!/bin/bash
S="$(cd "$(dirname "$0")" && pwd)"   # a scratch dir holding base/ base_inline/ bench.py -- see README
PY=/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python
WT="${WT:?set WT to the typed-features worktree decider2/src}"
cd $S
for round in 1 2; do
  echo "===== round $round ====="
  PYTHONPATH=$S/base/decider2/src        $PY $S/bench.py --label "base-r$round"        --reps 7 2>&1 | grep -v Warning
  PYTHONPATH=$S/base_inline/decider2/src $PY $S/bench.py --label "base-inline-r$round" --reps 7 2>&1 | grep -v Warning
  PYTHONPATH=$WT                         $PY $S/bench.py --label "typed-r$round"       --reps 7 2>&1 | grep -v Warning
done
echo "DONE"
