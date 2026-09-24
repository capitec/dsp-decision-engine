# Forcing branches and loops, value breakpoints

How the VS Code debugger forces a branch's arm or a loop's iteration count, and how it
pauses on a value (`tools/vscode-decider/python/controls.py`).

- **A force is an override of the condition's output.** At the checkpoint just after a
  branch's or loop's condition step, the output is replaced with `session.set`. Arm k
  becomes `True`/`False` for a bool condition or the int k. A loop forced to n iterations
  gets `True` before iterations 1..n and `False` after. The runner reads the condition
  only after that checkpoint, so it routes as told. Nothing in the engine changed, and
  the override is recorded like any other `set`, so lineage shows it as "set by you".
  Changing the IR or the runner to take a forced arm was the alternative, but that would
  mean an engine change for a debugging feature.
- **It hooks in as a predicate breakpoint.** `Controls.check` is the first entry in
  `session.breakpoints`. It applies forces and returns False, so it never pauses by
  itself. `_go` evaluates breakpoints with `any`, which stops at the first True. Being
  first means a user breakpoint that pauses on the same checkpoint can't get in before
  the force is applied.
- **The same object drives traces and forks.** `runs.trace` and `forks.fork` attach a
  fresh `Controls` with the forces. That is how "arm 0 vs arm 1" and "5 vs 10
  iterations" comparisons work: two traces, one force each. The sweep's from-here forks
  carry the live session's forces, so the replay reproduces the forced run.
- **Limits.**
  - `rewind` replays without checking breakpoints, so a force set after the branch ran
    applies only once the run is rewound to the branch itself. The UI's "Re-run forced"
    does exactly that.
  - A loop forced beyond `max_iterations` stops at the maximum.
  - A run paused on the condition's "after" checkpoint has already passed the
    breakpoint check. So the bridge applies forces at the current checkpoint before
    every step or resume.
- **Value breakpoints fire on records that newly meet the condition.** A watch is checked
  just after a step that writes the name. The step must be inside the watch's scope
  (path prefixes, empty for anywhere), and the watch can be limited to one record. It
  pauses only for records that didn't meet the condition at the previous check. A value
  that stays under a limit through five loop iterations pauses once, not five times.
  Every pause reports the records and their values.
- **An iteration breakpoint** pauses before the loop's condition step at iteration k,
  which is where the loop decides whether to run iteration k.
- **Comparisons label a step one run skipped as "not taken".** A trace can't tell a
  skipped step from an arm the other run never took, because both runs share one IR.
  The extension relabels steps it knows it skipped as "removed".
