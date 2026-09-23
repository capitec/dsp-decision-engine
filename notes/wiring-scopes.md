# Wiring: scopes, barriers and the forward-reference rule

`engine/wiring/resolve.py` binds every read to a `Version`. Choices that the
runner (T1.4), compiler (T3.2) and session (T1.5) rely on:

- **Unknown names are input columns** (decider2's rule), unless one is a close
  match (difflib cutoff 0.8) for a name produced *earlier* in the walk; then it
  is a typo error. A step's own outputs are excluded, so `term_cap_a` may read
  `term_cap`. Only earlier names are candidates, so in a `dag` written in reverse
  dependency order a typo of a later sibling becomes an input column.
- **Forward references are errors** (decider2 review finding): a node reads a
  name as an input column and a later node writes that name. The self-read
  waterfall (the same node reads and writes it) is allowed. This also rejects
  "read the input, narrow it later", as decider2 did.
- **Branch arms and loop bodies are scopes.** Only `modifies` (branch) and
  `carries` (loop) leave them, as one new version produced by the branch or
  loop path. Arm-internal names and the condition's output stay inside, but are
  still in `chains`, so `name@path` can emit them. Reading one after its
  branch or loop (when nothing outside wrote it), or emitting it by bare name,
  is an error rather than an input-column read: the arm write would otherwise
  be lost silently. A scope still hides a name that also exists outside it.
- **Branch merge:** `Merge.prior` is read only when some arm leaves the name
  alone, so a name every arm writes needn't be an input column.
- **Loop carry:** one version, produced by the loop, is what the condition and
  body read each iteration and what follows the loop. The runner copies
  `Carry.last` into it at the end of each iteration.
- **Unknown-lineage frames are barriers.** After one, every name not produced
  after it is read from it: the version is produced by the frame's path and
  listed in its `Call.writes`, and the runner checks the column exists in the
  frame it returns. Names read after a barrier never become input columns and
  are never typo errors.
- **Outputs:** input columns (always), root-scope values nothing reads, emitted
  values. `name@path` qualifiers are relative to the flow that declares the
  emit, or absolute (the node path). A drop of an unknown name is kept for frame
  columns no step reads, unless it is a close match for a known name.
- **Ids** are positions (calls in walk order, versions in creation order), so
  they depend on structure, never on path text.
