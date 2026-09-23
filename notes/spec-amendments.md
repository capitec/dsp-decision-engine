# Spec amendments agreed with the user

Changes to IR.md / Plan.md agreed after sign-off. IR.md itself is not edited;
where they disagree, this file wins.

## 2026-09-23

- **Model tiers.** Every task runs on Opus, whatever tier Plan.md names.
- **numba / polars imports (IR.md §1 point 7, §10 test 11).** Not a hard
  requirement. `steps/`, `engine/ir/` and `engine/params/` may import numba or
  polars when that is the simplest route. Acceptance test 11 is dropped.
- **Type tags live in the registry (T2.1), not T1.1.** T2.1 runs before T1.1;
  `ConfigurableStep` in T1.1 uses the registry for import-path tags, aliases,
  duplicate detection and did-you-mean.
- **`param()` without a usable default.** decider2 rejected `None` and `bool`
  defaults because `param()` returns a subclass of the default's type (so the
  bare function stays callable), and `bool`/`NoneType` can't be subclassed.
  New rule:
  - `param(default, ...)` accepts any default, including `None` and `bool`.
    For subclassable types it still returns a carrier of the default; for
    `bool`/`None` it returns a plain marker.
  - `param(required=True, ...)` declares a param with no default; missing from
    the params document means INVALID.
  - `FunctionStep.__call__` replaces any param marker left in the call with its
    default, so decorated steps are callable with defaults for every type.
- **Renames are allowed** where a decider2 name no longer fits (e.g. its
  reserved `shared` *bundle* is dropped; the top-level `"shared"` key in the
  params document stays).
- **Comments.** Code comments are minimal. Public API docstrings are user docs:
  what it does, non-obvious arguments, a short example. No references to docs,
  chapters, sections, stages or experiments. Design rationale goes here in
  `notes/`.
