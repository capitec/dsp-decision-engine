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

## 2026-09-23, after Phase 0

- **Autonomy.** The orchestrator runs every phase to completion without
  stopping at checkpoints; it still never pushes or opens PRs.
- **No code generation.** Nothing renders Python source, writes generated
  `.py` files or calls `exec`/`eval`. Fused kernels use decider2's
  `numba.extending.intrinsic` approach (fixed `kernel(n, cols, valids,
  params_all, outs)` signature, tuples for every count); trees and tables are
  data walked by generic kernels. Overrides any "real files" wording in
  Prompt.md or Design.md.
- **Performance.** A slight regression against decider2 is acceptable; a large
  one is not. T3.3 reports batch throughput and `score()` latency against
  decider2.
- **Kernel grouping default.** One kernel per innermost sequence of `scalar`
  CallNodes (a named `dag`/`flow` of plain function steps); outer sequences
  are a Python loop over kernels. Nothing wider is fused implicitly.
- **Python fallback** catches `NumbaError` and `UnsupportedBytecodeError`
  (compile-time only); runtime errors propagate.
- **Compile content key** includes constants (`co_consts`), not only
  `co_code`.
- **Trees** accept both decider_old's flat rules format and its v3 format
  through one `TreeConfig` interface. v1/v2 documents raise a clear
  deprecation error naming the version.
