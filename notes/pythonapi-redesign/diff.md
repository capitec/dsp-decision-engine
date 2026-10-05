# Diff: comparing two versions of a pipeline

## Who owns what

Five concerns currently share three words ("equivalent", "divergence", "compare"). Each gets one owner:

| Question | Owner |
|---|---|
| Is this one version risky? | `decider.checks` |
| Do the execution modes of one version agree (interpreted, stepped, fused, score, session)? | `decider.testing.assert_equivalent` (unchanged) |
| What changed between v1 and v2: structure, params, wiring? | **`decider.diff.structure`** |
| Which rows and columns change between v1 and v2, and by how much? | **`decider.diff.outputs`** |
| Run a population through many scenarios, batch by batch, with metrics and plots | `decider.experiments`. It *uses* `diff` and doesn't own it. |

The dependency then points one way: experiments → diff. Today the only real value-diff primitive is the private `Divergence._changed` in `experiments/metrics/compare.py:82`. The git-worktree machinery is `experiments/revision.py`. Both move to `diff`, and `Divergence` becomes a thin per-batch adapter over `diff`'s changed-value primitive. The subprocess isolation for revision scenarios (`notes/experiment-revision-subprocess-isolation.md`) stays where it is; `diff.revisions` reuses it.

## API

```python
from decider import diff

# structure: no data, no execution
s = diff.structure(old, new)
s.added, s.removed, s.renamed, s.moved      # StepRefs, matched by (flow_id, step_id) first, then path
s.changed_steps                             # StepChange(ref, code=..., consts=..., inputs=..., outputs=..., params=...)
s.params                                    # default / bounds / shared_key changes per ParamDecl
s.edges                                     # data edges added / removed (from describe())
s.identical                                 # nothing above changed, at IR-hash level (tier 1)

# outputs: over a frame
o = diff.outputs(old, new, frame, params=None, key="application_id", rtol=1e-6, atol=1e-9)
o.rows_compared, o.rows_changed
o.columns["offer"]                          # .changed .max_abs .max_rel .mean_delta .dtype_change .samples
o.added_columns, o.removed_columns
o.first_divergent_step                      # optional: walk node values, as Divergence.on_node_end does
o.table()                                   # polars frame: key, column, old, new, delta

# against git
diff.revisions("main", "HEAD", entry="credit.pipeline:build").outputs(frame)

# proof, tiered
diff.prove(old, new)                        # Proof(tier, identical: bool | None, evidence)
```

```bash
decider diff main HEAD credit.pipeline:build --data loans [--key application_id] [--format text|json]
```

Output-diff semantics are inherited from `Divergence`:
- Floats are compared with `is_close` and `rtol` / `atol`; `NaN == NaN` and `null == null`.
- Everything else is compared exactly.
- `Money(cents=True)` deltas are exact integers, never floats.

A corpus-based diff has three requirements that `checks.compare` doesn't meet today:
- The corpus is built from the **union** of both versions' inputs. Today a new input in the revision makes `compare` raise.
- Each version runs with its own defaults.
- The params documents are diffed as well.

`checks.compare` also uses `assert_equivalent` just to run a flow, which runs every mode on both sides. `diff` uses `Engine().bind(step).run`.

## Can we prove two versions identical?

**Verdict:** arbitrary Python can't be proven. Tier 1 is cheap, so ship it first. z3 (tier 4) is worth adding for trees, tables and expressions. Every result carries the tier it reached, and nothing returns `identical=True` from sampled evidence.

| Tier | Method | Cost | What it proves | Caveats |
|---|---|---|---|---|
| 0 | Same source text | trivial | nothing reliable | This is today's `_signature`. It is **wrong** because it ignores constants, closures, globals and config values. It is also over-sensitive to comments. |
| 1 | **IR content hash.** Per `CallNode`: `fingerprint(fn)` (bytecode, consts, closure, referenced globals; already used for the numba cache key in `engine/compile/fingerprint.py`) + kind, inputs, outputs, ParamDecls (name, annotation, default, bounds, shared_key, on_invalid) + `consts` + param-table content + the tree shape. Paths and names are labels, not hashed. | days | Both versions compute the same thing and bind the same data. Exact for pure steps, which `wall_clock` / `impure` police. | `fingerprint` hashes unknown objects by identity, so equal logic can read as different. That is a false negative, which is the safe direction. |
| 2 | Normalised config documents for trees, tables and scorecards: sorted ranges, merged identical adjacent leaves, defaults filled, then compare | 1–2 weeks | Two documents that differ in layout or format version but not in decisions | needs one normal form per document type |
| 3 | Exhaustive enumeration over small domains. Partition each numeric input at every threshold literal in either version (±ε, NaN, null, ±inf), take bool and code inputs whole, and run the full product if it's under about 1e6 rows. | days (the corpus machinery exists) | Exact for piecewise-constant decision logic (trees, tables) | only evidence for arithmetic steps such as `cents * rate / 12` |
| 4 | SMT (z3, as the `decider[prove]` extra) over trees and tables, which are linear predicates: ask for x where `old(x) != new(x)` | weeks | Complete for flows that are only trees and tables | Breaks on arbitrary Python, strings, rounding semantics, int64 wrap and NaN. Would need the restricted `steps/expr` language. |
| 5 | Symbolic equivalence of arbitrary Python | not feasible | | |

**Recommendation:**
- Build tier 1 now, together with the fix for the `compare` bug. It is also what makes `diff.structure` meaningful.
- Offer tier 3 as an evidence generator: `diff.outputs(..., domain="thresholds")`. Report it as evidence unless every step is piecewise-constant.
- **Build tier 4 with z3**, using `z3-solver` as the optional `decider[prove]` extra. A missing extra makes `prove` stop at tier 3 and say so; it doesn't fail. Scope it to what z3 handles cleanly:
  - tree and table steps, which are linear predicates over inputs;
  - `steps/expr` expressions;
  - integer and comparison arithmetic.

  A step outside that scope makes the proof fall back to a lower tier for the path through it. Worked counterexamples (`old(x) != new(x)` for this `x`) come back as `evidence`, which is often worth more than the proof.
- The same encoding also powers two checks: `unreachable_branch` (DCD201, "is this arm satisfiable?") for trees and tables, and table row overlap and gaps. So z3 earns its place in `checks` as well.
- Park tier 2. Normalised documents are mostly subsumed by tier 4 for trees and tables.

```python
class Proof(BaseModel):
    tier: Literal["ir_hash", "enumeration", "corpus", "none"]
    identical: bool | None      # None = could not prove either way
    evidence: StructureDiff | OutputDiff | None
```
