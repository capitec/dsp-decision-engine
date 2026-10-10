# Hidden state in steps

Status: proposal, 2026-10-09. Nothing in this folder is built. Read it before changing how steps read data that is not an input, a param
or a config document, or before changing `decikit/_engine/compile/fingerprint.py`.

## Purpose

A step that reads a Python object it did not receive (a register in a closure, a module-level dict, an instance) cannot be compiled by numba,
cannot be compared by `decikit.lab.diff`, and is invisible in config. The 12 shipped examples do this 113 times. This spec records what is wrong,
the owner's position, the design that replaces it, the checks the decider toolkit will add, and the order of work.

## Owner's position

- Steps must not read hidden objects. Data lives in config documents, params, tables, or in a `ConfigurableStep`/`CustomFunction` whose fields are
  the data and whose `to_ir` calls a plain function.
- The decider toolkit does not learn to digest arbitrary objects. A content digest of an instance without its class code would let a method edit read as
  "unchanged"; with the class code it is more machinery than the design needs.
- `@allow_fallback` stays, is discouraged, and must not silence a hidden-state warning.

## What is wrong today (verified by running; audit on `examples/final`)

| Finding | Effect |
|---|---|
| A closure cell holding an object was fingerprinted as `<empty>` (a `except ValueError` swallowed strict mode's error) | `diff.structure` said "identical" after an overlay changed from -18 to -36. 82 of 94 object-reading steps are closures. Fixed by the stable-fingerprint change. |
| A module data attribute read as `mod.K` is not fingerprinted | numba's disk cache serves the old value after `K` is edited; `from mod import K` is correct. 79 example steps read constants this way. |
| The fingerprint walk descends into stdlib and site-packages | 71 steps flagged for no reason (`statistics.mean`, `dataclasses.asdict`, `pl.lit`). |
| A scalar step declared with `**kwargs` crashes compiled modes | raises instead of falling back; works interpreted. |
| Every Python step in the examples is `@allow_fallback` (246 of 246) | `strict_compile` is already silenced, so a check built on the fallback path flags nothing. |

Inventory: 903 call steps, 113 read hidden objects (94 objects, 21 dict/list globals); 111 of them already run in Python. Four functions in
`shared-credit-core` explain about 76: `adjustments.py` `apply_stack_step` (23 steps), `dates.py` `EffectiveDatedSet` resolvers (39),
`reason_codes.py` `ReasonCodeRegistry` (12), `obligations.py` `TREATMENT_MATRIX` (9 across 8 examples).

## Target design

```python
def cap(x: float, limit: float) -> float: return min(x, limit)        # plain, compilable

class Cap(CustomFunction):                      # fields are the data; `impl` is a staticmethod
    type: Literal["cap"] = "cap"
    x: str                                      # column to read
    limit: Value[float] = 100.0                 # tunable value, also settable from params
    @staticmethod
    def impl(x: float, limit: float) -> float: return min(x, limit)
```

Same data twice diffs identical; a changed value diffs different; the step compiles.

| Today | Becomes | Proven (run) |
|---|---|---|
| `AdjustmentRegister` closed over by `apply_stack_step` | `AdjustmentStack(ConfigurableStep)`: fields are the overlays (scope, window, stack position, effect, tighten-only); `to_ir` passes them as consts; governance fields (owner, approval, rationale, review date) are not behaviour | 90 records equal in interpreted and fused; shift -18 -> -36 diffs different; rationale edit does not |
| `ReasonCodeRegistry` | `ReasonCodes(ConfigurableStep)`: `version`, `codes` | rank, unknown-code error and a changed severity all as before |
| `EffectiveDatedSet` | `EffectiveVersions(ConfigurableStep)`: `family`, `date_field`, `versions`; gaps and overlaps rejected at load | cut-over moved a month diffs different |
| `retail_credit.caps` register of lambdas | one small step per rule, in flow order; thresholds are literals in each rule; the appetite dict becomes a `DecisionTableConfig` | 1500 random applications equal; swapping two rules diffs different |
| `RateCardIndex` / `CreditLifeIndex` objects | a `param_table` read with `np.searchsorted`; compiles | 2000 lookups equal; a diff of a 63,360-row table costs about 1.9 s |
| dict/list module globals (`TREATMENT_MATRIX`, ...) | tuple of pairs, `param_table`, or `DecisionTableConfig` | a tuple global is seen by the diff |

No clean design yet: consolidation's stateful `Capacity`/`Buffers`/`ScenarioEvaluator` (a rewrite of the search around `each`/`optimise`),
the fraud `RuleSet` frame step (needs consts on frame nodes), and the governance harness (stays Python, takes everything as params, flows resolved
through an entry-point group).

Compilation does not change: these steps take a `date` or return `str`/lists, so they stay Python. The gain is diffability, visibility in config and
an honest warning, not speed.

## Checks the decider toolkit adds

1. Fingerprint fixes first: module constants keyed by value (the stale-cache bug), library callables keyed by module and qualname without
   walking into them, `**kwargs` steps fall back.
2. `hidden_state(fn)`, built from the same walk as the cache key so the two cannot disagree, called once per step at `Engine.bind` in every mode
   (0 to 6.5 ms for a whole pipeline; nothing per call). It emits `HiddenStateWarning`, which `@allow_fallback` does not silence.
   Warning text: names the step, the object and where it is read, says why it matters (the diff cannot compare it, numba cannot compile it) and what to do.
3. Later, once the examples are clean: raise under `strict_compile=True`; and a CI test that no example emits the warning.

Alternatives considered: warn only (chosen first), raise under `strict_compile`, raise always (every example would fail to build), do nothing
(the two correctness bugs stay).

## Order of work

0. Framework (branch `r3/hidden-state`): items above. Gate: the 12 examples, 247 tests.
1. `shared-credit-core`: add `AdjustmentStack`, `ReasonCodes`, `EffectiveVersions`; delete the three registry classes; convert `TREATMENT_MATRIX`.
   About 76 of 113 flagged steps disappear. Step names and params paths do not change, so params documents and stored evidence keep their digests.
2. One task per example, in parallel (affordability, business-nested, business-credit-e2e, campaign-trees, collections, credit-limit-management,
   loan-granting-pricing, retail-credit-e2e, transaction-fraud), then governance-replay and consolidation last.
3. Re-run the inventory: target 0 flagged steps except declared Python boundaries.

## Risks and open questions

- `credit_core` is imported by 11 examples; its change is the largest risk, so it lands and passes the gate alone before the others start.
- A diff of a large `param_table` default is slow; a faster digest is a separate task.
- Whether mutable containers of literals should be flagged (the stable-fingerprint change already keys exact literal containers by value).
- The engine rework (`2026-10-09_flexible_engine_rework`) defines per-engine loudness for fallbacks; the hidden-state warning is a design lint and is
  emitted in every mode. Confirm that split with the owner.
