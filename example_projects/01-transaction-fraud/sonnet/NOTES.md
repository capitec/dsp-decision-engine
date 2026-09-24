# NOTES

## 1. What I built

The real-time instant-payment flow of spec 01, at SCOPE.md's stated
volume: a **generated flat rule set of 521 live and 114 shadow rules**
across all five families (`CF` 214/31, `AT` 112/22, `MS` 96/40, `FP`
41/9, `AA` 58/12 -- spec §6.1's own table, verbatim), each carrying the
full per-rule attribute set (owner-equivalent metadata, family, severity,
action, priority, status, effective dates, event types, segments,
overlay-exemption, criticality, suppressibility, reason code, queue,
stale/absent behaviour). `MS`/`AT`/`AA` rules apply to instant payments;
`CF`/`FP` rules exist at full volume for the applicable-population cost
question (§10 item 15) but are correctly never applicable to this event
type (their real event types -- card authorisation, debit-order dispute
-- are out of this slice's scope).

Built at real depth: the degraded-mode verdict (§5.6, full bitset +
mode), hard blocks (§5.7, six gates with a severity floor), live/shadow
rule evaluation with **structural** isolation (§5.8, §5.10-§5.11), three
of the seven overlay kinds (§5.9, §6.5 -- sensitivity dial, threshold
multiplier, action escalation, including overlay-exemption, stack-off and
composition order), action resolution with critical override,
allow-listing suppression, governance exceptions and the counterfactual
action (§5.12), and reason ranking + client wording (§5.14) through
`core.reason_codes`. Batch/backtest mode (§5.17) reuses the identical
pipeline object -- see `backtest.py` and "Reuse" below for why that makes
the equivalence requirement structural rather than something to prove by
hand, and §4.8 below for why the batch path calls `Executable.score()`
once per event rather than `Engine.run()` on a `DataFrame` (a confirmed
`decider` defect, not a stylistic choice).

**Left out** (SCOPE.md, explicit): dispatch (queue/SLA routing, step-up
challenge selection, SAR triggers, downstream instructions), outcome
feedback (§5.16), wording in more than one language, and the latency
target -- §10 item 15's "cost holds at 900 rules" is tested as a fitness
check (635 rules load and evaluate in well under a second, see
`tests/test_rules.py`), not as a measured p99.

**Scoped down beyond SCOPE.md's own text**, and flagged here rather than
silently: full enrichment (merchant/MCC tables, counterparty tables) is
stubbed as already-resolved request fields, per SCOPE's "stub
enrichment"; velocity **staleness tracking** covers 3 of the spec's 168
aggregates (1-minute count, 10-minute count, 1-hour amount sum -- the
three the spec's own prose calls out by name as the sharpest staleness
examples), not all 168; each rule references at most one tracked
aggregate, not "per referenced feature" as §5.4 literally asks.

---

## 2. Reuse

**From `00-shared-credit-core`** (DEPS.md's exact list, all five used,
nothing extra needed): `core.reason_codes.ReasonCodeRegistry` (built a
200-code fraud registry, ranked through the same `.resolve_step()`
mechanism 00 uses for credit declines); `core.dates.EffectiveDatedSet`
(resolves `rule_set_version` against `decision_date`, the same
effective-dating mechanism 00 uses for table versions -- proof that the
mechanism generalises past "table version" to "any effective-dated
artefact"); `core.adjustments.AdjustmentRegister` (the overlay stack --
built the three required kinds directly against `apply_stack`, not the
convenience `apply_stack_step` wrapper, because that wrapper's provenance
schema is credit's four keys and its own docstring invites exactly this:
"a consumer needing a wider scope builds its own step from `apply_stack`
directly"); `core.consent.consent_verdict`/`channel_permitted` (gates the
client wording step); `core.rounding.round_instalment` (rounds the
backtest's `value_blocked` metric to the cent).

**From `decider`'s built-ins**: `decider.steps.trees.TreeConfig` with a
`prioritized_flat_rule` document and `mode="all"` is the mechanism the
whole rule set runs on -- see §13 Q1 below, this is the headline finding.
`frame_step` for `resolve_firing_set` (joins ~635 boolean columns against
governance data -- a variable-*width* problem, not `frame_step`'s usual
variable-*length* one) and `resolve_action` (same reason: reads the whole
firing-set row, not named columns). `flow(...)` for every same-name
waterfall (`resolve_action` -> `REASON_REGISTRY.resolve_step()`, both
writing `decline_reason_codes` -- the identical pattern 00's
`_reasons_unit()` uses). `param()`/`missing_as()` throughout.

**Written from scratch**: `fraud_interdiction/rules.py` (the rule
catalog, generator and flat-rule-document assembly -- there is nothing in
`00` or `decider` shaped like "a population of independently-governed,
individually-versioned predicates evaluated as a set"), `firing.py` (the
applicability filter and firing-set assembly -- this project's actual
answer to Q5), `action_resolution.py` (§5.12's precedence machinery),
`overlays.py`'s fraud-specific provenance and the three overlay
instances, `hard_blocks.py`, `features.py`'s degraded-mode verdict, and
the local vocabulary (`vocab.py`, spec §4.4's fifteen names).

---

## 3. Gaps in what I consumed

None. DEPS.md states 01's hard dependency as "00 only," and every one of
the five named capabilities worked exactly as `00`'s NOTES.md documents
it, with no missing piece and no workaround needed on my side. The one
adaptation -- writing my own `apply_stack`-based step instead of using
`apply_stack_step` -- is not a gap; it is the documented extension point.

---

## 4. Framework friction

Confirmed with a minimal reproduction before being written up, in the
order I hit them.

### 4.1 A rule id with a hyphen cannot be a tree threshold's param name

Spec §4.4 declares `rule_id` as `string(8)`, e.g. `CF-0412` -- and every
worked example in the spec (`MS-0208`) uses that exact hyphenated shape.
Deriving a per-rule threshold's param name from the rule id
(`f"{rule_id}_amount_thresh"`) fails at bind time, two frames deep in
`decider`'s own params machinery:

```
ValueError: Type names and field names must be valid identifiers: 'CF-0000_amount_thresh'
```

(`collections.namedtuple`, which decider's params bundle is built from,
rejects the hyphen.) Not a defect -- params must be valid Python
identifiers -- but it means the spec's own `rule_id` format cannot be
used verbatim as a `decider` param name, and nothing in `TreeConfig`'s or
`param()`'s docstrings says so. I ended up not needing per-rule params at
all (see 4.2), so this only bit me in a throwaway prototype, but it would
bite for real the moment someone wants individually-retunable per-rule
thresholds via `configs/<version>/params.json` rather than via a new rule
document version.

### 4.2 `TreeConfig` with `mode="all"` is decider's real answer to "is a flat rule set a core component kind" (spec 01 §13 Q1) -- confirmed at volume, with a real limit

`TreeConfig` accepts a `prioritized_flat_rule` document; with
`mode="all"` **every** rule is evaluated (no early exit) and each rule's
leaf value becomes its own `<rule.name>.<output_column>` output column
(`decider/steps/trees/encode.py::_outputs`). I generated and loaded the
full 635-rule estate (521 live + 114 shadow) as one such document and
confirmed both load (~0.07s) and evaluation (~0.24s for 3 rows, 540
output columns) are well within the budget SCOPE.md asks this slice to
prove (§10 item 15, "cost holds at 900 rules") -- see
`tests/test_rules.py::test_live_and_overlay_base_documents_load_and_run`.
This directly answers spec §13 Q1: yes, a flat rule set is expressible on
the framework's own vectorised evaluation path, not as a debug facility.

What it does **not** carry, and should not: owner, family, severity,
action, effective dates, segments, overlay-exemption, criticality,
reason code -- a tree node has no field for any of it, and putting it
there (620 constant-valued output columns, one per attribute) would make
every governance question a data-shape problem. The design that works is
exactly what 00 already does for `ReasonCodeRegistry`/`AdjustmentRegister`:
a plain Python `RuleCatalog` (governance) sits beside the `TreeConfig`
document (shape), joined back together by `rule_id` in `firing.py`. This
is not a framework gap so much as a confirmation that "the answer is a
set with an argmax over it, not a lookup" (spec §2) is a two-layer
problem, and the framework only needs to solve the vectorised-evaluation
layer.

### 4.3 `frame_step` has no `param()` support -- confirmed by reading the runner

A `frame_step`'s function receives **only** the `DataFrame`
(`decider/engine/run/runners/interpreted.py::_frame`: `out = node.fn(df)`
-- no `params` bundle is built or passed for a `"frame"`-kind call,
unlike `"scalar"`/`"row"` calls a few lines above, which do
`bundle = params.bundle(call.id, m)`). `overlays.py` needed a
pipeline-level "is the overlay stack enabled" toggle inside a
`frame_step` (the mule/scam amount multiplier); the natural spelling
(`adjustment_stack_enabled: bool = param(True)` as a second argument,
exactly like `credit_core.adjustments.apply_stack_step` does for its
`step()`-based cousin) is simply not read by the engine. Confirmed by
tracing the call site, not by an error message -- the parameter is
silently never populated. Workaround: a one-line scalar `step()`
(`overlays._stack_enabled`) that reads the `param()` and writes it as an
ordinary column, which every downstream `frame_step` then reads like any
other input. Worth a line in `frame_step`'s own docstring; nothing there
suggests this limitation.

### 4.4 A raw, untouched request field cannot be named in `.emit()` -- and the natural fix is to do nothing

This one cost the most time. The instinct, building a decision record
that must carry `event_id`/`decision_id`/`client_id` (09 §5.15 item 1's
stable identifier, chief among them), is "call `.emit()` on them to make
sure they survive." That is exactly backwards, and both directions
surprised me:

1. `.emit("event_id")` on the pipeline fails, because `.emit()` requires
   the name to resolve to a **step's declared output**:

   ```
   WiringError: fraud_interdiction: emit('event_id'): no step produces
   'event_id' and it is not a declared input column.
   ```

2. The obvious fix -- a one-line identity `step()` that reads
   `event_id` and writes `event_id` -- also fails, and for a more
   fundamental reason: the step's own read of `event_id` is resolved
   *before* the graph considers the step itself a producer of it, so it
   looks self-referential:

   ```
   WiringError: fraud_interdiction/_carry_decision_date: input
   'decision_date' is not produced by any earlier step and is not a
   declared input column.
   ```

3. 00's own workaround for this exact problem
   (`credit_core.evidence.stamp_decision_id`) reads the value through
   `param(required=True)` instead of a column read, which sidesteps (2)
   -- but only works if the caller supplies it via the **params**
   argument. Confirmed directly: `Engine().score({"decision_id": "x"},
   {})` raises `EngineError: 'decision_id' is produced by this pipeline
   and is also a column of the input frame` (a *third*, different error,
   because now the value exists as both a frame column and a param
   target); it only works as `Engine().score({}, {"stamp":
   {"decision_id": "x"}})`. That is unusable for a live per-event id in
   real serving, because `decider/serving/handler.py`'s actual request
   path is `live.executable.score(request, live.params)` -- `live.params`
   is the **fixed, per-deployment** params document, never derived from
   the request body. A per-request value genuinely cannot travel through
   `param()` in the HTTP serving path.

4. The actual fix: **do nothing.** `.score(record, params)` on a single
   record returns *every* column -- every untouched input plus every step
   output -- regardless of what is or isn't in `.emit()`. Confirmed
   independently with a two-line `flow`: `pipeline.run(df)` where `df`
   has an extra `client_id` column that no step reads keeps `client_id`
   in the output unprompted. `.emit()` turns out to matter only for
   `.run()` on a batch `DataFrame`, and only for *intermediate,
   step-computed* values that would otherwise be dropped -- never for raw
   input columns, which survive on both paths without being named
   anywhere. The docstring's own line ("intermediates are dropped unless
   emitted") is accurate but easy to over-read as "anything not emitted
   is dropped." `pipeline.py` ended up emitting only `decision_date` and
   `channel_code` (both already read by other steps, so genuinely
   "declared" -- see 4.5) for readability; `event_id`, `decision_id` and
   `client_id` are left out entirely and still appear in every response.

### 4.5 A `ConfigurableStep` subclass must be imported by the pipeline module, or `decider build`'s own generated boilerplate fails

`decider template`'s own generated `pipeline.py` comment says: "loads
`configs/<version>/tree.json` as a `ConfigurableStep`" and shows
`def build(tree)` with no type annotation at all. Annotating the
parameter with the actual base class, `ConfigurableStep` (which reads as
the more correct, self-documenting choice), fails:

```
Error: config version latest failed to build: RegistryError: 'tree' is
not a registered ConfigurableStep type.
```

The discriminated-union resolver that turns a document's own `"type":
"tree"` into a `TreeConfig` only knows about subclasses that have been
*imported somewhere in the process* -- nothing in `pipeline.py` otherwise
imports `decider.steps.trees`, so the "tree" tag is never registered.
Fix: annotate the parameter with the concrete subclass, `TreeConfig`,
which imports the right module as a side effect and is more precise
anyway. Worth flagging because it directly contradicts what the
template's own generated file implies is sufficient (an untyped or
base-typed parameter), and the error message names the document's type
tag but gives no hint that the fix is an import.

### 4.6 Two `ConfigurableStep`/step outputs sharing a column name is a `dag`-time `WiringError`, not silent -- and it caught a real bug

Loading the overlay-base document's rules with the same output column
name (`fired`) as the live document's -- both trees share the same
`rule_id`s for the 18 overlay-eligible rules, evaluated a second time at
their base threshold -- produced:

```
WiringError: dag 'fraud_interdiction': live_rules and overlay_base_rules
both write 'MS-0003.fired'; use flow(...) to apply them in written order,
the later one winning
```

This is good, not friction: it caught a genuine defect (I would have
silently lost the base evaluation to `flow`'s last-write-wins semantics
had I used `flow` to "fix" it, rather than renaming the output column to
`fired_base`, which is what the base evaluation actually needed).
Included here because the fix (rename, don't reorder) is the opposite of
what the error's own suggested remedy (`use flow(...)`) implies is the
right move for this specific case.

### 4.7 Compiled mode's `str`-literal restriction has a second trigger beyond 00's

00's NOTES.md documents compiled mode refusing to compare a `str` column
to another `str` column. This project hit a related but distinct
trigger: comparing a `str` column to a Python string **literal** inside
the function body (`velocity_completeness_band != "complete"`):

```
ValueError: fraud_interdiction/enrichment_degradation_code: `str` input
'velocity_completeness_band' enters a compiled kernel as a code, so a
literal in the function body would never match it; declare the literal
as a `str` param, e.g. `private: str = param("private")`
```

Clear, actionable, and (unlike most of the above) a documented design
choice rather than a gap -- counted as a limitation, not friction, same
as 00's classification of its own version of this. `SERVE.md` uses
`interpreted` mode for the same reason 00's does.

### 4.8 `Engine.run()` crashes on a batch containing both an empty and a non-empty `list[str]` row -- confirmed with an 8-line, pipeline-free repro, and it is exactly backtest mode's normal case

Found running `tests/test_pipeline.py::test_batch_and_real_time_paths_agree`
(three sample events, one of which fires no overlay-induced rules, so
`fired_on_overlay_ids` is `[]` for that row and non-empty for the other
two). `Engine().bind(pipeline).run(df, params)` panics inside `polars`,
called from `decider`'s own result-materialisation path:

```
decider/engine/run/state.py:182: in _series
    return pl.Series(name, (values if valid is None else np.where(valid, values, None)).tolist())
...
pyo3_runtime.PanicException: called `Result::unwrap()` on an `Err` value:
SchemaMismatch(ErrString("invalid series dtype: expected `String`, got
`object` for series with name ``"))
```

Isolated to an 8-line reproduction with no pipeline, no `credit_core`,
one `frame_step`:

```python
@frame_step(reads=["x"], writes=["y"])
def f(df):
    return df.with_columns(pl.Series("y", [["a","b"], [], ["c"]], dtype=pl.List(pl.Utf8)))
Engine().bind(f).run(pl.DataFrame({"x": [1, 2, 3]}), {})
# same PanicException
```

Two confirming variants, both fine: the identical ragged shape with
`list[int]` instead of `list[str]` (`[[1,2], [], [3]]`) works; the
identical `list[str]` shape with no row empty (`[["a","b"], ["z"], ["c"]]`)
also works. The bug is specifically "a `list[str]` output, at least one
row length zero, materialised through `Engine.run()`" -- not `frame_step`
in general (00's list-of-struct finding), not ragged lists in general
(the `list[int]` case), and not present in `Engine.score()` on a single
record (there is only ever one row, so the "different lengths across
rows" code path that panics is never entered).

This is not a corner case for this project: `fired_rule_ids`,
`fired_on_overlay_ids`, `counterfactual_fired_rule_ids` and
`shadow_fired_rule_ids` are all `list[str]`, and spec §5.18 states
outright that most rules never fire on most events -- a real 90-day
backtest batch (§5.17) is *certain* to mix empty and non-empty rows on
every one of those four columns. Since editing `decider` internals is out
of scope, `backtest.py` calls `Executable.score()` once per event through
one bound pipeline instead of `Engine.run()` on the whole `DataFrame` --
the identical rule-evaluation path, just entered through the per-record
door instead of the batch one -- and assembles the results into a
`DataFrame` in plain Python afterwards, which never touches the buggy
code. This costs `decider`'s own kernel-level batch vectorisation, which
is exactly what a 420M-event (90-day) backtest run (§5.17,
§8's 90-minute budget) would need back; SCOPE.md skips the latency and
batch-throughput targets for this slice, so the workaround is acceptable
here, but it is a real blocker for §5.17 at its stated scale until fixed
upstream.

### Smaller things

- `decider build`'s `_warm()` synthetic-record bug (00 NOTES.md 4.1:
  `date`/`list` top-level inputs get a bogus float `1.0`) reproduced
  identically in this, a second, independently-built project with its
  own `decision_date: date` and `client_segments: list[str]` inputs --
  strengthening 00's own prediction that every project in this set would
  hit it. Same workaround (`inference.py` monkeypatches `_warm` to use
  `sample_request.json`).
- Both this project and `00-shared-credit-core` ship a top-level
  `pipeline.py` (the brief's required filename for every project). Two
  projects on the same `PYTHONPATH` therefore have an ambiguous `import
  pipeline` unless the consuming project's own directory takes priority.
  `decider`'s own serving path gets this right for free
  (`sys.path.insert(0, code_path)` in `handler.py`), but a test harness
  or any other script manipulating `sys.path` by hand has to reproduce
  that ordering itself (see `tests/conftest.py`) or silently import the
  wrong `pipeline.py`.

### What worked well

`TreeConfig`'s `mode="all"` at real rule-set volume, genuinely -- this is
the headline finding, see 4.2. `core.adjustments.AdjustmentRegister.
apply_stack`'s "build your own step" extension point worked exactly as
its docstring promises, with a completely different provenance schema
than its designed-for consumer. `dag()`'s same-column `WiringError` (4.6)
is a real safety net, not friction. `ComputedFeature` (a tree condition's
feature can be an arbitrary expression over columns, e.g. `"amount -
8000.0 * mule_scam_amount_multiplier"`) is exactly the mechanism needed
to make a rule's *shape* stay fixed (§5.10: "its shape is not a
tunable") while its *effective threshold* varies per record under an
overlay -- not documented anywhere as "how to combine trees with
adjustments," but it composes cleanly once found.

---

## 5. Spec problems

- **Family precedence (§5.12 item 5) is a required tie-break input the
  spec never actually states.** The action-resolution rung reads
  "severity descending, then priority ascending, then family precedence,
  then `rule_id` ascending" but §5.12 never publishes what that order
  *is*. I chose AML-adjacent > mule/scam > account-takeover > card fraud
  > first-party fraud (regulatory families senior), which is defensible
  but invented, not read from the spec.
- **SCOPE.md's "cover one event type family (instant payments)" doesn't
  say what to do with `CF`/`FP` rule volume.** §6.1's per-family table
  (which SCOPE says to keep "at volume") includes families (`CF`, `FP`)
  whose real predicates reference card-authorisation/debit-order fields
  that don't exist on an instant payment. I generated them at full count
  but structurally inapplicable (never in the applicable population),
  which honours both "keep the volume" and "one event type" -- but a
  different implementer could reasonably have dropped them to 0 and
  under-counted the 521/114 total, or built them against fabricated
  instant-payment-shaped predicates and over-stated what they represent.
- **Whether a hard block also floors the *counterfactual* action is
  unstated.** §5.12 says overlays' effect is what the counterfactual
  strips out; hard blocks (§5.7) are a different mechanism entirely
  ("short-circuit the action... no rule and no overlay can soften
  them"). I floor the counterfactual with the same hard-block forced
  action as the live one (hard blocks aren't overlay-related, so
  stripping the overlay stack shouldn't touch them), but the spec never
  says so explicitly.
- **Per-referenced-feature stale/absent behaviour (§5.4) vs. per-rule.**
  The spec is explicit that a rule declares its behaviour "per referenced
  feature," implying one rule could declare *different* behaviour for
  two different stale aggregates it reads. This slice's rules reference
  at most one tracked aggregate each, so the distinction never actually
  bites, but a rule with two velocity references and two different
  declared behaviours is a real spec requirement this slice doesn't
  exercise.

---

## 6. What I would do next

1. Track staleness for all 168 velocity aggregates, not 3, and let a
   rule declare stale/absent behaviour per individual reference rather
   than once for the rule.
2. Extend the overlay mechanism to the other four kinds (severity shift,
   scope restriction, queue reroute) -- severity shift is the most
   interesting, since it changes the *tie-break*, not a threshold, and
   would need its own path through `action_resolution.py` rather than a
   `ComputedFeature`.
3. Push `frame_step`'s missing `param()` support upstream (4.3) instead
   of routing every pipeline-level toggle through a materialised column;
   it is a small, general fix (thread the same `bundle = params.bundle(...)`
   call `_call` already does for scalar/row kinds into `_frame`) that
   removes a real workaround from every future `frame_step`-heavy
   project.
4. A proper segment-definition table (`DecisionTableConfig`, effective-
   dated) rather than the eight hardcoded segment names -- §6.4 lists it
   as a real, analyst-edited, monthly-cadence table with 46 rows; this
   slice's `ALL_SEGMENTS` tuple is a stand-in.
5. Wire a second event type (card-not-present, 111) through the same
   rule set to prove the `CF` family's rules, currently inert, actually
   catch something -- and to test whether applicability filtering still
   costs nothing when two event-type populations diverge within one
   rule set.
6. Report 4.8 (`Engine.run()` panicking on a mixed empty/non-empty
   `list[str]` batch column) upstream with the 8-line repro, and once
   fixed, switch `backtest.py` back to `Engine.run()` on the whole
   `DataFrame` -- the per-event `.score()` loop this slice uses is
   correct but forgoes the kernel-level batch vectorisation §5.17's
   90-minute, 420M-event target actually needs.
