# NOTES

## 1. What I built

Campaign 23 (Flex Loan pre-approved top-up), spec 04 §5.3.3's own worked example, end to
end: population/suppressions (§5.2), the tree with full path capture (§5.3-§5.4), the
threshold-shift and cap-reduction overlays over a *published, unedited* tree (§5.3.4),
pre-assessment (§5.5, reusing project 03's published solve and pricing), holdout and
universal-holdout assignment (§5.7), and one assignment record per (client, campaign)
evaluation (§5.8). `decider build` succeeds and the sample request scores through the
handler unchanged (see SERVE.md; verified with both the handler script and the CLI's own
`decider build`, and independently re-verified with the warm-up workaround removed to
confirm it is load-bearing -- see "Framework friction" #3).

**The layer underneath the tree, not just the tree.** The hardest requirement in the spec
(§5.4.3, "node identity... stable across versions") and the second-hardest (§5.4.1, "the
path is an output, not a by-product") are both things `decider.steps.trees.TreeConfig`
does not give you natively (see "Framework friction" #1). `campaign_trees/tree_model.py` is
therefore the actual centre of this project: trees are authored once, in a small internal
representation, and two things are derived from that *one* authored tree --
`to_v3_document()` (what `TreeConfig` runs, at 400M-evaluation vectorised speed) and
`walk()` (a pure-Python traversal used both to capture a path live and to re-render one
later from stored feature values alone, §5.4.1(d) and §9.1, with no decider dependency and
no re-run). `tests/test_tree_model.py::test_walker_agrees_with_treeconfig_leaf` proves the
two traversal engines answer identically over a battery of records -- the check this
two-implementations risk earns.

**Node identity** (`tree_model.node_key`) is content-derived: a node's own condition
(canonicalised, term-order-independent) plus the set of (parent `node_key`, branch
direction) pairs that reach it -- so a node with two parents (campaign 23's own node 6,
reached from both node 3 and node 4, §5.3.3's own note) keeps one identity, an unrelated
edit elsewhere leaves it untouched, and an edit to *this* node or one of its ancestors
changes it. `tests/test_tree_model.py::test_node_identity_map_isolates_the_one_changed_node`
publishes `TREE_V2` (node 6's floor moved 2 200 -> 2 600, spec 04 §11 scenario 1/15) against
`TREE_V1` and asserts node 6 is classified `changed`, everything upstream of it is
`carried_forward`, and its own downstream (which legitimately inherits the identity change,
since its *position* -- "reached via node 6" -- genuinely changed) is `added`/`removed`. This
is a design choice, not the only one the spec's own §13 Q5 admits is possible -- see "Spec
problems" below for the trade-off it makes.

**Overlays** are two different mechanisms sharing one governance shape (`TreeOverlay`,
`overlays.py`): a threshold shift (volume dial, cut-off shift) moves through `TreeConfig`'s
own per-call `params={...}` mechanism (`pipeline.py` runs the tree *twice*, once with the
overlay's params and once with the published defaults, to get `leaf` and `unadjusted_leaf`
from the same executable -- §5.3.4 requirement 2, §5.14.3's "one implementation"); a cap
reduction moves through `credit_core.adjustments.AdjustmentRegister` unmodified, whose
`cap_adjustment` kind is already declared tighten-only for `multiply <= 1.0` -- exactly
"amount overlays may only reduce" (§5.5 requirement 6), enforced at *definition* time, for
free. Both worked examples in the reused table (§5.3.4: node 6's floor 2 200 -> 2 600; a
one-notch cut-off tightening) are built with real effective-date windows, so the sample
request (`cycle_date` 2026-09-24) exercises one in force and one not-yet-effective at once.

**Suppressions** (§5.2) are a 9-of-34 subset covering both classes -- absolute (removes a
client from evaluation entirely) and measurement-relevant (evaluated anyway, path recorded,
because "the analytics team reports the population that would have been targeted") -- with
every suppression that applies recorded, not merely the first.

**Pre-assessment** (§5.5) reuses project 03's actual published solve and pricing
(`loan_granting.solve.solve_term`, `loan_granting.pricing.PriceEvaluator`/`RateCardIndex`)
unmodified, and `core.appetite` (project 00) for the amount ceiling. The one piece that
would otherwise require importing project 02 (not a declared dependency of 04 -- DEPS.md:
04's hard deps are 00 and 03 only) is approximated: `ponytail:
max_affordable_instalment is estimated_discretionary_income * a fixed serviceability ratio,
not project 02's real affordability verdict via a real 03 batch run; upgrade path is to
replace preassessment.py's estimate_max_affordable_instalment with a read from 03's actual
stored batch output once that artefact exists`. Everything else in the §5.5 "Emits" contract
(amount, term, binding constraint, `risk_grade`, validity window) is real arithmetic.

**Holdout/control** (§5.7) is a `sha256`-based stable-hash assignment (not Python's own
`hash()`, which is per-process salted and would make "the same client lands in the same
group in March, April and May" false) -- deterministic from identifiers alone, re-derivable
with no stored table, proven by `tests/test_holdout.py` to move only the clients a design-
version bump should move.

**Arbitration** (§5.6) is deliberately *not* a `decider` step -- the spec says outright "this
cannot be decided one client at a time", and a per-row step (compiled or interpreted) has no
cross-row view. `arbitration.py` is plain Python (ranking-and-cutting, one of the four
mechanisms the spec explicitly permits without prescribing one) over a batch of
qualifications, proven separately in `tests/test_arbitration.py`: determinism,
re-runnability under shuffled input, channel capacity never exceeded, and a reason for every
non-contact. SCOPE.md explicitly cuts fairness reporting (the 35% rolling-cycle rule, the
60% rank-one-demand floor) from this slice; what's built is §5.6 requirements 1-4.

**Scale.** `generate_trees.py` generates ~59 more trees (campaign 23 is the 60th,
hand-built), 20-400 nodes each, from a synthetic feature pool, seeded and deterministic.
Validated with the real §5.9 validator this project also uses for campaign 23 -- and, left
un-retried, the generator legitimately produces self-contradictory nodes, which the
validator catches for real (`tests/test_generate_trees.py::test_validator_catches_generated_contradictions`),
not only against a hand-built fixture. Generation + validation for 59 trees runs in well
under a second, against the spec's 4-minute prep budget (§8) for 60.

### What I left out

- **59 of 60 trees are not served.** `pipeline.py` wires campaign 23 end to end; the other
  59 are generated and validated but not wired into `decider build`'s one served pipeline --
  exactly the choice 00 and 03's own demo pipelines make ("a representative slice, not
  everything built"; both their NOTES.md say so explicitly). Serving 60 different trees from
  one endpoint would mean picking a tree per request based on a row value, which `decider`'s
  static per-call binding doesn't support cleanly (see "Framework friction" #2) -- in
  production this is 60 campaign-scoped batch/serving deployments, one tree each, not one
  endpoint dispatching on `campaign_id`.
- **The 400M-evaluation/6-hour timing and the 120GB path-artefact budget** are not measured
  -- no compute at that scale here. What's proven instead: the *mechanism* (path capture,
  node identity, validation, overlays) works correctly at the spec's structural scale (60
  trees, ~9 200 nodes), and prep cost for that scale is far inside budget.
- **§5.9 validation, partial.** Built: no unreachable nodes, no cycles, contradiction
  detection (AND-only, single-feature numeric paths -- see the docstring in
  `tree_model._check_contradictions` for exactly what's out of scope: an OR node, or two
  *different* features compared against each other, is not interval-representable by this
  checker), unknown/prohibited feature references. **Not built**: population-impact
  estimation against a snapshot (§5.9 item 7) and overlay re-applicability on republish
  (§5.9 item 8) -- both need a live snapshot or a live overlay register at validation time,
  which this project's validator doesn't take as input. `overlays.overlay_errors` does the
  adjacent, run-time half of item 8 (an overlay past review fails the *cycle*), just not at
  *publication* time.
- **Dispatch, fairness reporting, holdout/challenger design beyond the hash mechanism** --
  explicitly cut by SCOPE.md.
- **The regulator pack (§9.2) and the client-dispute answer (§9.1) as rendered outputs** --
  the *mechanism* both depend on (`tree_model.walk`/`render_path`, re-derivable from stored
  features with no decider dependency) is built and tested; the pack/answer formatting
  itself is not.
- **Card campaigns from project 07** (§11 scenario 13, explicitly "a future change" per
  DEPS.md) -- out of scope.

---

## 2. Reuse

**From project 00 (`credit_core`), unmodified:** `credit_core.appetite.build_appetite_table`
(the amount ceiling every pre-assessment is bounded by, §4.3's own named match);
`credit_core.eligibility.decline_reason_codes_step`/`is_eligible_step`; `credit_core.consent.consent_verdict_step`/
`channel_permitted_step`; `credit_core.adjustments.AdjustmentRegister`/`Adjustment`/`AdjustmentEffect`
(the cap-reduction overlay -- its `cap_adjustment` kind's tighten-only rule reused as-is);
`credit_core.rate_card.generate_flex_loan_card` (via `generate_configs.py`, exactly 00/03's
own config-generation pattern).

**From project 03 (`loan_granting`), unmodified:** `loan_granting.solve.solve_term`,
`loan_granting.pricing.PriceEvaluator`/`RateCardIndex`/`CreditLifeIndex` -- the two entry
points 03's own NOTES.md names as published specifically for reuse ("03 §11 items 11-12"),
called here exactly as 03's own `granting.py` calls them internally, over a synthetic
affordability input (see "What I left out" and the `ponytail:` note above for the one gap).

**From `decider`'s built-ins:** `decider.steps.trees.TreeConfig` (twice per record, `.named()`
+ `.relabel(writes=...)`, matching 00's own "same capability, different settings" pattern --
`test_affordability.py::test_same_capability_twice_with_different_settings_in_one_flow` --
applied here to a tree instead of a scalar capability); `decider.steps.tables.DecisionTableConfig`
(inside `credit_core.appetite`, not rebuilt); `frame_step` for pre-assessment (per-application,
multi-phase, not vectorisable into one expression -- 03's own `solve.py`/`waterfall.py`
precedent for the same reason); `param()`/`missing_as()` throughout; `.relabel()` extensively
for the tree pair's parallel adjusted/unadjusted output columns and for effective-dating-style
column renames (00's own "column-naming convention" pattern, recurring here once per tree
instance instead of once per table).

**Written from scratch:** `tree_model.py` (node identity, path capture/rendering, §5.9
validation -- the reason this module exists at all is in "Framework friction" #1);
`campaign23.py` (the one hand-authored tree, faithful to §5.3.3); `generate_trees.py` (the
scale generator); `suppressions.py`, `overlays.py`'s `TreeOverlay` (see "Framework friction"
#4 for why it isn't built on `credit_core.adjustments.Adjustment`), `holdout.py`,
`arbitration.py`, `preassessment.py`'s frame_step wiring and its one approximated input.

---

## 3. Gaps in what I consumed

- **00's `credit_core.adjustments._TIGHTEN_RULES` is closed** (five declared kinds:
  `score_shift`, `odds_multiplier`, `rate_addon`, `cap_adjustment`, `buffer_adjustment`).
  This project's two threshold-shift overlay kinds (`volume_dial`, `cutoff_shift`) are "set a
  tree param to a new literal value", which fits none of them -- constructing
  `Adjustment(kind="volume_dial", tighten_only=True, ...)` raises `ValueError` at definition
  (`_TREE_OVERLAY_DIRECTION` has no entry `_is_tightening` recognises). I could not edit 00's
  dict (BRIEF: never edit another project's directory), so `overlays.TreeOverlay` is a small,
  independent dataclass with the same identity/governance fields and its own tighten-only
  check. **This is the same wall project 03 hit** over its own "boundary shift" and "scaling
  change" overlay kinds (03 NOTES.md "Gaps in what I consumed") -- two independent projects
  needing the same closed registry extended, confirmed a second time.
- **No project 03 batch output file to read from.** DEPS.md sanctions stubbing 03's batch
  pre-assessment for exactly this reason ("04 moves to wave 2"); this project takes the
  middle path DEPS.md doesn't quite name -- real reuse of 03's *solve and pricing*, fed by
  an approximated affordability input, rather than either a full re-import of 03's whole
  pipeline (which would mean also importing project 02, not a declared 04 dependency) or a
  pure invented-number stub. See "What I built" for the exact `ponytail:` line.

---

## 4. Framework friction

### 4.1 `TreeConfig`'s `path_output` gives only the terminal leaf, not the path

The docstring is explicit about this once you read it closely: "`path_output` names an extra
String column holding **the id of the leaf that answered**" -- not the ordered sequence of
nodes visited. Spec 04's central, load-bearing requirement (§5.4.1: "the path each client
took through each tree must be captured, stored, and queried afterwards... a system that
produces correct answers and no paths is, for this business, a failed system") needs the
traversal itself. There is no framework-level way to get it from `TreeConfig` -- no
`trace_output`, no per-node visited flag, nothing in `decider/steps/trees/` exposes
intermediate node evaluation. I confirmed this by reading `decider/steps/trees/schema/v3.py`,
`nodes.py` and `tree.py` directly (not just the docstring) before concluding there was no
supported extension point, which is more of the framework's internals than the BRIEF asks
for but was the only way to be sure. The workaround -- `tree_model.walk()`, a second,
pure-Python traversal engine over the same authored structure -- is real, tested-for-
agreement code, not a five-minute shim, and it exists specifically because the framework's
own tree kind doesn't give this project the one thing spec 04 says a tree kind must give it.
Spec 04 §13 Q17 asks outright whether "the decision-tree core kind survive[s] this project
unchanged" -- the honest answer, from the framework's actual behaviour rather than from
guessing, is **no**: path capture at the granularity §5.4.1 demands is not a property of
`decider`'s tree kind today, and any consumer with the same requirement will hit this.

### 4.2 A `TreeConfig`'s `params={...}` override is call-scoped, not row-scoped

This cost a real redesign. My first attempt made the overlay's resolved threshold a *decider
step output* (a dict, computed per row from `campaign_id`/`cycle_date` against the overlay
registry) and tried to feed it into the tree step somehow -- there is no such feed-in point.
`TreeConfig.run(df, params={tree_name: {param_name: value}})` (the docstring's own example,
and 00's `.named()` precedent) applies **one** params document to the **whole** call, not a
per-row value pulled from a column. This is, on reflection, the *architecturally correct*
shape for spec 04's own semantics -- "the stack is resolved by `cycle_date`, never by today"
(§5.3.4 requirement 7) means one cycle has one resolved stack, applied uniformly, which a
call-level params document models exactly -- but nothing in `TreeConfig`'s docstring states
this constraint up front, and the natural first instinct (an overlay depends on
`campaign_id`, which is a row value, so surely it's a per-row thing) is wrong for this
mechanism specifically, in a way that only becomes visible once you try to wire it and
`emit()`/`dag()` give you nothing that looks like an error, because a dict-typed step output
is entirely valid decider on its own -- it's just useless for actually parameterising the
tree next to it. `pipeline.py`'s `overlay_stack_id_param` docstring records the corrected
design; `generate_configs.py` now does the per-cycle resolution the params document should
carry.

### 4.3 `decider build`'s warm-up still can't handle a `date` input

Exactly 00 and 03's own finding (`_DUMMY = {bool: False, int: 1, str: "", bytes: ""}`,
falling back to the float `1.0` for everything else, `decider/serving/handler.py::_warm`),
confirmed a third time independently: `cycle_date: date` (mandatory here per 09 §5.15 item
4, "no reliance on today", exactly as it is in 00 and 03) makes `decider build` crash before
serving anything (`TypeError: unsupported operand type(s) for +: 'float' and
'datetime.timedelta'`, from `preassessment.py`'s `cycle_date + timedelta(...)`) unless
`inference.py` replaces `_warm` with one that warms from `sample_request.json`, the same
workaround 00/03 document. I verified this is load-bearing, not cargo-culted, by temporarily
removing `inference.py` and re-running `decider build` -- it fails with exactly that error --
before restoring it. Three implementer sessions across three different projects hitting the
identical bug independently is a strong signal this belongs fixed upstream, not worked
around a fourth, fifth and sixth time.

### 4.4 `emit()` can only select a column a step produced, never a raw input directly

`dag(...).emit("client_id")` fails with `no step produces 'client_id' and it is not a
declared input column` even though `client_id` is very much a declared input `decider`
itself required at bind time. An evidence record that must carry identifiers straight
through untouched (`assignment_id`, `client_id`, `campaign_id`, `cycle_id`, `cycle_date`,
`tree_version` here) needs one trivial identity step per field
(`pipeline.py::_echo_step`) purely to make it re-emittable. This is a small, easily-worked-
around gap, but it is the kind of thing every project in this set with a pass-through
identifier is likely to hit once and solve slightly differently -- worth a one-line
`emit()` extension (or documented convention) upstream rather than five different
`_echo_step` helpers across five projects.

---

## 5. Spec problems

- **§5.3.3's own leaf count disagrees with its own table.** The prose says "seven leaves";
  the table that follows it lists eight (901-904, 910-913). This project's tree has eight,
  matching the table, which is also what makes the worked example's own leaf 912 exist.
- **Stage numbering implies an order the worked example itself violates.** §5.3's Stage 3
  (the tree) is numbered before §5.5's Stage 5 (eligibility and amount), but campaign 23's
  own worked-example tree tests `pre_assessed_amount` *inside* the tree, at nodes 7 and 8 --
  which means Stage 5 must run before Stage 3 for this campaign, not after. `dag()` resolves
  the true dependency from each step's reads/writes rather than from the stage numbers, so
  this project is simply correct regardless -- but a reader following the spec's own stage
  order would build the pipeline in the wrong sequence and only discover the problem when a
  tree node reads a column that doesn't exist yet.
- **`unadjusted_leaf` (§4.4) names one axis; this project's tree actually has two.** A
  threshold-shift overlay changes which *leaf* is reached (§5.3.4's `unadjusted_leaf`); a
  cap-reduction overlay changes the *amount a leaf's tier advertises*, independently, on top
  of whichever leaf was reached. §4.4 declares one `unadjusted_leaf` field as if there's one
  overlay axis per evaluation; this project needed `unadjusted_leaf` (threshold-shift off)
  *and* `tier_ceiling_before_cap_overlay` (cap-reduction off) as genuinely separate values,
  because they're different overlay mechanisms over different targets that can each be on or
  off independently. Not a contradiction, but the vocabulary under-specifies a case the
  spec's own §5.3.4 table (which lists both overlay kinds) implies will occur together.
- **Node identity's "changed" cascade is a real, spec-flagged ambiguity, not a bug I found
  and didn't fix.** §13 Q5 asks outright "what exactly makes two nodes 'the same test'...
  who decides, and is the decision reviewable?" This project's answer -- a node's identity
  depends on its own condition *and* the (parent identity, direction) pairs that reach it --
  means a node whose *own* test is untouched can still get a new identity purely because an
  ancestor on the path to it changed (worked example: node 6's edit cascades a new identity
  to node 8, even along node 8's *other*, untouched incoming edge from node 5). The
  alternative (identity independent of ancestor changes entirely) would under-flag: two
  nodes that are textually identical but now sit under genuinely different logic would read
  as "the same test", which is a worse failure for an audit system than over-flagging. I
  judged the conservative direction correct and documented the trade-off rather than
  resolving it, because the spec itself says this decision should be reviewable, not
  silently made once in code.

---

## 6. What I would do next

1. Wire all 60 trees (the 59 generated ones plus campaign 23) into a genuine batch driver --
   one `decider` bind per campaign, run over a shared synthetic population -- to measure
   evaluation throughput against the spec's 18 500/s target, which this project's scale
   tests (structural, not throughput) don't touch.
2. Replace `preassessment.py`'s approximated `max_affordable_instalment` with a real read
   from project 03's stored batch output (or a real call into 03's whole pipeline), once
   that artefact exists in this repository -- the `ponytail:` line names exactly this as the
   upgrade path.
3. Extend the §5.9 validator's contradiction check past AND-only, single-feature paths (an
   OR node, or two different features compared against each other, e.g. via an interval
   solver over a small constraint system) -- the current checker is honest about what it
   does and doesn't catch, but a real validation pipeline would want the fuller case.
4. Build the node-level volume/response reporting (§5.4.2) and the dead-branch detection
   over two cycles' worth of stored paths -- the path artefact and the node identity map
   this project builds are both inputs to that reporting, which itself isn't built.
5. Raise the four framework-friction findings above (especially #1 and #2, which cost real
   design time to discover) as issues against `decider` itself, alongside 00 and 03's own
   findings on the warm-up bug -- three to six independent confirmations of the same few
   gaps is worth fixing once upstream rather than working around indefinitely.
