# decider2 against the decision-engine landscape: features worth having and what to change

Date: 2026-09-21. Companion to [decision-engine-landscape.md](decision-engine-landscape.md)
(the survey of 72 engines, 2026-09-20).

## 0. Purpose and how to read

Five comparison reports were written against the survey, one per engine group, each
reading decider2's docs and source (`feature/decider-v2`, including uncommitted edits)
and the engines' own documentation or source. This note consolidates them. It carries
three things the reports were asked for:

1. **Actionable recommendations** (§2), de-duplicated across all five reports and grouped
   by theme. Every item names the decider2 module or doc it touches, an effort
   (S = days, M = 1 to 3 weeks, L = a month or more, one engineer) and the engine feature
   that motivates it. Item tags such as `[G7]` point at the numbered recommendation in
   the source report: G = GoRules, C = credit platforms, F = fraud engines,
   O = open-source and DMN, B = enterprise BRMS.
2. **A feature inventory** (§3): the useful features the field has, who has them, and
   whether decider2 has them.
3. **The GoRules verdict** (§4), because it was singled out as promising.

The full reports are in [comparisons/](comparisons/). They contain per-engine detail,
`file:line` citations, URL citations, evidence grading, and the unverified lists. This
note cites them rather than repeating the evidence.

| Report | Lines | Written by | Notes |
|---|---|---|---|
| [comparisons/gorules-jdm-zen.md](comparisons/gorules-jdm-zen.md) | 264 | Opus | Cloned `gorules/zen` and `jdm-editor`; benchmarked the published `zen-engine 2.0.2` wheel on this machine |
| [comparisons/credit-platforms.md](comparisons/credit-platforms.md) | 1568 | Opus draft, finished on Sonnet | SAS Intelligent Decisioning deepest, with whole-book SAS PDFs grepped |
| [comparisons/fraud-engines.md](comparisons/fraud-engines.md) | 2510 | Opus fragments, assembled on Sonnet | 16 engines; every claim graded first-party / secondary / unreachable |
| [comparisons/open-source-and-dmn.md](comparisons/open-source-and-dmn.md) | 890 | Opus | DMN 1.5 PDF read via pdftotext; jDMN cloned and its Python codegen read |
| [comparisons/enterprise-brms.md](comparisons/enterprise-brms.md) | 637 | Opus | Broke through the IBM docs shell via product-code static paths |

Two caveats on citations. `trees/schema.py` and `trees/codegen.py` were mid-edit in the
working tree while the reports were written, so line anchors into them drift by up to
about 90 lines. `boundary/dtypes.py` and `boundary/nulls.py` were also being edited.
Table anchors were re-checked and are stable.

## 1. Headline verdicts

- **decider2 is uncontested on execution.** About 1 µs per record compiled, 0.72 µs per
  row in batch at 400 inputs and 633 outputs, hot swap at 0.177 µs, rollback with zero
  compile events, p99 flat from 1 to 16 threads with `nogil=True`. No surveyed engine
  vectorises, none compiles to machine code, and none publishes a comparable figure.
  Best first-party rivals: Higson 0.23 ms single-thread, InRule "single-digit
  milliseconds", SAS "5 to 10 ms per transaction" (marketing page, not the admin guide,
  which has no numbers). IBM ODM and ADS publish no latency figure anywhere.
- **decider2 is unique on the equivalence assertion.** `interpreted ≡ stepped ≡ fused ≡
  score()` as one automated test. Every rival tests *a* build; none asserts that traced
  semantics equal deployed semantics. One hole: the assertion skips any pipeline with a
  `str` input, and string tables are normal in credit `[O20]`.
- **decider2's governance tier is a specification, not code.** `lineage()`, `render()`,
  `diff()`, `impact()` and the decision record are all documented and none exists. The
  `origin` provenance argument is specified as mandatory and both entry points discard it
  with `del origin`. This is where the field is ahead, and the cheapest fixes are here.
- **Single-record latency is currently behind GoRules, not ahead.** ZEN answers a real
  7-node credit graph in 42.8 µs end to end through Python FFI. decider2's measured
  framework overhead on the single-record path is 971 µs p50 until the whole-row marshal
  fix from experiments N1 and N2 lands (projected 145 µs). "ZEN is too slow for
  real-time" is not a true statement today `[G16]`.
- **The table vocabulary is decider 1's, not a standard's.** A grep for DMN, hit policy,
  PMML or ONNX across all of `decider2/docs` and `example_projects` returns zero hits.
  Hit policy is hard-wired to first-match and never documented as a choice. Matched rows
  have no stable identity or annotation. Output types are inferred from data.
- **Cross-column gap and overlap analysis is the biggest single missing check.** decider2
  validates one-dimensional `between` contiguity only. jDMN ships three Apache-2.0
  sweep-line validators, Corticon auto-generates missing rows, ODM reports never-selected
  and conflicting rules. The table document is already a typed pydantic object, so this
  is tractable.
- **No vendor ships complete per-record reason codes.** Not SAS (its "adverse
  characteristics" are a training-time aggregate report), not Pega (zero doc hits for
  "adverse action"). Alloy's typed `adverse_action` field is the closest. decider2
  requires per-record contributions and can lead here, but only once the ragged-output
  question (doc 06 O5) is decided.
- **Champion-challenger precedent is looser than decider2's own plan.** Pega and Alloy
  both assign with a live random draw per call, and Pega documents the drift. decider2's
  deterministic-hash requirement is stricter than any shipped product. Shadow-mode
  isolation as a lineage assertion would put decider2 ahead of the entire fraud set,
  including AWS Fraud Detector, which has no shadow mode across 74 API operations.
- **Buy, don't build, applies to exactly three things:** the visual table editor and
  simulator (jdm-editor, dmn-js or kie-tools, all free), the table-analysis algorithms
  (jDMN, Apache-2.0), and the approval or promotion workflow (a bank already has one;
  decider2 needs an attachable hook, not an engine). Nothing on the execution path has an
  alternative.

## 2. Actionable recommendations, consolidated

Ranked within each theme. "Must" means an example project's contract or a regulated
bank's evidence expectation fails without it, as judged by the source report.

### 2A. Decision-table semantics

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| A1 | Add `hit_policy` to `DecisionTable`, values `first` (today's behaviour, now named), `collect` (emit a per-row hit bitmask, `ceil(rows/64)` int64s, decoded to ids in polars like string outputs already are), `unique` (validated at build via A-check B1), and later `priority` and `class_first_match` (ordered action classes, first match within class, which the fraud example already implements by hand in prose). Name them in the industry's words so JDM, DMN, AWS and DecisionRules import is a mapping. | `tables/schema.py`, `tables/codegen.py`, `tables/build.py` | S naming, M collect | Universal vocabulary; AWS `FIRST_MATCHED`/`ALL_MATCHED`; Stripe's action-class ordering; SAS rule sets collect by default | [G3][G4][O1][O2][C1][F1][F16][F19] |
| A2 | Add per-column collect: an output name ending `[]` collects across all matches while the rest stay first-hit. One table then yields one scalar `action_code` plus a list of `fired_rule_ids`. | `tables/schema.py`, `tables/codegen.py` | S after A1 | JDM `field[]` | [G4] |
| A3 | Add `COLLECT_SUM`: when the policy is collect with sum aggregation and all outputs are numeric, emit a scan that accumulates instead of returning, plus a matched-row count. An additive scorecard becomes one table. | `tables/codegen.py` | M | DMN `C+`, Camunda SUM, PMML Scorecard | [O4][O9] |
| A4 | Give every table row a required stable `id` and an optional `annotation`, as reserved columns excluded from condition resolution; derive the emitted row identity from the id, not the position. Add `annotation` to tree leaves too. Must. | `tables/schema.py` `ParametersConfig`, `tables/build.py` `decode()`/`explain()`, `trees/schema.py` `LeafNode` | S | DMN `rule id` and annotation clause; jDMN `Rule(index, annotationText)`; doc 04 §5.1's own instability rule | [O4][O5][C3][G1] |
| A5 | Add reviewer-facing `labels: Dict[str,str]` per column and `descriptions` per row, carried through `explain()`. Cheapest win on the doc 04 §6 reviewer problem. | `tables/schema.py` `ParametersConfig`, `tables/build.py` | S | JDM `name` beside `field`, `_description` per rule | [G1][G6] |
| A6 | Make declared `dtypes` authoritative for output typing; warn instead of inferring `int64` from all-integer data. Add an optional `domain` per condition input (min/max or allowed values) so gap analysis has a universe. Must for the output half. | `tables/codegen.py`, `tables/schema.py` | S / M | DMN `typeRef`, input values, `itemDefinition.allowedValues` | [O7][O8] |
| A7 | Add an explicit wildcard: an absent cell means "this condition does not constrain this row", emitted by skipping the guard via a `_has_*` array as `BetweenExpression` already does for open bounds. Render it as `any`. Distinguish it from a blank by accepting a `"-"` sentinel. | `tables/schema.py` `Expression.emit()` | S | JDM empty cell; DMN `-` | [G5][O12] |
| A8 | Add `NotExpression` with a negation-normal-form pass before `to_dnf()`, so tables match trees, which already have `TLogicOp.NOT`. Add first-class one-sided comparisons so a threshold does not need `between` plus neighbour-fill. | `tables/schema.py` | M / S | DMN grammar rules 5, 12.b, 13 | [O10][O11] |
| A9 | Write a unary-test cell parser accepting empty, `>= 95`, `[18..65]`, `(0..100)`, `'US','CA'` into the typed `Expression` union; reject both-inclusive ranges with a message naming `BoundMode`, or rewrite `[a..b]` to `[a..b+1)` for integer inputs only. One file unlocks JDM, DMN, Camunda and Excel import without changing semantics. | new `tables/cells.py` | S to M | JDM and DMN S-FEEL unary tests | [G6][G8][O13] |
| A10 | State the divergence in the codegen docstring: DMN's default is UNIQUE with unordered matching; decider2's is FIRST with early return, so row order is load-bearing here and not in DMN. | `tables/codegen.py` | S | DMN §8.2.11 | [O2] |

### 2B. Static analysis and diagnostics

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| B1 | Write three passes over a `DecisionTable`: `missing_intervals` per numeric column, `overlapping_rows` across all conditions jointly, `missing_rows` over the condition cross-product. Port jDMN's `SweepRuleOverlapValidator`, `SweepMissingIntervalValidator`, `SweepMissingRuleValidator` (Apache-2.0, algorithm cited in-source). Reuse `BetweenExpression.resolved_bounds()`. Corticon's variant auto-generates the missing rows; do that too. Must. | new `tables/analysis.py` | M | jDMN sweep validators; Drools `ANALYZE_DECISION_TABLE`; Corticon completeness checker; ODM consistency checking | [O3][B7][B8][C11] |
| B2 | Add `REDUNDANT_TABLE_ROW` (shadowed by earlier rows) and `NON_DISCRIMINATING_COLUMN` (removal changes no outcome), both pure functions of `ParametersConfig.data` plus the DNF form. Copy GoRules' message text. | new `tables/hygiene.py` | M | ZEN `policy/linter/table_hygiene.rs` | [G8] |
| B3 | Extend `BetweenExpression.allow_gaps` into whole-table `exhaustive()` and add the same for tree `CasesRanges`; emit `MISSING_DEFAULT_BRANCH` when a node is neither provably exhaustive (every enum value, both booleans, gap-free range) nor closed by a default. | `tables/schema.py`, `trees/schema.py` | M | ZEN `match_block.rs` exhaustiveness proof | [G9] |
| B4 | Add unreachable-node and missing-edge checks to the `Tree` validator (walk from root over children; every `sourceIndex` has an edge), and `UNREACHABLE_NODE` to the graph interface for steps nobody reads that are not terminals. | `trees/schema.py` validator, `graph/interface.py` | S | ACTICO, Corticon; ZEN `analysis.rs` | [C11][G13] |
| B5 | Introduce a structured diagnostic model: `Diagnostic(code, severity, location, message)` with `Location(module, step, expression, span)`; route the six build errors and three lints through it, keeping today's text. Prose `ValueError` is useless to a UI, a metric or a suppression list. | new `diagnostics.py`; `graph/interface.py`, `graph/pipeline.py`, `lint.py` | S frame, M to convert | ZEN's 40 diagnostic codes with three severities and span locations | [G2][G3] |
| B6 | Emit `IMPLICIT_ANY` when a step parameter has no annotation, because the driver silently makes it `float64`. Add `REPEATED_DERIVATION` for two steps whose bodies normalise to the same AST (decider 1 had 79 identity passthrough steps). | `params.py` `harvest_signature`, `lint.py` | S / M | ZEN `IMPLICIT_ANY`, `REPEATED_DERIVATION` | [G12][G14] |
| B7 | Write the graph well-formedness pass as a named, tested validation: acyclic, every step reachable from a root, every terminal is a decision, no path from a `shadow_*` population to a terminal. Closes the "no empirical support" gap in doc 00. | `graph/` | M | Sift's published workflow-engine constraints | [F25][F12] |
| B8 | Add `decider2 check --tables` with per-check flags (`--gaps`, `--overlaps`, `--missing-rows`, `--all`, `--off`) so analysis is a CI gate. | `cli.py` | S after B1 | Drools `<validateDMN>` granularity | [O19] |

### 2C. New interior kinds

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| C1 | Build a `Scorecard` kind mirroring `tables/`: `characteristics[]`, each `{name, feature, bins[], neutral_points}`, a `Bin` of `{min, max \| values \| is_null, points}`, bins in `shared` so a points edit recompiles nothing. Outputs `score` plus one contribution column per characteristic. Copy SAS's two mechanics outright: points-to-double-odds scaling (`score = ln(odds) * factor + offset` from author-facing `Odds` and `Points to double odds`) and the neutral-score contribution (`actual_points − neutral_points`, ranked worst-first). SAS computes that only as a training aggregate; decider2 computing it per record is an improvement on the one shipping precedent. Null is its own bin, always. Must. | new `scorecards/schema.py`, `codegen.py`, `build.py`; register as the third generic-kernel kind in doc 08 §3.4 | M | SAS Model Studio Scorecard node; FICO, Pega, GDS Link, Zoot scorecard primitives; PMML Scorecard | [C4][O4][O15] |
| C2 | Decide the ragged-output convention (doc 06 O5) before C1's contributions or any reason-code mechanism: fixed-width matrix with a declared maximum (works in numba today, matches the dtype-grouped 2D convention) or an offsets-plus-values pair. Delete the doc 04 claim that reason codes need no new machinery. Must as a decision. | `docs/06`, `docs/05` §3.1, `docs/04` | S decide, L for offsets | Nine invented names across six example projects | [C5][C8] |
| C3 | Add a `keyed_set` kind (`key -> bool`, `contains(key)` inside the kernel) and settle the keyed-lookup spelling (O4). Lists are the most universal fraud object: AWS `@list` at 100k entries, Stripe value lists at 50k, Ravelin Datalists, Actimize Platform Lists. Seven of eleven example projects needed a lookup and spelled it six ways. Must. | `tables/` or `params/tables.py` | M | AWS, Stripe, Ravelin, SAS lookup tables | [F4][C10] |
| C4 | Add a `temporal_table` kind, `(key, instant) -> row` stored as intervals, so "was this beneficiary on the list at 14:22:03 on 3 March" is answerable without 61,320 snapshots. No vendor publishes this; it is a bank requirement. Must for disputes and sanctions evidence. | `tables/` | L | Fraud example project's 2M-row mule list refreshing 24 times a day | [F5] |
| C5 | Rebuild `ruleset` as a generic kernel, not codegen: an array encoding per rule (feature index, operator code, threshold index, group id, on-absent policy) and one njit walker compiled once. Experiments measured 56.6 s to compile 100 rules in one unit, and a disabled rule still costs full compile time. A 521-rule fraud set cannot meet the "under 10 minutes including approval" criterion any other way. Give rules an `enabled` field meaning "not emitted" if codegen is kept. Must; the single biggest lever in the fraud report. | new `interiors/ruleset.py`; doc 08 §3.4 | L (S for `enabled`) | Every fraud vendor deploys a rule change in seconds; `experimentation/jittree` Approach B already shows the shape | [F2][F3] |
| C6 | Add a `Dictionary` document (named, labelled value sets) usable as a table output type and for reason codes; validate output cells against it at document time; replace the magic `MissingInputPolicy.reason = 4101`. Fixes the decline-reason taxonomy dropped in decider 1. | new `dictionaries.py`; `types.py` | M | ZEN dictionary blocks and `PREFER_DICTIONARY` hint | [G7][G10] |
| C7 | Add a `stage`/`checkpoint` field to rule and policy documents, independent of event type, validated against a per-project list like `Segments`. Two fraud vendors independently found event-type scoping insufficient. | rule document schema | S | Ravelin's checkpoint × category matrix; Forter's pre-auth/post-auth toggle | [F23] |
| C8 | Add a `variant` dimension to the input schema: one field catalogue keyed by event type, with per-field effective dates, so an 18-month backtest across a scheme field addition is possible. Must for fraud. | schema layer (doc 00 Layer 1) | M | AWS event types with per-type variables; FICO Falcon namespaced fields | [F17] |
| C9 | Decide overlays: build the per-record gain-vector mechanism (5 × 8 float64, built once per event, multiplied into a tunable where a predicate reads it) or declare it out of scope in doc 02. Leaving it undecided means every project invents it. | `docs/02` §3; possibly `params.py` | S decide, L build | No vendor publishes overlay-with-enforced-expiry | [F15] |
| C10 | Evaluate a bounded per-element execution mode (`Loop`/`Each`/`grain`, decider2's most-invented gap). GoRules' `executionMode: loop` is minimal evidence that a bounded form suffices for real documents. | `graph/` | L | JDM `executionMode: loop` | [G9 borrow list] |

### 2D. Trace and explainability

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| D1 | Add an opt-in trace emission variant that writes each condition's boolean result into a bit-packed int64 column beside the row or path column, and records the input values the matched row read. Same kernel, more outputs; never a forked code path (GoRules disables its row index under trace, which decider2's equivalence ladder forbids). This is the direct answer to doc 04's reviewer finding that "a reviewer cannot adjudicate a mismatch they are shown". Must. | `tables/codegen.py`, `trees/codegen.py` | M | ZEN `reference_map` plus `conditions[{id,result}]`; SAS rule-fired data; ACTICO full path | [G11][C2] |
| D2 | Add a `firing_set` convention: a fixed-width bitset column (`uint64[10]` covers 640 rules) plus a frame-tier `bitset_explode`, so "every rule that fired at 3,500 events per second, no sampling" is a column, not a debug mode. Then a `firing_counts_by(rule_id, minute)` rollup and a `dead_rule` report fall out as a polars group-by. Must. | `tables/codegen.py`, frame tier | M, then S | AWS `ruleResults`; Stripe per-rule performance chart; DataVisor response shape | [F6][F14] |
| D3 | Record arms not taken: emit the set of reachable leaf ids per tree so a batch run reports leaves no record reached. Add a `tree_node_stats(frame, tree, path_column)` function returning per-node entry, per-edge exit and zero-traffic nodes. The campaign-tree example specifies this exactly (1,340 of 9,200 nodes dead). | `trees/codegen.py`, new `trees/stats.py` | M / S | ODM "List of Rules Not Fired"; Experian strategy design aids | [B12][C9] |
| D4 | Add a `rendered_predicate` diagnostic: the fired rule's structure with each leaf's runtime value substituted inline as text. decider2's own reviewer test found values in a separate table were not enough. Must. | rendering function over rule structure plus emitted values | S to M | AWS `expressionWithValues` | [F22] |
| D5 | Emit a per-interior metadata constant from codegen (name, kind, hit policy, condition count, row count, source hash) so the audit record can say which interior ran. Must for the governance story. | `tables/codegen.py`, `trees/codegen.py` | S | jDMN `DRG_ELEMENT_METADATA` | [O6][O13] |
| D6 | Add literal-only template strings for reason text, resolved outside the kernel at write-back. | write-back / render layer | S | ZEN `render_template` | [G10 borrow list] |
| D7 | Generate a per-artefact reviewer sheet, `TableModule.sheet()` and `TreeModule.sheet()`, in the shape of the fraud example's rule-sheet artefact: authored value beside in-force value, why they differ, per-input missing behaviour, approval row. Smallest honest step on the top-ranked risk; gives experiment E4 something to show a reviewer. Also render `Step.doc` and `Step.implements`, which are parsed and unused. | `tables/build.py`, `trees/build.py`, new `observe/render.py` | M | SAS generates PDF docs for five object kinds; Oracle IA; ODM verbalization | [C19][B16] |
| D8 | When timing is recorded, serialise as integer microseconds; when a trace is recorded, use an ordered list, never a dict keyed by node id. A design choice worth fixing before `observe/` exists. | `observe/` design | S | Two defects in ZEN's tracer versus its correct policy trace | [G15] |

### 2E. Governance and lifecycle

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| E1 | Record `origin` instead of `del origin` at both entry points; make it non-empty as doc 08 §6.2 already specifies. The cheapest must-have in the survey. | `runtime/invoke.py`, `graph/pipeline.py` | S | InRule check-in audit; decider2's own spec | [B1] |
| E2 | Write the decision record: `observe/audit.py` emitting the doc 08 §8 record (skeleton identity, structure fingerprint, compiled artefact id, params digest and origin, generation, variants, fallback set, inputs, outputs, emitted values, plus the evaluated-conditions mask and the cap chain). Include `decision_latency_ms`. Nothing in the package writes one today. Must; the single largest gap. | new `observe/audit.py`, `runtime/invoke.py` | L | ODM Decision Warehouse's 17 options; Pega "what it chose, what it rejected, and why"; AWS `GetEventPredictionMetadata` is thinner | [B5][C13][F11][F27] |
| E3 | Add audit verbosity and a correlation id to the request: `{"record":…, "audit":"off\|fired\|full", "correlation_id":…}`, echoed back. Doc 04 §6.5 item 3 specifies this. | `serving/dispatch.py`, `runtime/serve.py` | S | DecisionRules `X-Audit`, `X-Correlation-Id`; ADS `executionTraceFilters` | [B6] |
| E4 | Add effective dating: reserved `valid_from`/`valid_to` columns on `ParametersConfig` (and tree leaves), resolved before the kernel against an explicit `as_of` argument so compiled code never sees a date and `decision_date` stays an input. Per-entry effective dating on params too. decider2 has no temporal dimension at all, and example project 09 makes decision date a replay precondition. Must. | `tables/schema.py`, `tables/build.py`, `trees/schema.py`, params schema, doc 08 §3.4 | M | Corticon `decisionServiceEffectiveTimestamp`; Higson time versioning; ODM expiry filters; SAS lookup activation; Stripe 180-day rule log | [B13][O17][C10][F9] |
| E5 | Name generations: add `generation_id` and label to `ServeHandle`, report them from `/health`, persist each staged document plus fingerprint and origin to the build directory under the content-addressed cache, and offer `restore(generation_id)` on start. Today `_history` is an in-process list that dies at restart. Must. | `runtime/serve.py`, `compile/cache.py`, `serving/dispatch.py` | S + M | InRule revisions and labels; Corticon Major.Minor resident versions; Higson snapshots | [B2][B14] |
| E6 | Add `POST /params/by-ref {"generation": id}` and let `stage()` accept a retained id, so production can be configured to accept only references. Closes doc 04 §2.1's payload-params hole; experiment N3 already removed the performance objection (1.28% of budget at 50 modules). | `serving/dispatch.py`, `runtime/serve.py` | M | InRule "caller names a revision label"; DecisionRules aliases | [B3] |
| E7 | Carry an approver: `approved_by`, `approval_ref`, `author`, `change_class`, and a marker for values written by an automated control, on `StagePlan` and the params document; required in `live` mode; recorded verbatim, never parsed, the posture `origin` already has. Answers O17 at document granularity. Must. | `runtime/serve.py` `StagePlan`, `serving/dispatch.py`, doc 04 §5.2 | S | ODM release approval; ACTICO audit-proof approvals; Hawk "4-eyes" is a claim with no published mechanism; BioCatch's "segregation of duties" is RBAC only | [B15][C12][F10] |
| E8 | Implement `pipeline.lineage(name)` and `used_by(name)` on the DAG the pipeline already walks, and `decider2.diff(old, new)` over params documents and the two generated driver source files (real files, so `difflib` suffices). All three are documented and do not exist. Model the diff result on jdm-editor's shape: per-element status in `{added, removed, modified, unchanged, moved}` with previous value and index. Must for lineage. | `graph/pipeline.py`, new `observe/diff.py` | S | ODM rule analysis; SAS "Determine Which Objects Use"; Higson compare; jdm-editor `dg-types.ts` | [B4][B18][C18][G18] |
| E9 | Add optimistic concurrency to the authoring surface: a version token from `GET /params` required on `POST /params`, rejecting stale writes. Cheap now, expensive after the first clobber. | `serving/dispatch.py` | S | Sift `ETag`/`If-Match` | [F24] |
| E10 | Document the interface contract file (`contract=` already writes and diffs `contracts/{name}.json` over inputs, outputs, terminals and params) as a governance artefact. A working gate no doc advertises. | `docs/04`, `docs/07` | S | ADS protected branches | [B17] |

### 2F. Simulation, testing and analytics

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| F1 | Implement `decider2.impact(active, candidate, sample) -> ImpactReport` by generalising `ServeHandle.preview()` from one record to a frame: fraction of rows whose terminals changed, per-output change distribution, which rows or leaves newly matched or stopped. Defer the exact boundary solve (O24) and say the report sampled. Expose as `POST /impact`. Must. | new `testing/impact.py` or `observe/blast_radius.py`; `runtime/serve.py`; `serving/dispatch.py` | M + S | ODM simulation with KPIs; Experian strategy simulation; Oscilar backtesting; Taktile | [B9][B10][C7][C8] |
| F2 | Add a `backtest(baseline=, metrics=)` entry point producing a decision pack over the fraud example's already-written metric list (hit rate, precision, incremental catch, FPR, overlap matrix, value blocked, action churn). Add a counterfactual-FPR metric for suppressing actions from the model score's calibration curve, because a declined transaction never produces an outcome label. Must. | `runtime/` | M + M | Stripe 6-month backtests with "Est. false positive rate"; Sift 30-day backtests; Ravelin; DataVisor | [F13][F21] |
| F3 | Add `asserts_no_effect(pipeline_a, pipeline_b, corpus)`: exact equality on every declared output, no tolerance, as the CI gate that blocks a promotion. Example project 09 calls this the rule that catches more defects than the rest of the suite. Must. | `testing/` | S | Decisions unit tests at deploy; InRule DevOpsServices regression | [B11] |
| F4 | Close the `str`-input hole in the equivalence ladder so string tables are covered by the generic assertion. This is the one place decider2's strongest property is not fully asserted. | `testing/equivalence.py`, `runtime/invoke.py` dictionary-code path | M | Nothing external | [O20] |
| F5 | Add a labelled-fixture format (`{labels, inputs, expected}`) and seed it with the untested table semantics: `upper_inclusive` boundary equality, empty `in` set, `None` bound, string absent from every row. | `testing/`, `tests/` | S | DMN TCK `<labels>` convention | [O18] |

### 2G. Champion-challenger and shadow mode

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| G1 | Add `Experiment(id, name, salt, arms={name: pct}, eligible, opens, closes, measures, approval_reference)` and an `ExperimentRegistry`, taking the schema the collections example already designed. Assignment is `hash(stable_id + salt) % 100`, never random, emitted as an ordinary arm-index value. Validate a minimum share per arm. Add a lint failing any step that imports `random` or calls `np.random`. Must. | new `experiments/schema.py`, `graph/combinators.py`, `lint.py` | M (lint S) | Alloy's arms-and-minimum-5% shape; Pega's documented random-draw drift is the mechanism to reject | [C6][B17] |
| G2 | Add a `population` argument (`live`/`shadow`) to the ruleset declaration and make shadow isolation a lineage assertion: no path from `shadow_*` to any terminal, checked at import with no data. No vendor publishes an enforcement mechanism; AWS Fraud Detector has no shadow mode at all. Must. | `graph/`, ruleset schema | M | Sardine, Unit21, DataVisor, Hawk publish the feature name only | [F12] |

### 2H. Serving and production safety

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| H1 | Prioritise the whole-row marshal fix from experiments N1 and N2 (971 µs to a projected 145 µs). Not a borrow; a competitive fact against ZEN's measured 42.8 µs. | `runtime/invoke.py` | M | ZEN benchmark on this machine | [G16] |
| H2 | Add `deadline_ms` to `/invocations` and write the fail-open page for doc 02 §3.6 with two defaults, not one: an infrastructure deadline breach resolves to a declared per-project safe action; decider2's own circuit breaker always fails closed and carries a marker. Add a `partial`/`degraded` response state for when an upstream table was degraded. Today `serving/` has no timeout, deadline or degraded response. Must. | `serving/dispatch.py`, `docs/02` §3.6 | S | Ravelin's two-mode split (750 ms timeout fails open; own rate-limit trip fails closed); Forter publishes a timeout and no policy; Hawk HTTP 206 | [F7][F26] |
| H3 | Make `nogil=True` the default for kernels built through `serve()`, or mark `/health` as `degraded` when a GIL-holding kernel is served. Experiment N4 measured p99 at 1,270% of budget at 16 threads without it. Must. | `runtime/serve.py` | S | No vendor serves from CPython | [F8] |
| H4 | Add a declared divert channel: `apply()` returns `(kept, diverted)` when a step or null policy marks a row, each diverted row carrying its reason. `route_required_nulls` already computes the mask and has nowhere to send it. Must. | `graph/pipeline.py`, `boundary/nulls.py` | M | SAS filtering rule sets; Alloy step-up and manual-review nodes; `docs/REVIEW.md` names it a defect | [C17] |
| H5 | Add a `decider2/docker/Dockerfile` that runs `decider2 build --verify` as a build step and records the artefact id in image labels. | new `docker/` | S | SAS Container Runtime; ACTICO on Kubernetes; IBM ODM container repo | [C14] |
| H6 | Decide on a `fan()` combinator for concurrent independent enrichment, or document in doc 02 §3.6 that the caller does it. | `graph/combinators.py` or doc 02 | M or S | Every fraud vendor overlaps enrichment; the fraud example needed a fifth combinator | [F18] |
| H7 | Publish a one-page latency and throughput sheet in vendor-comparable units, stating that decider2's number is in-process, not network-inclusive. | docs | S | DataVisor's "30 ms / 15,000 QPS" framing | [F28] |

### 2I. Interchange

| # | Do | Where | Effort | Motivated by | Sources |
|---|---|---|---|---|---|
| I1 | Export JDM: one `decisionTableNode` per `TableModule`, a `switchNode` chain per `TreeModule`, input and output nodes from `Interface`, decider2-only facts (null tiers, `InputRef`, `bound`) parked in an ignored key. The MIT `jdm-editor` then serves as a free visual editor, diff viewer, Excel round-trip and simulator (it bundles a WASM engine, so no backend). Lossy export only, never storage; the simulator's numbers are ZEN's, not decider2's, so ship it behind a banner. | new `export/jdm.py` | M to L | `@gorules/jdm-editor` 1.52.0 | [G7] |
| I2 | Export DMN 1.3 `<decisionTable hitPolicy="FIRST">` for the subset decider2 can express, so `dmn-js` or kie-tools become a table editor and reviewer view. Import DMN at conformance level 2 via the A9 parser; refuse anything outside CL2 by name. Do not add FEEL. | new `interop/dmn_export.py`, `interop/dmn_import.py` | S / M | Free mature DMN editors; jDMN proves the mapping is mechanical | [O13][O14][C15] |
| I3 | Import PMML `Scorecard` into C1's schema (attribute predicates to bins, `partialScore` to points, `reasonCode` to annotation, `baselineScore` to default) and PMML `TreeModel` into `Tree`. SAS `PROC PSCORE` and Pega's import allow-list both use PMML Scorecard as a live path, stronger evidence than DMN CL2 has. Scope to those two elements only. Refuse ONNX; it forces the kernel split experiments measured at 23 to 77 times slower. | new `interop/pmml_scorecard.py`, `interop/pmml_tree.py` | M each | SAS, Pega, GDS Link, Zoot | [C20][O15][O16] |
| I4 | Add an Excel round-trip for `DecisionTable` documents (header row is `labels`, body is `data`). Treasury authors the rate card in a spreadsheet in the loan-granting example. | `tables/` | S to M | jdm-editor `excel.ts`; Sparkling Logic lookup models from spreadsheets | [G17] |
| I5 | Write `docs/09-standards-and-interop.md`: DMN CL2 for tables in and out, PMML for scorecards and trees, JDM export for the editor, ONNX out of scope and why, with the mapping tables. | new doc | S | Zero-hit grep for any standard across the doc set | [O21] |

### 2J. Documentation corrections

| # | Do | Sources |
|---|---|---|
| J1 | Delete the doc 04 claim that reason codes need no new machinery (refuted in `docs/REVIEW.md`). | [C16] |
| J2 | Doc 06 E3 and the example projects' FINDINGS treat `score()` agreeing with `apply()` as a missing fourth rung; it is implemented and asserted, and it is decider2's strongest differentiator. State it as done. | [C16] |
| J3 | Replace the superseded `Table` sketch in doc 03 §? (lines 720 to 748) with the real `decider2.tables` API rather than annotating it. | [C16] |
| J4 | Fix doc 08 §6.2's statement that the framework refuses to run without `origin`, or fix the code (E1). One of them is currently false. | [B1] |

### What not to adopt

- **JavaScript or any expression-language escape hatch in config** (GoRules `functionNode` on QuickJS, Microsoft RulesEngine, Azure Logic Apps rules). Doc 08 §1 forbids code in config on evidence, and GoRules' own numbers show it drops throughput from 150k to 3 to 50k per second.
- **HTTP calls or `rand()` inside a decision.** Both break the bit-identical replay example project 09 requires; nothing in ZEN enforces the ban.
- **Trace as a forked code path.** ZEN disables its row index under trace; decider2's trace must be the same kernel with more columns.
- **Natural-language rules as the executable artefact** (Oracle IA, InRule BLE, ODM BAL, Oscilar, Provenir). Doc 04 §6 tested a rendered artefact twice and both failed; generating rules from prose makes the reviewer problem worse.
- **JDM or DMN as the internal representation.** Both lose the four null tiers, `InputRef`, the `name@module` chain and `NotApplicableAs`. Export, never store.
- **FEEL three-valued null logic and `Decimal`.** The four-tier boundary is stronger for credit; `Decimal` panics at the boundary.
- **DMN's aggregating hit policies beyond `C+`**, and `Any`, `Rule order`, `Output order`. No credit or fraud vendor uses them; Camunda skipped two of them.
- **A promotion or approval workflow engine.** Banks have one. Make the activation hook attachable (E5, E7) instead.
- **Windowed velocity features inside the engine.** Keep them upstream, but state it as a product boundary with a declared freshness state on the input, not as a scope note.

## 3. Feature inventory

Consolidated from the five reports' inventories (which together run to about 250 rows).
Status is against built code, not docs. Full per-feature citations are in the appendices.

### 3.1 Authoring primitives

| Feature | Who has it | decider2 | Verdict |
|---|---|---|---|
| Decision table with named hit policy | DMN engines, FICO, ACTICO, SAS (collect by default), Pega ("Evaluate all rows"), AWS, DecisionRules, JDM | partial: first-match only, unnamed | adopt (A1) |
| Collect / per-column collect | JDM, DMN, AWS `ALL_MATCHED` | no | adopt (A1, A2) |
| Additive scorecard table (`C+`) | DMN, Camunda, PMML | no | adopt (A3) |
| Scorecard object with points scaling and per-characteristic contributions | SAS Model Studio (training-time only), FICO, Pega, GDS Link, Zoot | no; required by example projects | adopt (C1), and compute per record |
| Strategy or decision tree with positions | Experian, Pega, SAS, decider2 | yes | keep |
| Rule set as a first-class object | SAS, FICO, ACTICO, Pega, Zoot, ODM | no; doc-only, Layer 4 blocked | adopt as a generic kernel (C5) |
| Lookup and keyed set tables | SAS lookup tables, AWS lists, Stripe value lists, Ravelin Datalists | no (O4 open) | adopt (C3) |
| Temporal membership as at an instant | none | no | build (C4); a bank requirement, not a vendor feature |
| Dictionary / governed enum | JDM policies | no | adopt (C6) |
| Wildcard cell, `-` irrelevant | JDM (empty cell), DMN | no | adopt (A7) |
| Both-inclusive ranges `[a..b]` | DMN, JDM | no, deliberately | parser only (A9) |
| `not(...)` in tables | DMN | trees yes, tables no | adopt (A8) |
| Reviewer label per column, description per row | JDM | no | adopt (A5) |
| Stable rule id and annotation | DMN, jDMN, pyDMNrules | no; positional index | adopt (A4) |
| Declared output type and input domain | DMN `typeRef`, `allowedValues` | partial; inferred | adopt (A6) |
| Per-element loop | JDM `executionMode: loop` | no (O5) | evaluate (C10) |
| Treatments / offer allocation | Pega, SAS, Experian | partial via tables | covered by tables plus C1 |
| Effective-dated rows and params | Corticon, Higson, ODM, InRule, SAS lookup activation, Stripe rule log | no | adopt (E4) |
| Stage / checkpoint scoping | Ravelin, Forter | no | adopt (C7) |
| Per-event-type field catalogue | AWS event types, FICO Falcon | no | adopt for fraud (C8) |
| Business-user parameter editing with bounds | InRule, ODM, Higson, all | yes (pydantic `Field`, JSON Schema out) | ahead |
| Zero-recompile threshold and row edits | Higson, DecisionRules | yes, measured | best in class |
| Concurrent enrichment combinator | every fraud vendor | no | decide (H6) |

### 3.2 Static analysis and validation

| Feature | Who has it | decider2 | Verdict |
|---|---|---|---|
| Cross-column gap, overlap, missing-rule analysis | jDMN, Drools, Corticon, ODM | one-dimensional contiguity only | adopt (B1); highest-value single import |
| Auto-generated missing rows | Corticon | no | adopt (B1) |
| Redundant row, non-discriminating column | ZEN table hygiene | no | adopt (B2) |
| Exhaustiveness proof, missing default | ZEN match blocks | partial | adopt (B3) |
| Unreachable node / dead branch | ZEN, ACTICO, Sift | no | adopt (B4, B7) |
| Duplicate writer, cyclic dependency | ZEN policies, decider2 | yes | keep; copy the message style |
| Implicit any / untyped parameter warning | ZEN | silent `float64` | adopt (B6) |
| Structured diagnostics with codes, severity, span | ZEN (40 codes) | no; prose `ValueError` | adopt (B5) |
| Per-check CI flags | Drools `validateDMN` | no | adopt (B8) |
| Optimistic concurrency on edits | Sift `ETag` | no | adopt (E9) |

### 3.3 Trace, explainability, audit

| Feature | Who has it | decider2 | Verdict |
|---|---|---|---|
| Matched row or path as a free output column | decider2 | yes | ahead |
| Values actually read plus per-condition booleans | ZEN `reference_map`, `conditions[]` | no | adopt (D1) |
| Rule text with runtime values inlined | AWS `expressionWithValues` | no | adopt (D4) |
| Firing set for hundreds of rules | AWS `ruleResults`, Feedzai, Actimize, DataVisor | no | adopt (D2) |
| Rules not fired, tasks not executed | ODM Decision Warehouse | no | adopt (D3) |
| Node-level traffic statistics | Experian, Pega (strategy shapes only) | no | adopt (D3) |
| Per-interior metadata in the record | jDMN `DRG_ELEMENT_METADATA` | no | adopt (D5) |
| Decision record with inputs, outputs, version, rules fired | ODM, ADS, Pega, DecisionRules, AWS (thinner) | specified, unbuilt | build (E2); largest gap |
| Per-request audit verbosity and correlation id | DecisionRules, ADS | no | adopt (E3) |
| Trace linked back to the authored rule | Corticon, ODM | partial | adopt via D5 plus D7 |
| Per-decision latency in the record | Hawk `took` | no | adopt (E2) |
| Printable reviewer sheet per artefact | SAS (five object kinds) | no | adopt (D7) |
| Reason codes per record from a scorecard | none complete; Alloy typed field closest | required, undesigned | lead (C1, C2) |
| Template strings for reason text | ZEN | no | adopt (D6) |
| Policy-clause join key on a step | none | yes (`Step.implements`), unrendered | render it (D7) |

### 3.4 Lifecycle and governance

| Feature | Who has it | decider2 | Verdict |
|---|---|---|---|
| Provenance token on the running config | none as cleanly as decider2 specifies | specified, discarded | fix (E1) |
| Named, retained, addressable generations | InRule, Corticon, Higson, DecisionRules | in-process LIFO, lost on restart | adopt (E5) |
| Params by reference, not by value | InRule labels, DecisionRules aliases | no | adopt (E6) |
| Approver and change class on activation | ODM, Sapiens, ACTICO, Taktile | no (O17) | adopt (E7) |
| Effective-dated selection per request | Corticon, Higson | no | adopt (E4) |
| Lineage, used-by, diff | ODM, SAS, Higson | documented, absent | build (E8) |
| Interface contract gate | ADS protected branches | yes, undocumented | document (E10) |
| Hot swap without restart | InRule, ODM, ADS | yes, 0.177 µs measured | best in class |
| Rollback | InRule, Corticon, Higson | in-process only | persist (E5) |
| Branching and merging | ADS, ODM | git for the skeleton | keep |
| Approval workflow engine | ODM, InRule, Decisions | no, by decision | do not build |
| Champion-challenger | Pega, Alloy (random draw); ACTICO, SAS, FICO, Equifax (claims) | no | adopt with deterministic hash (G1) |
| Shadow mode enforced by lineage | none | no | build (G2); would lead the field |
| Simulation with KPIs on a sample | ODM, Experian, FICO, Oscilar, Taktile, Higson | one-record preview only | build (F1) |
| Backtest with per-rule precision and FPR | Stripe, Sift, Ravelin, DataVisor | metric list written, no harness | build (F2) |
| Golden-set gate at promotion | Decisions, InRule | primitives, no gate | adopt (F3) |
| Batch equals real-time, asserted | none | yes, four rungs | unique; close the string hole (F4) |
| TCK-style labelled fixtures | DMN TCK, jDMN | partial | adopt (F5) |

### 3.5 Serving and operations

| Feature | Who has it | decider2 | Verdict |
|---|---|---|---|
| Timeout, deadline, fail-open/closed policy | Ravelin (two-mode), Forter (timeout only) | none in `serving/` | adopt (H2) |
| Degraded or partial response | Hawk HTTP 206 | no | adopt (H2) |
| Row-level divert to review | SAS filtering rule sets, Alloy step-up | mask computed, no destination | adopt (H4) |
| Container with compile-at-build | SAS Container Runtime, ACTICO, IBM | specified, no Dockerfile | adopt (H5) |
| Published latency sheet | DataVisor, GoRules | scattered across experiments | write (H7) |
| Saturated counter state on velocity inputs | Stripe (caps at 25) | fresh/stale/absent only | add if upstream counters are bounded |
| Sanctions direct vs ownership hit | Hawk | no | add if compliance routes them differently |

### 3.6 Interchange and tooling

| Feature | Who has it | decider2 | Verdict |
|---|---|---|---|
| Free visual editor with simulator | jdm-editor (MIT, WASM engine), dmn-js, kie-tools | no | export to it (I1, I2) |
| Excel round-trip | jdm-editor, Higson, Sparkling Logic | no | adopt (I4) |
| DMN conformance level 2 import/export | Camunda, Drools, jDMN, Trisotech; FICO names DMN | no | adopt (I2) |
| PMML scorecard and tree import | SAS `PROC PSCORE`, Pega, GDS Link, Zoot, Drools | no | adopt (I3) |
| ONNX | listed by OMG | no | refuse |
| Field-level diff model for change review | jdm-editor | change class only | adopt the shape (E8) |
| Content-addressed compiled artefact cache | none found | yes | ahead |

## 4. The GoRules verdict

GoRules is JDM (a vendor JSON format, not a standard), the ZEN engine (Rust, MIT, Python
and Node bindings), jdm-editor (React, MIT, bundles a WASM engine), a newer "policies"
authoring form with 40 static checks, and a commercial BRMS for branching, approvals and
audit. The structural side-by-side, with `file:line` into the zen repo and decider2, is
[comparisons/gorules-jdm-zen.md](comparisons/gorules-jdm-zen.md) §1.

**Structure, briefly.** JDM is an explicit `nodes` plus `edges` graph with eight node
types (`inputNode`, `outputNode`, `decisionTableNode`, `expressionNode`, `functionNode`,
`switchNode`, `decisionNode`, `customNode`), traversed topologically with merged-object
data flow. decider2 infers order from a flat name registry and gives typed, emittable
`name@module` version chains, which is stronger. Decision tables have two hit policies
(`first`, `collect`), per-column collect via `field[]`, an empty cell as wildcard, a
row-pruning bitset index above eight rows (disabled under trace), and unary-test cell
syntax shared with DMN S-FEEL. Expressions run on a bytecode VM in the ZEN expression
language, which shares FEEL's range and comparison syntax but not its function library.
`functionNode` runs JavaScript on QuickJS. The trace records per node the input, output,
timing, matched-row cell text, a `reference_map` of values actually read, and
per-condition booleans. A switch node prunes losing edges and restarts the walk, which is
why the trace is keyed by node id rather than ordered.

**Measured on this machine** (published `zen-engine 2.0.2` wheel, 7-node credit graph):

| Path | Per evaluation |
|---|---|
| `engine.evaluate` with a precompiled `ZenDecisionContent` loader | 42.8 µs |
| `engine.evaluate` re-reading the JSON each call | 291 µs |
| `ZenDecision.evaluate` after `create_decision`, the README quickstart | 1,408 µs |
| `evaluate_batch`, 20k rows, 28 CPUs | 86 µs per row |
| decider2 batch, 400-in/633-out, 100k rows (doc 01) | 0.72 µs per row |
| decider2 single-record framework overhead today (doc 01 N1) | 971 µs p50 |

The quickstart pattern is 33 times slower than the loader path with byte-identical
results; anyone benchmarking from the README gets the wrong number. Policies are fully
implemented in the MIT engine but unreachable from Python: `create_decision` raises on
policy content and the diagnostics API is Node-only. The documented Polars integration is
per-row `map_elements` with a JSON round trip inside the UDF.

**Where GoRules is a genuine alternative.** One slice only: a low-width, real-time-only,
single-record decision such as a fraud rule set on a few dozen fields, served at 43 µs,
authored in a free editor with a working simulator, no compile step. It would be live in
a week. It is not defensible for anything batch-shaped, wide, or replay-certified: batch
is 100 times slower per row serial and 5 to 10 times slower fully parallel; `rand()` and
HTTP are in the standard library with nothing enforcing a ban; and a 1,000-key JSON
object per record does not fit the 400-in/633-out shape.

**The right reading.** GoRules is finished versions of four things decider2 has
specified and not built: a diagnostic model (B5), a trace shape (D1), a visual editor
(I1), and a table document format (A9). Borrow those. Do not copy the JavaScript escape
hatch, in-decision HTTP, `rand()`, graph mutation during evaluation, timing as a
formatted string, a trace keyed by node id, or a trace that disables an optimisation.

Three corrections to the landscape note's §8 came out of reading the source: `switchNode`
statements have no `isDefault` (a default is an empty condition); the table wildcard is
an empty cell, not `-`; and policies are not docs-only. These are now noted in that file.

## 5. What decider2 already does better, and should not trade away

- The compiled record tier and its numbers, above.
- The four-rung equivalence assertion, unique in the survey.
- Free reference-data edits: table rows live in `shared` arrays scanned by one generic
  kernel, so a row change recompiles nothing. Stronger than SAS's "activation and
  locking" and it beats all nine of Decisions.com's rule kinds on the governance
  property that matters most.
- Four declared null tiers including not-applicable distinct from missing, stronger than
  FEEL's three-valued logic for credit.
- Typed, emittable `name@module` version chains, stronger than JDM's `$nodes` addressing.
- The interface contract file, a working "interface changed" gate.
- Zero-recompile threshold retunes, measured; hot swap and rollback, measured.
- A reviewer finding no vendor has: "displaying a value is not the same as making it
  checkable." Adopting GoRules' or Oracle's materials without that finding would
  reproduce their failure at higher fidelity.

## 6. Suggested order

If one engineer starts on Monday, the reports converge on this order, by cost of late
discovery rather than by effort:

1. E1 `origin`, E7 approver fields, B5 diagnostic frame, A5 labels, A4 stable row ids.
   All S, and everything downstream needs them.
2. A1 hit policy naming plus collect, A6 authoritative output types, A7 wildcard, D5
   metadata constant. Table document changes before anything imports or exports it.
3. B1 gap and overlap analysis, B3 exhaustiveness, B4 unreachable nodes, B8 CI flags.
4. D1 per-condition trace, D2 firing set, E2 decision record, E3 verbosity. The audit
   story, in the order the record needs its inputs.
5. E4 effective dating, E5 named persisted generations, E6 by-ref params, E8 lineage
   and diff. The lifecycle story.
6. F1 impact, F3 no-effect gate, F4 string-input hole, G1 experiments, G2 shadow lineage.
7. C1 scorecard, after C2 is decided. C3 keyed set. C5 ruleset as generic kernel.
8. I1 JDM export and I2 DMN export, then I3 PMML import, then I4 Excel. Tooling last,
   once the documents it renders are stable.
9. H1 marshal fix, H2 fail-open, H3 `nogil` default, H4 divert. Serving, before any
   production traffic. H2 and H3 are S and could go first.

## 7. What could not be verified

Per report, in the appendices' unverified sections. The recurring ones: all FICO product
pages (bot wall); Mastercard Brighterion entirely; TransUnion product pages (403);
Feedzai, Unit21, Taktile, Oscilar developer docs (login walls); Featurespace product
pages (JavaScript shells); SAS Container Runtime docs (docset not locatable); Sardine's
widely quoted "about 25 ms" (no first-party source found); DataVisor's real first-party
latency is 30 ms, not "under 100 ms". Two of the five reports were finished on a smaller
model after an API rate limit; their evidence grading is explicit per claim, and the
credit report's SAS section rests on whole-book documentation PDFs.
