# Open-source rule/decision engines and OMG DMN vs `decider2`

Scope: §5 (DMN implementations + TCK, stateless table evaluators, Rete/PHREAK, DAG
orchestrators) and §7 (standards) of `decider2/docs/research/decision-engine-landscape.md`,
verified against primary sources. GoRules/JDM is another agent's brief and appears here only
where a cross-engine comparison needs it.

Primary sources read locally:

- **DMN 1.5** — OMG `formal/24-01-01`, PDF fetched from `https://www.omg.org/spec/DMN/1.5/PDF`
  and extracted with `pdftotext` (`scratchpad/dmn15.pdf`, `scratchpad/dmn15.txt`).
  **Numbering caveat:** the PDF's TOC and body disagree by one after §8.2.5 — the TOC lists
  "8.2.10 Hit policy / 8.2.11 Default output values" (`dmn15.txt:228-229`) while the body
  headings read "8.2.11 Hit policy" (`dmn15.txt:3480`) and "8.2.12 Default output values"
  (`dmn15.txt:3589`). Citations below give the body numbering with the TOC number in brackets.
- **jDMN** — `git clone --depth 1 https://github.com/goldmansachs/jdmn` (Apache-2.0),
  including its generated-Python expectation fixtures and the vendored DMN TCK corpus.
- **DMN TCK** — `https://github.com/dmn-tck/tck` README; TCK fixtures as vendored in jDMN.
- Web: Drools DMN docs, Camunda DMN engine + hit-policy pages, feel-scala README,
  pyDMNrules README, PyPI `jdmn-python-runtime`.
- Long-tail engine facts (licences, versions, dormancy, vendor benchmarks) come from the
  fetch pass recorded in `scratchpad/part_oss_notes.md` (2026-09-20); anything only in that
  file is marked *(landscape pass)* rather than re-verified here.

`decider2` citations are `file:line` against the working tree at
`/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/`. **Nothing under
`decider2/` was edited.** `src/` has uncommitted user changes and was being edited *during* this
pass: `boundary/dtypes.py` and `boundary/nulls.py` both moved mid-session (`ColumnPlan` became a
tagged union of plan classes), so their line numbers here are the post-move ones and may move
again. `tables/{schema,codegen,build}.py` and `trees/{schema,codegen}.py` were re-checked at the
end of the pass and their anchors are stable.

One framing fact that colours everything below: **`decider2` has no DMN, PMML or ONNX story
at all.** `grep -rn "DMN\|hit polic\|PMML\|ONNX" decider2/docs/*.md decider2/example_projects/*.md`
returns zero hits. Its decision-table vocabulary is inherited wholesale from decider 1
(`tables/schema.py:1-9`), not from a standard.

---

## 1. DMN decision tables vs `decider2` decision tables — semantic table

`decider2`'s table is `DecisionTable` (`tables/schema.py:551-588`): a `ParametersConfig` of
rows (`:152-196`), one `Expression` tree evaluated per row, a list of `outputs` column names,
and an optional `default` list. Its execution is a single emitted `for r in range(n)` scan
that returns the first matching row index, or `-1`
(`tables/codegen.py:200-219`).

| Concern | DMN 1.5 | `decider2` | Verdict |
|---|---|---|---|
| **Rule/row model** | Rules are rows of *input entries* (one per input clause) + output entries + annotations; "the number of input entries in rule N does not match the number of input clauses" is a validation error (jDMN `DefaultDMNValidator.java:553-557`) | Rows are dicts of arbitrary columns; the *expression* names which columns are bounds/values (`tables/schema.py:314-317`, `:408-409`, `:497-498`). A row is not required to have an entry per condition | **Different on purpose.** decider2's shape is "N rows × M conditions with uniform operators" — doc 08 §3.4's test for a generic kernel. It buys the free-interior property; it loses per-cell authoring |
| **Hit policy: Unique (U)** | Default; "no overlap is possible, and all rules are disjoint" (§8.2.11 [8.2.10], `dmn15.txt:3527`); implementations SHALL NOT contain overlapping rules (`:3500`) | Not expressible. There is no hit-policy field | **Missing.** U is the *only* policy with a static-analysis obligation, and decider2 has no overlap check |
| **Hit policy: Any (A)** | Overlap allowed, all matches must have equal outputs; non-equal ⇒ "hit policy is incorrect, and the result is undefined" (`:3529-3531`) | Not expressible | Missing, low value (A is U plus a licence to be sloppy) |
| **Hit policy: Priority (P)** | Multiple match; return the match highest in the declared *output values* list; "priorities are independent from rule sequence" (`:3532-3535`) | Not expressible | Missing, **genuinely useful** for credit decline-reason precedence (worst outcome wins regardless of row order) |
| **Hit policy: First (F)** | "The first hit by rule order is returned (and evaluation can halt)" (`:3536-3540`) | **This is decider2's only policy**, and it is hard-coded: `tables/codegen.py:217-218` `if matched: return r`; `tables/schema.py:561-564` documents "First match wins" | Match. decider2 also gets DMN's own "evaluation can halt", which jDMN does not (§10.1) |
| **Hit policy: Collect (C)** | Returns the list of all hits in arbitrary order (`:3546-3548`) | Not expressible — the kernel returns one `int` row index | Missing; needs a different return shape (see §B-9) |
| **Collect + aggregator `+ < > #`** | sum / min / max / count over the matched outputs; undefined over compound outputs (`:3557-3563`, `:5866-5874`) | Not expressible | Missing. Cheap to add *for numeric single-output tables* as a second scan mode; this is how scorecard-style additive tables are written in DMN |
| **Hit policy: Rule order (R)** | All hits in rule order (`:3545`) | Not expressible | Missing, low value |
| **Hit policy: Output order (O)** | All hits in decreasing output-priority order; SHALL be single output (`:3543-3544`, `:5896`) | Not expressible | Missing, low value |
| **Unary test: literal equality** | `"HIGH"`, `5` | `EqExpression` against a per-row scalar column (`tables/schema.py:489-532`); strings route through a hoisted matcher and store a literal *index* (`:511-525`) | Match, with the string caveat: adding a new distinct string literal is a *shape* change (`tables/codegen.py:44-52`) |
| **Unary test: comparison `<, <=, >, >=`** | Grammar rule 5 / 13 (`dmn15.txt:3900-3912`, `:4826-4834`) | Only as a degenerate `between` with one bound (`BetweenExpression`, `:300-400`). No standalone `lt`/`gt` expression kind | **Partial.** A one-sided bound is expressible but only via the neighbour-fill machinery, which imposes "only row 0 may have an open lower edge" (`:367-375`) |
| **Unary test: interval, all four bracket forms** | `[a..b]`, `[a..b)`, `(a..b]`, `(a..b)`, plus `]a..b[` spellings (§9.2 grammar rules 6-10, `dmn15.txt:3901-3911`) | Two forms only: `BoundMode.lower_inclusive` → `[lo, hi)`, `upper_inclusive` → `(lo, hi]` (`tables/schema.py:144-149`, emitted at `:387-389`). `both_inclusive`/`both_exclusive` are **deliberately** absent because "they create overlaps or gaps at shared boundaries" (`:56-58`) | **Different on purpose, and defensible** — but it means a DMN table with `[0..18]` and `[19..65]` (integer-banded, both-inclusive) cannot be imported without rewriting the bounds |
| **Unary test: comma list (disjunction)** | `simple positive unary tests = simple positive unary test , {"," , ...}` (grammar rule 11, `dmn15.txt:3915`) — so `<10, >20` is one cell | `InExpression` for set membership over a per-row list column, compiled as a CSR scan (`tables/schema.py:403-465`), and `OrExpression` for arbitrary disjunction (`:281-297`) | Match for value lists; `OrExpression` covers mixed-operator disjunction but at the whole-expression level, not per cell |
| **Unary test: negation `not(...)`** | Grammar rule 12.b / 15.b (`dmn15.txt:3921`, `:4834`) | **Absent, and the code depends on its absence**: `AndExpression.to_dnf()` comments "decider 1's table vocabulary has no NOT, so every expression is already monotone and this terminates without negation pushing" (`tables/schema.py:259-262`) | **Missing.** Adding `not` means negation-normal-form pushing before DNF. Note trees *do* have it: `TLogicOp` includes NOT (`trees/schema.py:160`, applied at `:873-876`) — so the two interiors disagree |
| **Unary test: `-` (irrelevant)** | Grammar rule 12.c / 15.c; "The input cell entry '-' means 'irrelevant'" (`dmn15.txt:3307`, `:3919-3921`); when no input values are supplied it means `e != null` (`:5843`) | Implicit: a `None` in a bound/value column makes the condition vacuous for that row (`_bool([v is not None ...])` guards at `tables/schema.py:393-394`, `:520-521`, `:530`) | **Match in effect, but not in meaning.** decider2's null-means-irrelevant is silent, and it collides with DMN's "`-` also asserts not-null". A reviewer cannot tell a blank cell from a missing cell |
| **Generalized unary test (`?`)** | DMN 1.2+: "a decision table input entry may be any FEEL expression, substituting `?` for the input expression" (`dmn15.txt:9529-9533`) | Not expressible, and correctly so: doc 03 §10 bans `{"expression": "a - b"}` config outright; `trees/schema.py:188-236` raises `ComputedFeatureRemoved` | **Deliberate divergence, keep it.** A derived value is a step placed before the table |
| **Input values (allowed-value list per input clause)** | Optional per input clause; "If provided, it is a list of unary tests that must be satisfied" (`dmn15.txt:3366-3368`); enables domain-completeness checking | Absent. `ParametersConfig.dtypes` names types, not domains (`tables/schema.py:159-174`) | **Missing, and it is the enabler for gap analysis** — without a declared input domain you cannot say a numeric column is "covered" |
| **Output values list** | Drives P and O priority ordering (`:3388-3391`) | Absent | Missing (follows from no P/O) |
| **Output typing** | Output clause carries `typeRef`; compound outputs carry names (§8.2.7 body [8.2.6]) | Inferred from the data: bool if all bools, int if declared `Int64`/all ints, else float (`tables/codegen.py:239-246`). **String outputs are dropped from the kernel entirely** (`:237`) and re-attached in polars by `TableModule.decode()` (`tables/build.py:62-90`) | **Partial.** Inference-from-data is fragile: an all-integer column of a genuinely float quantity silently becomes `int64`. DMN's declared `typeRef` is better governance |
| **Compound (multiple) outputs** | Supported; restricts hit policies (aggregation undefined over compound outputs, `dmn15.txt:3563`, `:5872-5874`) | Supported: `outputs: list[str]`, one emitted step per column (`tables/schema.py:569`, `tables/codegen.py:234-271`), tested at `tests/test_tables_ported.py:165` | Match, and decider2's is cheaper (one row index, N output reads) |
| **Default output** | "A decision table may have no rule hit ... the result is given by the default output value, or null if no default output value is specified. **A complete decision table SHALL NOT specify a default output value**" (`dmn15.txt:5875-5878`; §8.2.12 [8.2.11]) | `default: list[Any] | None`, length-checked against `outputs` (`tables/schema.py:570`, `:579-583`), read when the scan returns `-1` (`tables/codegen.py:267-269`); tested `tests/test_tables.py:131` | **Match, except the completeness rule.** decider2 cannot detect a complete table, so it cannot enforce (or exploit) the SHALL NOT |
| **Gap / overlap validation** | Not mandated by the spec, but the spec's U policy presupposes it | Only *one-dimensional and only for `between`*: `allow_gaps=False` requires each row's upper to equal the next row's lower (`tables/schema.py:376-382`); trees have the sibling check for `CasesRanges` (`trees/schema.py:881-934`) | **Biggest single gap.** jDMN ships three sweep-line validators over *all* input columns jointly (§10.1); decider2 has no cross-column analysis at all |
| **Rule annotations** | Zero or more annotation clauses per table; "implementations may use the annotations for auditing, debugging, logging, documentation, analytics" (`dmn15.txt:3520-3523`); validated for presence (jDMN `DefaultDMNValidator.java:504-522`) | Absent. A row has no id, no label, no annotation. `EmittedCondition` carries `kind`/`variable` for introspection (`tables/schema.py:120-121`) but nothing per row | **Missing, and it matters for doc 04 §6.** The row index is the only identity a matched row has (`tables/build.py:57-60`) — an unstable identity if rows are reordered, which is exactly the failure doc 04 §5.1 warns about |
| **Rule id stability** | Rules carry XML `id` attributes (`0115-...dmn` `rule id="DecisionRule_07toq2m"`) | Row index only | Missing; see above |
| **Merged input entry cells / crosstab** | §8.2.9 body [8.2.8]; crosstab tables "are always Unique and need no indicator" (`dmn15.txt:3496-3497`, `:3583`) | Absent | Not worth adopting — presentation, not semantics |
| **Table orientation / hit indicator shorthand** | §8.2.2, §8.2.10.1 body (`dmn15.txt:3292`, `:3454`) | Absent | Not worth adopting (presentation) |
| **Evaluation order** | "Every rule in the rule list is matched with the input expression list. **Matching is unordered**" (`dmn15.txt:5883`) — order only matters for F and R | Strictly ordered scan, early return (`tables/codegen.py:217-218`) | Divergence that is *invisible* for F and *incorrect* for U/A/P/C. Worth stating in docs |

---

## 2. Trees vs DMN

**DMN has no tree node.** The metamodel offers decision tables, literal expressions,
contexts, relations, invocations, function definitions, lists and (1.5) conditional/iterator
boxed expressions. A strategy tree is therefore expressed in DMN as one of:

1. **A DRD of chained decisions**, each a small decision table — this is exactly what the TCK's
   own credit example does: `0004-lending.dmn` decomposes into `ApplicationRiskScore` →
   `PreBureauRiskCategory` → `BureauCallType` → `Eligibility` → `Strategy` → `Routing`, each a
   table, wired by information requirements (see the generated `Strategy.py:92-93`, which calls
   `self.bureauCallType.apply(...)` and `self.eligibility.apply(...)` before its own rules).
2. **Nested contexts / BKMs** as reusable functions (`dmn15.txt:1015`, `:1152`).
3. **Decision services** to expose a subgraph as one callable unit (`dmn15.txt:1053`).

`decider2` takes the opposite route: a first-class v3 tree document (`trees/schema.py:1055-1122`)
with `PositionedNode`s, `MultiSourceEdge`s carrying `sourceIndex` lists (`:981-1003`), node kinds
`leaf` / `unary` / `cases:{ranges,string_match,isin}` / `composite` (`:952-965`), and a
`TreeOutput` leaf-value table selected by `result_idx` (`:1006-1043`). It compiles to nested
`if`/`elif` — doc 08 §3.4's "needs a switch over node types ⇒ codegen"
(`trees/codegen.py:3-8`).

Comparison that matters:

- **Path capture.** decider2 emits `<name>_path` returning the reached leaf's `result_idx`
  (`trees/codegen.py:452-462`, `:535`). DMN has no equivalent: the closest thing is the chain of
  `startDRGElement`/`matchRule` listener events (§7), which is a runtime trace, not an emittable
  column. `example_projects/04-campaign-targeting-trees.md` §1 makes path capture a hard
  requirement ("a targeting system that produces correct answers and no paths is ... a failed
  system") — DMN would satisfy it only by decomposing each tree into per-level decisions and
  emitting each one's matched-rule id, i.e. 4–12 extra DRG elements per tree × 60 trees.
- **Leaf values are emitted source in a tree, arrays in a table.** `trees/codegen.py:557`
  emits `return {_literal(row.get(column), py_type)}`; `tables/codegen.py:252-258` puts the
  same data in `shared`. So editing a tree leaf costs a staged compile and editing a table
  output cell costs nothing (`tables/codegen.py:224-233` says so explicitly). DMN draws no
  such line — everything is a model edit.
- **Short-circuiting.** decider2's tree takes one branch (doc 03 §8.2, "only the taken arm
  executes"). jDMN's generated table evaluates every rule unconditionally (§10.1). For a
  12-level campaign tree that is the difference between 12 comparisons and 2^12 rule bodies.
- **What DMN has that trees lack:** a declared *interface* per sub-decision. decider2's tree is
  one module with `required_features()`/`required_params()` derived from the whole document
  (`trees/schema.py:1126-1136`) — there is no sub-tree interface, so a 400-node tree is one
  audit unit. DMN's DRD gives you 12.

---

## 3. Expression language and null semantics

| | FEEL / S-FEEL | `decider2` |
|---|---|---|
| Numbers | `decimal`, 34 significant digits; jDMN's generated Python uses `decimal.Decimal` throughout (`PreBureauRiskCategoryTable.py` rule signatures: `applicationRiskScore: typing.Optional[decimal.Decimal]`) | float64 / int64 in numba. `Decimal` **cannot cross the boundary — "it raises a Rust panic"** (doc 03 §1.2, citing doc 05 §1.5); money is scaled int64 cents; there is an `EntryMode.SCALED_INT64` for polars `Decimal` columns (`boundary/dtypes.py:91`) |
| Rounding | FEEL `decimal(n, scale)`, half-even by spec | Measured divergence: `round(2.675, 2)` is 2.67 in CPython, **2.68** in njit, 2.68 in `Decimal` HALF_UP (doc 03 §1.2 table). `round_half_up` is specified but **NOT BUILT** (doc 03 §1.2 admission) |
| Three-valued logic | Explicit truth tables (§10.3.2.4 `dmn15.txt:5289-5292`; Table 50 at `:6246-6263`; Table 51 negation at `:6267-6276`). Note it is *not* strict Kleene: `false and otherwise = false`, `otherwise or true = true` | **No null inside the kernel at all, by design.** Nulls are resolved at the boundary into four tiers: REQUIRED → *routed* to a `Decision` (`boundary/nulls.py:311-379`), MISSING_AS / NOT_APPLICABLE_AS → filled at extraction with distinct reason codes (`:102-135`, `:225-...`), OPTIONAL → numba `Optional` per row (`:14-16`) |
| Null propagation | Any FEEL operator on `null` yields `null` (or `false`, per Table 50) | A filled null is indistinguishable from a real value inside the step body — **that is the stated design goal** (`boundary/nulls.py:8-13`, module docstring) |
| Dates / durations | First-class `date`, `time`, `date and time`, `days and time duration`, `years and months duration`; jDMN imports `datetime` + `isodate` in every generated file | `EntryMode.AS_INTEGER` — Date/Datetime/Duration enter as their physical integer (`boundary/dtypes.py:89`). No duration arithmetic vocabulary |
| Contexts / lists / filters | §10.3.2.5 lists and filters, `for`/`some`/`every`, context construction | None. A kernel takes scalars. `param([1,2,3])` is explicitly rejected as a kernel input: "reflected lists are deprecated and `typed.List` is slow" (doc 03 §4.4) |
| External functions | Java method or **PMML model** as an externally-defined function (§10.3.2.13.3, `dmn15.txt:6072-6087`); "if PMML invocation results in a single predictor output, the result ... is the single predictor output's value" | None. A model step would be a Python step, which forces a kernel split (doc 02 §3.2, EXPERIMENTS §B: "a nopython driver cannot call Python") |

**The honest summary:** FEEL and decider2's compiled scalar Python are not comparable
languages. FEEL is a small interpreted functional language with decimals, temporal types and
ternary logic; decider2's step language is Python-that-numba-accepts over float64/int64 with
nulls eliminated upstream. Any DMN import must therefore be *restricted to S-FEEL over numbers,
booleans and strings* — which, usefully, is exactly DMN Conformance Level 2 (`dmn15.txt:412-415`:
CL2 adds "the simple expression language (S-FEEL)"; CL1 "is never required to interpret
expressions", `:407-409`).

---

## 4. Typing and schema

- **DMN `itemDefinition`** (§7.3.3 `dmn15.txt:2841`) gives named structured types with
  `itemComponent` children, `typeRef`, `isCollection`, and an `allowedValues` unary-test
  constraint. jDMN generates a Python interface + impl class per itemDefinition
  (`templates/dmn/python/common/itemDefinitionClass.ftl`, and the generated
  `type_.TApplicantDataImpl`). DMN can also import an XSD or a PMML file (`dmn15.txt:1762`).
- **`decider2`** has no nominal types. An interface is *inferred* from step signatures
  (`graph/interface.py:46-110`) and materialised as `Interface` with `Input`/`Output` records;
  structure enters through the `ColumnPlan` plan classes at `boundary/dtypes.py:192-311` (dtype +
  tier + entry mode + nullability), built by `plan_column` (`:313`); nullability is explicitly
  **not** inferable from polars
  (`boundary/dtypes.py:197-201`). `contract=True` freezes an interface to a checked-in JSON file
  (doc 03 §5.1) — that is decider2's closest analogue to an itemDefinition, and it is per module,
  not per type.
- **Where DMN is better:** `allowedValues` on a type is a domain declaration reusable across
  every table that reads that input. decider2 has bounds on *params* (`param(48.0, ge=6, le=60)`,
  `params.py:250`) but nothing on *inputs*.
- **Where decider2 is better:** the dtype ladder is a cost model, not just a type system —
  `DtypeTier.ZERO_COPY/COPY/CONVERT` (`boundary/dtypes.py:73-79`) tells you what a type choice
  costs at the boundary. DMN has no such concept because it never crosses one.

---

## 5. Compile-ahead vs interpret

| Engine | Model | Evidence |
|---|---|---|
| **jDMN** | Both. "The decisions can be interpreted or translated to Java and executed on a JVM" (`jdmn/README.md:13`); cross-transpilers to Java, Kotlin, **Python** (`docs/getting-started.md:7`). Python dialect is `JavaTimePythonStandardDMNDialectDefinition.java`. Generated code is one class per DRG element, one method per rule, plus a `*RuleOutput` value class with `__eq__`/`__hash__` for the ANY/PRIORITY policies | `Strategy.py`, `StrategyRuleOutput.py` in `dmn-test-cases/.../0004-lending/translator/expected/python/dmn/` |
| **Drools / KIE** | DMN models are evaluated per request by the DMN engine; DRL has a separate "executable model". Drools DMN docs state "runtime support for DMN 1.1, 1.2, 1.3, and 1.4 models at conformance level 3" and say nothing about DMN codegen | Drools DMN docs, "DMN conformance levels" |
| **Kogito** | Build-time codegen (quarkus extension). Runtimes repo archived 2026-07-23, merged into `incubator-kie-drools` *(landscape pass)* |
| **Camunda 7/8** | Interpreted. DMN 1.3, FEEL via feel-scala (Apache-2.0, "Full support for unary-tests and expressions") | feel-scala README; Camunda DMN engine docs |
| **pyDMNrules** | Interpreted, per row (below) | pyDMNrules README |
| **`decider2`** | Compile-ahead to numba nopython kernels, content-addressed on disk (`compile/cache.py:1-35`), with a hard 500-emitted-line cap per kernel (`trees/codegen.py:86`, enforced at `tables/codegen.py:274-281`) because compile time is ∝ lines^1.4 (`trees/codegen.py:36-46`) | |

**Published numbers.** Neither jDMN, Drools, Camunda nor pyDMNrules publishes per-record
latency or compile time on any page fetched. jDMN has a `dmn-performance` Maven module with
`run-performance.bat` but no results committed. The only numbers in the whole open-source set are
vendor README benchmarks: Grule 100 rules ≈ 0.0097 ms / 1000 rules ≈ 0.569 ms, loading 1000
rules 933.6 ms and 488 MB; Durable Rules Miss Manners-128 450 ms Node / 600 ms Python; RuleGo
memory on a Raspberry Pi 2; OpenRules "100,000+ requests/second with 16,369-rule tables" (no
methodology); Higson "over 9,000 API calls per second", "0.23 ms"; Nected "~300 rps"; GoRules
ZEN 91k eval/s *(all landscape pass)*. **None is measured on a comparable workload and none
should be compared to decider2's own figures.**

Against that, decider2's measured envelope is unusually well documented: 20 ms single-record
budget, 400-in/633-out target boundary verified (`docs/02-architecture.md:34-38`), dispatch
0.44 µs/kernel so 30 unfused kernels ≈ 13 µs = **0.07% of budget** (`02-architecture.md:132-133`),
`.emit()` at +0.11 ns/row, `activate()` at 0.177 µs, a threshold retune at **0 compile events**
(doc 08 §2.1), and the counter-results that changed the design (fusion non-monotone; 400 literal
kwargs = 5.95% of budget; `nogil=False` p99 = 1270% of budget at 16 threads).

---

## 6. Batch execution

**No open-source DMN engine vectorises.** Findings:

- **pyDMNrules** is the only one with a DataFrame entry point: `decidePandas(dfInput)` returns
  `(dfStatus, dfResults, dfDecision)` — and "the implementation processes rows individually (not
  vectorized)" (README fetch). Its `dfDecision` frame is interesting for a different reason: it
  carries `DecisionName, TableName, RuleId, DecisionAnnotation, RuleAnnotation` per row, i.e. a
  *tabular trace* aligned to the result frame. That is structurally what decider2's
  `.emit("<table>_row")` gives (`tables/build.py:56-60`) but with names instead of indices.
- **jDMN** generated code is per-record object graphs; batching means a loop.
- **Drools / Camunda** evaluate one context per call.
- **GoRules ZEN** Python binding is a per-record FFI call *(landscape pass)*.

`decider2` is the only engine in this set where batch and realtime are the *same* kernel, and
it tests that: `testing/equivalence.py:218` `assert_equivalent` runs
`interpreted ≡ stepped ≡ fused` and then a fourth rung comparing `score()` per row against the
fused batch frame (`:160-216`), with an explicit message naming which rung broke. Two honest
caveats in that file: rows that `score()` *routes* under the null policy are skipped
(`:178-183`), and **any pipeline with a `str` input is skipped entirely** (`:194-197`) because
`runtime.invoke.score` has no dtype ladder. Since decision tables on string columns are normal
in credit, that exclusion is load-bearing — `tests/test_tables.py:242`
(`test_score_takes_a_string_input_and_agrees_with_apply`) suggests the gap is narrower than the
docstring implies, but the generic assertion still opts out.

---

## 7. Trace and explainability

**jDMN is the strongest model here and it is worth copying the shape.** Every generated
decision has:

- a class-level `DRG_ELEMENT_METADATA` naming the namespace URI, element name, element kind,
  **expression kind** and **hit policy** as data (`Strategy.py:47-55`);
- `eventListener_.startDRGElement(...)` / `endDRGElement(..., output_, elapsed_µs)` around every
  decision, with an `Arguments` bag keyed by fully-qualified input name (`Strategy.py:70-79`);
- per-rule `startRule` / `matchRule` / `endRule` with a `Rule(index, annotationText)` record
  (`Strategy.py:114-137`) — so "which rule fired, in which decision, with what inputs, in how
  long" is a built-in runtime event stream, not an opt-in;
- a `NopEventListener` default so the instrumentation costs a virtual call when unused
  (generated test file imports `jdmn.runtime.listener.NopEventListener`).

Drools exposes the same idea as configuration: `org.kie.dmn.runtime.listeners.$LISTENER_NAME`
"loads and registers a DMN Runtime Listener onto the Drools DMN engine at start time" (Drools
DMN docs, "Configurable DMN properties"). Camunda 7 records DMN decision evaluation in process
history when run with the process engine *(landscape pass; the specific listener class names
were not on the fetched page — see §13)*.

`decider2` deliberately does **not** do per-record eventing: doc 04 §4.3 says OTel spans wrap
"module and kernel boundaries ... Per-record spans would be millions per batch and are never
emitted", and per-record diagnostics travel as *columns* — `.emit("term_cap@*")`,
`.emit("<Branch>_path")` at +0.11 ns/row (doc 04 §4.1). Doc 04 §6.5 goes further: "trace is
data; every rendering is replaceable".

Assessment: decider2's column-shaped trace is the right call for 14.2 M-row batch (jDMN's
listener model would emit ~10^8 events). What decider2 is missing is not the *mechanism* but the
*metadata*: jDMN's `DRG_ELEMENT_METADATA` (kind, expression kind, hit policy, element id) and its
per-rule annotation text are exactly the fields doc 04 §6.1's failed reviewer test was missing —
"the sheet rendered the adjustment; it did not render the policy constraint the adjustment
broke". A matched row index with no annotation and no rule id cannot carry that.

---

## 8. Validation and testing tooling

**jDMN's sweep validators are the single most transferable piece of engineering in this survey.**
`dmn-core/src/main/java/com/gs/dmn/validation/` contains:

- `SweepRuleOverlapValidator.java` — finds sets of rules that overlap across *all* input columns
  jointly, builds an overlap graph and reports "Decision table rules '%s' overlap in decision
  '%s'" (`:110-113`);
- `SweepMissingIntervalValidator.java` — per column, reports uncovered intervals;
- `SweepMissingRuleValidator.java` — reports whole missing *rules* (uncovered cells of the input
  cross-product), with a `merge` flag that also *simplifies* by merging adjacent rules
  (`:24-42`);
- supporting types `Bound`, `BoundList`, `Interval`, `NumericInterval`, `EnumerationInterval`,
  `MissingIntervals`, `MissingRuleList`, `RuleGroup`, `Table`, `TableFactory` in
  `validation/table/`.

The algorithms are cited in-source to **"Semantics and Analysis of DMN Decision Tables"**
(`SweepRuleOverlapValidator.java:116-125`), and both that paper and
"Semantics, Analysis and Simplification of DMN Decision Tables" are vendored in
`jdmn/docs/articles/`. Plus: `TypeRefValidator`, `UniqueRequirementValidator`, five cyclic-
dependency validators, and `DMNModellingStyleValidator`.

Drools ships the same capability behind a Maven flag: `ANALYZE_DECISION_TABLE` — "DMN decision
tables are statically analyzed for gaps or overlaps and to ensure that the semantic of the
decision table follows best practices" (Drools DMN docs, "Configurable DMN validation").
Trisotech claims equivalent checks *(landscape pass)*.

**TCK-style fixtures.** The DMN TCK format is small and worth copying. A test case is
"a serialized DMN model, a serialized set of input data, serialized set of output/response data"
plus a human-readable representation (TCK README), CC-BY-SA licensed. Concretely
(`0115-sum-collect-hitpolicy-test-01.xml`): a `<testCases>` root with `<modelName>`, a
`<labels>` block that tags what the case exercises (`Hit Policy: COLLECT`, `Aggregator: SUM`,
`Data Type: Number`), then `<testCase id>` with `<inputNode name>` values and `<resultNode
name type="decision"><expected>`. jDMN compiles these into generated `unittest` classes
(`_0004LendingTest01Test.py:45-55`: build typed inputs, then
`self.checkValues("ACCEPT", Adjudication.Adjudication().apply(...))`).

`decider2`'s equivalent is `testing/corpus.py` + `testing/equivalence.py` +
`golden.record(...)` (doc 04 §7), which are *property* tests (modes agree) rather than
*conformance* fixtures (this input yields this output, and here is why this case exists). Doc 03
§1.2 already identifies the missing piece: "The corpus must include boundary values. The
overflow above was found by binary search, not by sampling". The TCK's `<labels>` mechanism is
precisely a machine-readable statement of *which semantic each fixture pins* — which is what a
regression suite for hit policies, bound modes and null tiers needs.

---

## 9. Interop: what `decider2` would gain

**(a) Importing DMN CL2 decision tables — worth doing, bounded.** A CL2 table is
S-FEEL-only: literals, comparisons, intervals, comma-lists, `not(...)`, `-`
(`dmn15.txt:3915-3921`). Mapping onto `tables/schema.py`: interval → `BetweenExpression`
(bracket forms 3 and 4 need new `BoundMode` members, or rejection with a clear message);
comma-list of literals → `InExpression`; single literal → `EqExpression`; boolean → `IsTrueExpression`;
multiple input clauses → `AndExpression` of per-column conditions; `-` → `None` in that row's
column. What cannot be imported without new schema: `not(...)`, a U/A/P/C hit policy, an
`allowedValues` domain, a rule annotation. That is the import's specification *and* its
feature-gap list.

**(b) Exporting DMN for editors — high leverage, low cost.** `dmn-js` (bpmn.io licence) and
the kie-tools DMN editor (Apache-2.0) are free, mature, browser-based decision-table editors
*(landscape pass)*. decider2 currently has no editor and doc 04 §6 records two failed attempts at
a bespoke reviewer artefact. Exporting a `DecisionTable` as a DMN 1.3 `<decisionTable
hitPolicy="FIRST">` is mechanical for the numeric/string cases and would give authoring and
review UIs for free. The round-trip is lossy in one direction only (DMN → decider2 loses
policies decider2 lacks), which is the right way round.

**(c) PMML scorecard/tree import — the strongest ML-interop case.** PMML is *normatively
referenced by DMN 1.5* (`dmn15.txt:517-518`, DMG PMML 4.2) and a PMML model is a first-class
externally-defined function (`:6072-6087`). PMML's `Scorecard` (bins → partial scores →
reason codes, with `baselineScore` and ranked reason codes) and `TreeModel` map almost exactly
onto decider2's two existing interiors: a PMML `Scorecard` is a `DecisionTable` per
characteristic with numeric outputs summed; a PMML `TreeModel` is a `Tree` with `SimplePredicate`
/`SimpleSetPredicate` nodes ≈ `UnaryNode`/`CasesIsIn`. **PMML reason codes are the decline-reason
taxonomy doc 04 §4.1 says was dropped in a previous port** — importing them gets that vocabulary
for free rather than re-inventing it.

**(d) ONNX for model steps — do not.** ONNX is named on the OMG DMN listing page as a
compatible format but claimed by no vendor in this survey *(landscape pass)*. An ONNX step
inside decider2 cannot be numba-compiled, so it forces a kernel split at exactly the point
EXPERIMENTS §B says is most expensive. The right shape is a model *scored upstream*, its output
arriving as an input column — which needs no framework feature at all.

---

## 10. Per-engine blocks

### 10.1 jDMN (Goldman Sachs, Apache-2.0) — the closest compile-ahead analogue

The only perfect DMN TCK score (10.0.0, level 3, 3391/3391) *(landscape pass)*, and the only
engine in this survey that transpiles DMN to **Python**. Its Python runtime ships as
`jdmn-python-runtime` on PyPI, **1.0.5, released 2025-02-21, Apache-2.0, Python ≥3.10**
(PyPI JSON API).

What the generated Python actually looks like (`.../0004-lending/translator/expected/python/dmn/`):

- One class per DRG element extending `DefaultDMNBaseDecision`, with `apply()` (instrumented)
  delegating to `evaluate()` (`Strategy.py:62-110`).
- Sub-decisions are constructor-injected and called at the top of `evaluate()`
  (`Strategy.py:57-60`, `:92-93`) — so the DRD becomes a hand-wired object graph.
- One `rule{N}()` method per rule, each building a `*RuleOutput` and setting `matched`
  (`Strategy.py:112-139`).
- **Every rule is evaluated, always.** `evaluate()` calls `rule0`, `rule1`, `rule2` in sequence
  and *then* applies the hit policy (`Strategy.py:96-108`); the template has no short-circuit
  branch even for FIRST (`templates/dmn/python/tree/common/apply.ftl:101-112`). The DMN spec
  explicitly permits halting on FIRST (`dmn15.txt:3536-3537`) and jDMN declines to.
- Hit policies live in the *runtime*, not the generated code:
  `RuleOutputList.applySingle(HitPolicy)` implements U (size≠1 ⇒ `null`, **not an error**), ANY
  (distinct-set size must be 1 else `null`), FIRST (first matched) and PRIORITY (sort, take
  first) — `dmn-runtime/.../RuleOutputList.java:39-76`; `applyMultiple` covers COLLECT /
  RULE ORDER / OUTPUT ORDER (`:78-89`).
- Aggregators are generated as Python one-liners over the matched list:
  `self.sum(list(map(lambda ..., ruleOutputs_)))` etc. (`PythonFactory.java:179-205`).
- `*RuleOutput` classes carry generated `__eq__`/`__hash__` because ANY needs value equality
  (`PreBureauRiskCategoryTableRuleOutput.py:19-34`).
- Numbers are `decimal.Decimal` via `self.number("100")` (`PreBureauRiskCategoryTable.py`
  rule bodies) — which is why this output can never be numba-compiled.
- Intervals compile exactly as decider2's `between` does:
  `booleanAnd(numericGreaterEqualThan(x, number("100")), numericLessThan(x, number("120")))`
  vs decider2's two guarded compares (`tables/schema.py:396-399`).
- Validation: the three sweep validators plus type-ref, cyclic-dependency and modelling-style
  validators (§8).

**Verdict:** jDMN is the reference for *what a compile-ahead DMN engine must cover* and for
*hit-policy and validation semantics*. Its generated code shape is the opposite of what decider2
needs (objects, Decimal, no short-circuit, per-rule instrumentation, no batch) — so it is a
specification to read, not a runtime to adopt.

### 10.2 Drools / Apache KIE (Apache-2.0)

DMN CL3 for 1.1–1.4 (Drools DMN docs, "DMN conformance levels"), 3388/3391 TCK *(landscape
pass)*. Three things stand out against decider2: (1) `ANALYZE_DECISION_TABLE` static analysis
for gaps/overlaps, configurable per build via `<validateDMN>` in `pom.xml`, with options for
schema-only / model / decision-table analysis / off — a good model for a `decider2 check`
subcommand's granularity; (2) pluggable DMN runtime listeners via a single system property
(`org.kie.dmn.runtime.listeners.$NAME`); (3) SceSim test-scenario editors in `kie-tools`
(Apache-2.0) *(landscape pass)*. Against it: PHREAK forward chaining with `insertLogical()`
truth maintenance, `modify()` re-evaluation and `salience` is the wrong execution model for
stateless per-record scoring, and GoRules' own migration page says these "have no structural
equivalent in a stateless DAG" *(landscape pass)*. Kogito runtimes archived 2026-07-23 and
merged into `incubator-kie-drools` *(landscape pass)* — a consolidation risk signal for anyone
betting on the Kogito codegen path specifically.

### 10.3 Camunda DMN (Apache-2.0 engine, source-available platform)

DMN 1.3 (`docs.camunda.org` DMN engine page, verbatim: "version 1.3 of the OMG DMN standard").
Hit policies **U, A, F, R, C** with aggregators SUM/MIN/MAX/COUNT, and "if the Collect hit
policy is used with an aggregator, the decision table can only have one output"
(docs.camunda.org hit-policy page) — **PRIORITY and OUTPUT ORDER are absent from the supported
list**, which is a useful signal: the two policies that require an *output values* ordering are
the two a mainstream engine skipped. TCK 2741/3391 for Camunda 7.21.0 and 2850/3391 for
DMN-Scala 1.9.0 *(landscape pass)*. FEEL is `feel-scala` (Apache-2.0), "Full support for
unary-tests and expressions", usable standalone as a library or JSR-223 script engine
(feel-scala README) — the one component here that could plausibly be reused in isolation, except
that it is Scala/JVM. The Camunda 7 CE platform repo was archived 2025-11-04 *(landscape pass)*.
Camunda 8's `bindingType = latest | deployment | versionTag` is a neat answer to the decision-
version-pinning question decider2 answers with a structure fingerprint + artefact id (doc 04
§5.2) *(landscape pass)*.

### 10.4 The long tail, one paragraph each

**pyDMNrules** (GPL-3.0, 1.4.5 2026-08-22) — Excel-authored DMN tables, S-FEEL via `pySFeel`,
`decide()` returning `{'Result', 'Executed Rule': (Decision, Table, RuleNumber),
'DecisionAnnotations', 'RuleAnnotations'}`, `decidePandas()` returning result + decision frames
row-by-row, and a built-in `test()` driven by a 'Test' sheet in the same workbook. The
*decision frame* idea (`DecisionName, TableName, RuleId, annotations`, aligned to results) and
tests-in-the-artefact are both worth stealing; the GPL licence rules out using the code in bank
software. **Trisotech** — commercial DMN reference implementation, CL3, 3390/3391, PMML execution
engine, browser simulation, audit logging; the useful datum is that its TCK score and Drools'
bracket jDMN's, i.e. CL3 is genuinely achievable and genuinely rare. **OpenRules** — commercial
(openrules.com now redirects to openrules.ai), Excel/Sheets tables, sequential execution,
"100,000+ requests/second with 16,369-rule tables" with no methodology, TCK 7.0.0 stale at
0/3391; names J.P. Morgan, RBS, Commerzbank. **Easy Rules** (MIT) — maintenance mode since Dec
2020, last release 4.1 (Jun 2020); annotations/fluent/MVEL/YAML rules with priorities. Do not
build on it. **json-rules-engine** (ISC, 7.3.1 Jan 2025) — JSON `all`/`any` condition trees,
priorities, JSONPath facts, an "almanac" fact cache; the almanac is a memoisation idea decider2
gets for free from step purity. **Grule** (Go, Apache-2.0, v1.20.4) — Drools-like GRL with
salience and "a form of RETE"; its published load cost (1000 rules ≈ 934 ms, 488 MB) is the only
public data point on rule-loading cost and it is worse than decider2's compile budget.
**RuleGo** (Go, Apache-2.0, v0.37.0) — JSON rule-chain DAG of components with a visual editor,
IoT-first; relevant only as evidence that "JSON DAG + visual editor" is a common shape.
**Microsoft RulesEngine** (MIT, v6.0.0 Jun 2024) — JSON workflows whose rules are
`System.Linq.Dynamic.Core` lambdas; `ExecuteAllRulesAsync` returns a `RuleResultTree` of
pass/fail plus messages, i.e. a minimal explanation record. The lambda-in-config model is exactly
what doc 03 §10 forbids. **durable_rules** (MIT, Node/Python/Ruby over C) — Rete with event
streams and statecharts; PyPI `durable-rules` 2.0.28 dates from Jun 2020. **business-rules**
(venmo, MIT, 1.1.1 Mar 2022) — dormant; JSON rules over decorated variable/action classes.
**rule-engine** (zeroSteiner, BSD-3, 5.0.2 Jul 2026) — a typed expression language with
*parse-time type checking* and `.matches()` over dicts; the type-checked-expression idea is the
one thing here decider2 does differently (it type-checks a signature instead). **Experta**
(LGPL-3.0, 1.9.4 Nov 2019, dormant) and **ClipsPy** (BSD-3, CLIPS 6.42, 1.0.6 Dec 2024) — CLIPS
lineage; wrong execution model, and Experta is unmaintained. **DecisionRules.io** — commercial
SaaS; `POST /rule/solve/{ruleId}/{version}` with strategies STANDARD / FIRST_MATCH / ARRAY /
EVALUATE_ALL (a non-DMN re-spelling of single-hit vs collect) and per-rule audit logs capturing
input+output with configurable retention. **Higson** (ex-Hyperon, commercial Java BRMS) —
decision and *parameter* tables, Excel import, effective dating, versioning, RBAC, audit trail;
effective dating is a real gap in decider2 (§11-8). **Nected** — proprietary; hit policies
First/Unique/Collect/Order, "~300 rps" self-host. *(All long-tail facts: landscape pass.)*

---

## A. Feature inventory

`decider2` column: **yes** / **partial** / **no**, with `file:line`.

| Feature | Engine / spec | What it does | In `decider2`? | Adopt? |
|---|---|---|---|---|
| Hit policy **U** (Unique) | DMN §8.2.11 [8.2.10]; jDMN, Drools, Camunda | At most one rule may match; overlap is an error | **no** | **yes** — as a *validated* mode over the existing scan; it is the only policy with a checkable invariant |
| Hit policy **A** (Any) | DMN; jDMN `RuleOutputList.java:49-59`; Camunda | Overlap allowed if all matches agree | **no** | no — adds a runtime equality check for no credit use case |
| Hit policy **P** (Priority) | DMN §8.2.11; jDMN `:66-72` | Highest-ranked *output value* wins, independent of row order | **no** | **yes** — decline-reason precedence is a real credit need and row order is the wrong encoding for it |
| Hit policy **F** (First) | DMN; jDMN `:60-65` | First match by rule order, may halt | **yes** (only) `tables/codegen.py:217-218` | n/a — already the default |
| Hit policy **C** (Collect, no operator) | DMN §8.2.11; jDMN `:80-81` | List of all matched outputs | **no** | maybe — needs a list-shaped return the kernel cannot produce; better as "N boolean emit columns" |
| Hit policy **C+ / C< / C> / C#** | DMN `dmn15.txt:5866-5871`; jDMN `PythonFactory.java:179-205`; Camunda (SUM/MIN/MAX/COUNT) | sum / min / max / count of matched outputs | **no** | **yes for C+** — an additive scorecard table is the canonical credit shape and needs only a second scan that accumulates instead of returning |
| Hit policy **R** (Rule order) | DMN; jDMN `:82-83` | All hits in rule order | **no** | no |
| Hit policy **O** (Output order) | DMN; jDMN `:84-85` | All hits in output-priority order | **no** | no — Camunda skipped it too |
| Unary test: literal equality | DMN §9.2 | `x = "HIGH"` | **yes** `tables/schema.py:489-532` | n/a |
| Unary test: comparison `<,<=,>,>=` | DMN grammar rule 5/13 | one-sided bound | **partial** — only via `between` with one bound + neighbour fill (`tables/schema.py:322-340`) | **yes** — add a first-class one-sided comparison so a non-contiguous threshold table does not need the fill rules |
| Unary test: interval `[a..b)` / `(a..b]` | DMN grammar rules 6-10 | half-open ranges | **yes** `tables/schema.py:144-149`, `:387-389` | n/a |
| Unary test: interval `[a..b]` / `(a..b)` | DMN grammar rules 6-10 | both-closed / both-open | **no**, deliberately (`tables/schema.py:56-58`) | maybe — needed for *importing* integer-banded DMN tables; add as import-time rewrite, not as a new `BoundMode` |
| Unary test: comma list | DMN grammar rule 11 | `<10, >20` in one cell | **partial** — `InExpression` for value sets (`:403-465`), `OrExpression` for mixed (`:281-297`) | no new work |
| Unary test: `not(...)` | DMN grammar rule 12.b | negation of a cell | **no**; `to_dnf()` relies on monotonicity (`tables/schema.py:259-262`). Trees have NOT (`trees/schema.py:160`) | **yes** — the table/tree asymmetry is a trap, and NNF-before-DNF is a contained change |
| Unary test: `-` (irrelevant) | DMN grammar rule 12.c; `dmn15.txt:3307` | cell always matches; also asserts not-null when no input values given | **partial** — `None` in a column is vacuous (`:393-394`, `:520-521`) but silent and carries no not-null meaning | **yes** — make it explicit and renderable |
| Generalized unary test (`?` + FEEL) | DMN 1.2+, `dmn15.txt:9529-9533` | arbitrary FEEL per cell | **no** | **no** — doc 03 §10 bans expressions in config for good reasons |
| Input values (`allowedValues` domain) | DMN §7.3.3, §8.2.4 | declared domain per input, reusable | **no** (`ParametersConfig.dtypes` is types only) | **yes** — it is the precondition for meaningful gap analysis and for a form-rendering UI |
| Output values list | DMN §8.2.6 body | ordered output domain; drives P and O | **no** | yes, if P is adopted |
| Output `typeRef` | DMN §8.2.9 body | declared output type | **partial** — inferred from data (`tables/codegen.py:239-246`) | **yes** — inference can silently pick `int64` for a float quantity |
| Compound (multiple) outputs | DMN §8.2.7 body | several output columns per rule | **yes** `tables/schema.py:569`; `tests/test_tables_ported.py:165` | n/a |
| Default output | DMN §8.2.12 [8.2.11], `dmn15.txt:5875-5878` | value when no rule hits | **yes** `tables/schema.py:570`, `tables/codegen.py:267-269` | n/a |
| "Complete table SHALL NOT have a default" | DMN `dmn15.txt:5877-5878` | completeness ⇒ default forbidden | **no** (cannot detect completeness) | maybe — falls out of gap analysis |
| Gap detection (per column) | jDMN `SweepMissingIntervalValidator.java`; Drools `ANALYZE_DECISION_TABLE` | uncovered intervals per input | **partial** — only `between` contiguity (`tables/schema.py:376-382`), only one column | **yes, highest value** |
| Missing-rule detection (cross-column) | jDMN `SweepMissingRuleValidator.java` | uncovered cells of the input cross-product | **no** | **yes** |
| Overlap detection (cross-column) | jDMN `SweepRuleOverlapValidator.java:110-113`; Drools | rule sets that can both match | **no** | **yes** |
| Table simplification / rule merging | jDMN `SweepMissingRuleValidator(merge=true)`; "Semantics, Analysis and Simplification of DMN Decision Tables" | merge adjacent equivalent rules | **no** | maybe — nice reviewer aid, not correctness |
| Rule annotations | DMN §8.2.11 preamble `dmn15.txt:3520-3523`; jDMN `Rule(index, annotationText)` | free-text per rule for audit/doc | **no** | **yes** — doc 04 §6 needs it |
| Stable rule ids | DMN `rule id=`; jDMN `Rule(index, ...)` | durable identity across edits | **no** — row index only (`tables/build.py:56-60`) | **yes** — doc 04 §5.1 requires deterministic node identity |
| Merged cells / crosstab / orientation / hit indicator | DMN §8.2.2, §8.2.9-10 body | presentation of the table | **no** | no |
| Decision services | DMN §6.3.x, `dmn15.txt:1053`, TCK 0085 | expose a DRD subgraph as one callable interface | **partial** — a `Pipeline`/`Module` with an inferred `Interface` and optional frozen `contract=` (doc 03 §5.1; `graph/interface.py:46-110`) is the same idea | no new work; worth naming the equivalence in docs |
| Business Knowledge Models (BKM) | DMN `dmn15.txt:1015`, `:1152` | reusable parameterised function invoked by decisions | **yes**, as an ordinary step/module | n/a |
| `itemDefinition` structured types | DMN §7.3.3 `dmn15.txt:2841`; jDMN generates classes | named types, components, collections, allowed values | **no** (interface is inferred, not nominal) | **partial yes** — adopt `allowedValues` only |
| Imports (XSD / PMML / other DMN) | DMN `dmn15.txt:1762` | reuse type and model definitions | **no** | no |
| DRD / DRG diagram + DMNDI | DMN §6, DMNDI15.xsd | visual model + layout | **partial** — trees carry `Position` (`trees/schema.py:970-979`); no pipeline diagram (`render()` NOT BUILT, doc 03 §9) | maybe — export instead (§9b) |
| FEEL ternary logic | DMN §10.3.2.4, Tables 50-51 | null-aware and/or/not | **no**, by design — nulls resolved at the boundary (`boundary/nulls.py:8-16`, module docstring) | **no** — decider2's four-tier boundary is stronger for credit (it distinguishes missing from not-applicable, `boundary/nulls.py:102-113`) |
| FEEL decimal numbers | DMN §10.3; jDMN uses `decimal.Decimal` | exact decimal arithmetic | **no** — `Decimal` panics at the boundary (doc 05 §1.5); money is scaled int64 | no — but `round_half_up` is still owed (doc 03 §1.2, NOT BUILT) |
| FEEL date/time/duration | DMN §10.3.4 | temporal types and arithmetic | **partial** — enter as physical ints (`boundary/dtypes.py:89`); no vocabulary | maybe — effective dating needs it (§11-8) |
| FEEL contexts / lists / filters / `for`/`some`/`every` | DMN §10.3.2.5 | collection expressions | **no** | no — kernels take scalars |
| External Java function | DMN §10.3.2.13.3 | call a JVM method | **no** | no |
| External **PMML** model as a function | DMN `dmn15.txt:6072-6087` | invoke a PMML model from decision logic | **no** | **maybe** — as *import to native rules*, not as a runtime call (§9c) |
| PMML `Scorecard` (bins, partial scores, reason codes, baseline) | DMG PMML 4.2, referenced by DMN `dmn15.txt:517-518` | additive scorecard with ranked reason codes | **no** | **yes** — maps onto `DecisionTable` + C+ aggregation, and brings a decline-reason taxonomy (doc 04 §4.1) |
| PMML `TreeModel` | DMG PMML 4.2 | decision tree with Simple/Compound predicates | **no** | **yes** — maps onto `Tree` + `UnaryNode`/`CasesIsIn` |
| PMML `RuleSetModel` | DMG PMML 4.2 | firstHit / weightedSum rule sets | **no** | maybe — `firstHit` is decider2's existing table semantics exactly |
| ONNX | named on OMG DMN listing *(landscape pass)* | ML model exchange | **no** | **no** — forces a kernel split (EXPERIMENTS §B) |
| Runtime listeners / decision events | jDMN `Strategy.py:70-137`; Drools `org.kie.dmn.runtime.listeners.$NAME` | per-decision and per-rule start/match/end events with timing | **no**, deliberately (doc 04 §4.3: no per-record spans) | **no** for events; **yes** for the *metadata* they carry (kind, hit policy, element id, rule annotation) |
| Matched-rule id in the result | DMN "decision result"; pyDMNrules `'Executed Rule': (Decision, Table, RuleNumber)`; MS RulesEngine `RuleResultTree` | which rule produced the answer | **partial** — matched row *index* as an emittable column (`tables/build.py:56-60`); tree `_path` as `result_idx` (`trees/codegen.py:452-462`) | **yes** — upgrade index to stable id |
| Tabular batch trace aligned to results | pyDMNrules `decidePandas` → `dfDecision` | one trace row per input row | **partial** — `.emit()` columns (doc 03 §7, doc 04 §4.1) | already better; document the equivalence |
| Vectorised / columnar evaluation | none | batch without a per-row loop | **yes**, uniquely — one kernel over polars columns | n/a |
| batch ≡ realtime equivalence test | none | same definition, proven agreement | **yes** `testing/equivalence.py:218`, 4 rungs | n/a (note the `str`-input skip at `:194-197`) |
| Executable-model / transpiled codegen | jDMN (Java/Kotlin/Python), Kogito, Drools DRL exec-model | avoid interpretation | **yes**, further than any of them (numba nopython) | n/a |
| Content-addressed compiled-artefact cache | none found | reuse compiled code across processes safely | **yes** `compile/cache.py:1-35` | n/a |
| Emitted-line cap as a build error | none found | bound compile time | **yes** `trees/codegen.py:86`; `tables/codegen.py:274-281` | n/a |
| Hot config swap without recompile | Camunda 8 `bindingType`; Higson versioning; decider2 `handle.stage/activate` | change values in a live service | **yes** and measured (doc 08 §4: `activate()` 0.177 µs) | n/a |
| Effective dating / temporal validity of rules | Higson; InRule; Sparkling Logic *(landscape pass)* | a rule version valid between two dates | **no** | **yes** — `example_projects/04` §3 describes expiring overlays; nothing in `tables/schema.py` or `trees/schema.py` carries validity dates |
| TCK-style conformance fixtures with labels | DMN TCK; jDMN generated `unittest` | model + inputs + expected + *what this case tests* | **partial** — `testing/corpus.py`, `golden.record` (doc 04 §7) are property/regression tests | **yes** — adopt the `<labels>` idea as fixture tags |
| Tests inside the authoring artefact | pyDMNrules 'Test' sheet; SceSim *(landscape pass)* | fixtures live with the rules | **no** | maybe — a `tests:` block on a table/tree document |
| Decision-table static analysis as a build flag | Drools `<validateDMN>` with per-check granularity | opt into schema / model / table analysis | **no** (`decider2 check --policy` exists for policy keys only, doc 03 §1.3) | **yes** — the granularity model is worth copying |
| Champion/challenger, simulation, impact analysis | Sparkling Logic, InRule, DecisionRules *(landscape pass)* | measure a change before shipping | **no** — doc 04 §2 says a param change "needs impact review" and doc 08 §5 specifies a mechanism; not built | maybe — outside this brief's engines |

---

## 11. Improvements for `decider2` — ranked, de-duplicated

Ranking is by (cost of *not* having it in a bank credit setting) × (evidence that a mature
engine considers it table stakes), not by effort.

1. **Cross-column gap and overlap analysis for decision tables.** *Gap:* decider2 checks only
   that one `between` column is contiguous (`tables/schema.py:376-382`); a two-condition table
   can have an unreachable row or an uncovered cell and nothing says so.
   *Demonstrated by:* jDMN's three sweep validators (`SweepRuleOverlapValidator`,
   `SweepMissingIntervalValidator`, `SweepMissingRuleValidator`) and Drools'
   `ANALYZE_DECISION_TABLE`. *Maps onto:* a new `tables/analysis.py` plus a doc-04 §6 renderer
   input. *Effort:* **M** (the algorithm is published and cited in jDMN's own source).
   **Must-have.**
2. **Stable row identity + per-row annotation.** *Gap:* a matched row is an integer index
   (`tables/build.py:56-60`); reorder rows and every stored audit record repoints — the exact
   failure doc 04 §5.1 calls "auto-generated and unstable". *Demonstrated by:* DMN `rule id=`,
   jDMN's `Rule(index, annotationText)`, pyDMNrules' `'Executed Rule'` tuple. *Maps onto:*
   `tables/schema.py` (`ParametersConfig` row `id`/`label` columns), `tables/build.py.decode`,
   doc 04 §5.2. *Effort:* **S.** **Must-have.**
3. **A declared hit policy on the table document, even if only two values are implemented.**
   *Gap:* first-match-wins is hard-coded and undocumented as a *choice*
   (`tables/codegen.py:217-218`). *Demonstrated by:* every DMN engine; Camunda's supported set
   (U, A, F, R, C) shows which subset is commercially sufficient. *Maps onto:*
   `tables/schema.py` `DecisionTable.hit_policy: Literal["FIRST","UNIQUE"]`, with `UNIQUE`
   adding recommendation 1's overlap check as a build error. *Effort:* **S** for the field +
   validation, **M** including `PRIORITY` and `COLLECT+`. **Must-have** for the field;
   nice-to-have for the extra policies.
4. **`COLLECT` with `+` for additive scorecard tables.** *Gap:* an additive scorecard is
   currently N tables plus an adding step. *Demonstrated by:* DMN `C+` (`dmn15.txt:5866-5867`),
   jDMN `makeSumAggregator`, Camunda's SUM aggregator, PMML `Scorecard`. *Maps onto:* a second
   emitted scan in `tables/codegen.py` that accumulates instead of returning on first match.
   *Effort:* **M.** Nice-to-have, but it unlocks PMML scorecard import (item 6).
5. **`not(...)` in the table expression vocabulary, and parity with trees.** *Gap:*
   `AndExpression.to_dnf()` documents its dependence on the absence of NOT
   (`tables/schema.py:259-262`) while `TLogicOp` in trees has it (`trees/schema.py:160`). Two
   interiors with different boolean algebras is a trap for authors and for any importer.
   *Demonstrated by:* DMN grammar rule 12.b. *Maps onto:* a `NotExpression` class plus
   negation-normal-form pushing before `to_dnf()`. *Effort:* **M.** Nice-to-have.
6. **PMML `Scorecard` and `TreeModel` importers.** *Gap:* no ML/analytics interop at all, and
   doc 04 §4.1 records that a legacy decline-reason taxonomy "was dropped in translation" —
   PMML scorecards carry exactly that as ranked reason codes. *Demonstrated by:* PMML is
   normatively referenced by DMN (`dmn15.txt:517-518`) and claimed by GDS Link, Pega, Sparkling
   Logic, Trisotech and Drools *(landscape pass)*. *Maps onto:* `tables/` and `trees/` importers
   producing existing documents — no engine change. *Effort:* **M** each. Nice-to-have, high
   leverage.
7. **DMN CL2 import and DMN 1.3 export for decision tables.** *Gap:* no path to or from any
   editor; doc 04 §6 records two failed bespoke reviewer artefacts. *Demonstrated by:* `dmn-js`
   and kie-tools DMN editors are free and mature *(landscape pass)*; jDMN proves the mapping is
   mechanical. *Maps onto:* new `interop/dmn.py`; import targets `tables/schema.py` classes,
   export writes `<decisionTable hitPolicy="FIRST">`. *Effort:* **M** (import) / **S** (export
   of the subset decider2 can express). Nice-to-have that may be the cheapest route to the
   top-ranked unsolved risk in doc 04 §6.
8. **Declared input domains (`allowedValues`) and declared output types.** *Gap:* output dtype
   is *inferred from the data* (`tables/codegen.py:239-246`) — an all-integer float quantity
   silently becomes `int64`; and no input carries a domain, which is why gap analysis has
   nothing to be complete *against*. *Demonstrated by:* DMN `itemDefinition` + input/output
   values. *Maps onto:* `ParametersConfig.dtypes` becoming authoritative for outputs, and a new
   optional `domain` on the table's inputs. *Effort:* **S**/**M.** **Must-have** for the output
   half.
9. **Effective dating of rows and trees.** *Gap:* nothing in `tables/schema.py` or
   `trees/schema.py` carries validity dates, while `example_projects/04-campaign-targeting-trees.md`
   §3 describes "approved, expiring overlays that change a published tree's answer without
   editing the tree" on a different clock from the artefact. *Demonstrated by:* Higson, InRule,
   Sparkling Logic *(landscape pass)*. *Maps onto:* `ParametersConfig` valid-from/valid-to
   columns filtered at `shared`-build time (so it stays a *values* change, doc 08 §2).
   *Effort:* **M.** Nice-to-have, but it is a real requirement in the repo's own example set.
10. **TCK-style labelled conformance fixtures.** *Gap:* `testing/` proves modes agree but does
    not pin *semantics* case by case, and doc 03 §1.2 already says the corpus must include
    boundary values found by search, not sampling. *Demonstrated by:* the TCK's
    `<labels>Hit Policy: COLLECT</labels>` convention and jDMN's generated `unittest` classes.
    *Maps onto:* `testing/` + `tests/`. *Effort:* **S.** Nice-to-have.
11. **Make "irrelevant" explicit rather than encoding it as `None`.** *Gap:* a null bound/value
    silently disables a condition for that row (`tables/schema.py:393-394`, `:520-521`), which
    is unreviewable and collides with DMN's `-` also meaning "not null". *Demonstrated by:* DMN
    grammar rule 12.c. *Maps onto:* a sentinel (e.g. `"-"`) accepted in the column plus a
    rendered "any" in `explain()`. *Effort:* **S.** Nice-to-have.
12. **A first-class one-sided comparison expression.** *Gap:* `>= 5` requires `BetweenExpression`
    plus the neighbour-fill and open-edge rules (`tables/schema.py:322-375`), which is more
    machinery than the test needs and rejects legitimate non-contiguous threshold tables unless
    `allow_gaps=True`. *Demonstrated by:* DMN grammar rule 5/13. *Maps onto:* one new class in
    `tables/schema.py` (the file's own docstring says adding a kind is one class, `:22-26`).
    *Effort:* **S.** Nice-to-have.
13. **Carry table/tree metadata into the decision record.** *Gap:* doc 04 §5.2's audit-record
    table has no row for "which interior document, of which kind, with which policy, at which
    version". *Demonstrated by:* jDMN's `DRG_ELEMENT_METADATA` (`Strategy.py:47-55`) carries
    namespace, name, element kind, expression kind, hit policy and input count as generated data.
    *Maps onto:* `tables/codegen.py` / `trees/codegen.py` emitting a metadata constant, consumed
    by doc 04 §6.5's renderers. *Effort:* **S.** **Must-have** for the governance story.

---

## B. Actionable recommendations

Each item is a concrete change, phrased as "do X in Y so that Z", with the file/doc it touches,
effort, and the engine/spec feature that motivates it.

1. **Add `hit_policy: Literal["FIRST", "UNIQUE"] = "FIRST"` to `DecisionTable`** in
   `src/decider2/tables/schema.py` (beside `outputs`/`default`, `:569-570`) **so that** the
   current semantics stop being an undocumented default and `UNIQUE` becomes expressible.
   Extend `validate_config` (`:572-585`) to reject `UNIQUE` until item 3 lands. *Effort:* S.
   *Motivated by:* DMN §8.2.11 [8.2.10] (`dmn15.txt:3527`), jDMN
   `RuleOutputList.java:43-48`, Camunda's supported set.
2. **State the divergence in `tables/codegen.py`'s module docstring** (`:38-42` already
   describes first-match-wins) **so that** a reader knows DMN's default is `UNIQUE` with
   unordered matching (`dmn15.txt:5883`) and decider2's is `FIRST` with early return — i.e. that
   row order is load-bearing here and is not in DMN. *Effort:* S. *Motivated by:* DMN §8.2.11.
3. **Write `src/decider2/tables/analysis.py` with three passes over a `DecisionTable`**:
   `missing_intervals(table)` (per numeric condition column), `overlapping_rows(table)` (across
   all conditions jointly), `missing_rows(table)` (uncovered cells of the condition
   cross-product) **so that** a `UNIQUE` table can be validated and a `FIRST` table can report
   unreachable rows. Port the recursive column-sweep from
   `jdmn/dmn-core/src/main/java/com/gs/dmn/validation/Sweep{RuleOverlap,MissingInterval,MissingRule}Validator.java`
   and its `validation/table/{Bound,Interval,NumericInterval,EnumerationInterval}.java` types;
   the algorithms are cited in-source to "Semantics and Analysis of DMN Decision Tables"
   (`SweepRuleOverlapValidator.java:116-125`, paper vendored at `jdmn/docs/articles/`).
   Reuse `BetweenExpression.resolved_bounds()` (`tables/schema.py:322-340`) for bound extraction.
   *Effort:* M. *Motivated by:* jDMN sweep validators; Drools `ANALYZE_DECISION_TABLE`.
4. **Add an optional `id: str | None` and `annotation: str | None` per row** — as two reserved
   column names in `ParametersConfig` (`tables/schema.py:152-196`), excluded from
   condition/output resolution — **so that** `TableModule.decode()` (`tables/build.py:62-90`)
   can map a matched row index to a stable id and a human sentence instead of an integer.
   Add them to `TableModule.explain()` (`:92-107`). *Effort:* S. *Motivated by:* DMN rule
   annotations (`dmn15.txt:3520-3523`), jDMN `Rule(index, annotationText)`
   (`Strategy.py:114`), pyDMNrules `'Executed Rule': (Decision, Table, RuleNumber)`.
5. **Add `annotation: str | None` to `LeafNode`** in `src/decider2/trees/schema.py:653-672`
   and carry it into the tree's emitted metadata **so that** a campaign tree's leaf carries the
   business meaning of the leaf, not just a `result_idx`. *Effort:* S. *Motivated by:* the same
   DMN annotation clause; `example_projects/04-campaign-targeting-trees.md` §1's path-capture
   requirement.
6. **Emit a per-interior metadata constant from `tables/codegen.py` and `trees/codegen.py`** —
   name, kind (`decision_table` / `v3-tree`), hit policy, condition count, row count, source
   hash — as a module-level dict in the generated file, mirroring jDMN's
   `DRG_ELEMENT_METADATA` (`Strategy.py:47-55`) **so that** doc 04 §5.2's audit record has a
   row for "which interior ran" and doc 04 §6.5's renderers have something to render.
   *Effort:* S. *Motivated by:* jDMN `DRGElement` metadata; Drools DMN listener events.
7. **Make `ParametersConfig.dtypes` authoritative for output typing** in
   `tables/codegen.py:239-246`: when a declared dtype exists use it; when it does not, *warn*
   rather than infer `int64` from all-integer data **so that** a float quantity that happens to
   hold whole numbers today does not silently become an integer column tomorrow. *Effort:* S.
   *Motivated by:* DMN output clause `typeRef` (§8.2.9 body).
8. **Add an optional `domain` per condition variable** to `DecisionTable` — a min/max for
   numeric, an allowed-value list for string — **so that** `analysis.py`'s gap pass has a
   universe to be complete against, and so a config UI can render a bounded input the way
   `param()`'s pydantic `Field` already lets it render a bounded knob (`params.py:250`).
   *Effort:* M. *Motivated by:* DMN input values (`dmn15.txt:3366-3368`) and `itemDefinition`
   `allowedValues` (§7.3.3).
9. **Add a `COLLECT_SUM` scan mode to `tables/codegen.py`**: when `hit_policy == "COLLECT"` and
   `aggregation == "SUM"` and every output is numeric, emit a scan that accumulates
   `shared.<out>[r]` over all matching rows instead of `return r` (`:217-218`), and emit the
   matched-row *count* as a companion value **so that** an additive scorecard is one table
   rather than N tables plus an adder. *Effort:* M. *Motivated by:* DMN `C+`
   (`dmn15.txt:5866-5867`), jDMN `PythonFactory.java:201-205`, Camunda SUM aggregator,
   PMML `Scorecard`.
10. **Add `NotExpression` to `tables/schema.py`'s `Expression` union** (`:535-545`) and insert a
    negation-normal-form pass ahead of `AndExpression.to_dnf()` (`:257-278`) **so that** the
    table vocabulary matches the tree's, which already has `TLogicOp.NOT`
    (`trees/schema.py:160`, `:873-876`). Update the "no NOT ⇒ monotone" comment at `:259-262`.
    *Effort:* M. *Motivated by:* DMN grammar rule 12.b (`dmn15.txt:3921`).
11. **Add `LessThanExpression` / `GreaterThanExpression` (with inclusive flags) to
    `tables/schema.py`** following the file's own "one class here" recipe (`:22-26`) **so that**
    a one-sided threshold does not have to be modelled as a `between` with neighbour-fill and
    the open-edge restriction at `:367-375`. *Effort:* S. *Motivated by:* DMN grammar rules 5
    and 13.
12. **Accept an explicit `"-"` sentinel wherever a `None` currently means "irrelevant"** in
    `tables/schema.py` (`:393-394`, `:520-521`, `:530`) and render it as `any` in
    `TableModule.explain()` **so that** a blank cell and a deliberately-irrelevant cell are
    distinguishable to a reviewer. *Effort:* S. *Motivated by:* DMN grammar rule 12.c,
    `dmn15.txt:3307`.
13. **Write `src/decider2/interop/dmn_import.py` mapping a DMN 1.3/1.5 `<decisionTable>` to a
    `DecisionTable`**: interval → `BetweenExpression`, comma-list of literals →
    `InExpression`, single literal → `EqExpression`, boolean → `IsTrueExpression`, multiple
    input clauses → `AndExpression`, `-` → the item-12 sentinel, `hitPolicy="FIRST"` → accept,
    anything else → a refusal naming the unsupported policy. Rewrite `[a..b]` into
    `[a..b+ε)` only for integer-typed inputs and refuse otherwise. **So that** existing DMN
    assets and third-party editors become inputs. *Effort:* M. *Motivated by:* DMN CL2
    (`dmn15.txt:412-415`); jDMN proves the mapping is mechanical.
14. **Write `src/decider2/interop/dmn_export.py` emitting a DMN 1.3 `<decisionTable
    hitPolicy="FIRST">` from a `DecisionTable`** (one input clause per condition, one rule per
    row, `outputLabel` per output, row annotation → `<annotationEntry>`) **so that** `dmn-js`
    or the kie-tools DMN editor can be used as decider2's table editor and reviewer view,
    instead of building a third bespoke artefact after doc 04 §6's two failures. *Effort:* S.
    *Motivated by:* `dmn-js` / kie-tools being free authoring-only DMN editors *(landscape
    pass)*.
15. **Write `src/decider2/interop/pmml_scorecard.py` producing a `DecisionTable` per PMML
    `Characteristic`** (attribute predicates → conditions, `partialScore` → a numeric output,
    `reasonCode` → the item-4 row annotation, `baselineScore` → the `default`) **so that**
    analytics-team scorecards land as native tables and their reason codes survive — the
    taxonomy doc 04 §4.1 says was previously dropped. *Effort:* M. *Motivated by:* PMML
    `Scorecard`, normatively referenced by DMN (`dmn15.txt:517-518`).
16. **Write `src/decider2/interop/pmml_tree.py` producing a `Tree`** — `SimplePredicate` →
    `UnaryNode` with the matching comparison op (`trees/schema.py:309-331`),
    `SimpleSetPredicate` → `CasesIsIn` (`:809-835`), `CompoundPredicate` → `CompositeNode`
    (`:837-879`), `Node@score` → a `TreeOutput` row — **so that** a tree trained in a modelling
    tool reaches production without hand transcription, which
    `example_projects/04-campaign-targeting-trees.md` §3 describes as the actual workflow.
    *Effort:* M. *Motivated by:* PMML `TreeModel`.
17. **Add `valid_from` / `valid_to` as reserved `ParametersConfig` columns filtered when
    `TableModule.shared` is built** (`tables/build.py:46-54`) **so that** effective dating stays
    a *values* change with no recompile (doc 08 §2's "free" class) rather than becoming a shape
    change. *Effort:* M. *Motivated by:* Higson / InRule effective dating *(landscape pass)*;
    `example_projects/04` §3's expiring overlays.
18. **Add a labelled-fixture format to `src/decider2/testing/`** — a list of
    `{labels: [...], inputs: {...}, expected: {...}}` records, run by a generic parametrised
    test — and seed it with one fixture per table semantic currently untested (no-match default
    ✓ exists at `tests/test_tables.py:131`; missing: `upper_inclusive` boundary equality, an
    empty `in` set, a `None` bound, a string literal absent from every row). **So that** the
    suite pins *semantics* and each case says what it exists for. *Effort:* S. *Motivated by:*
    the DMN TCK's `<testCases><labels>` convention
    (`jdmn/dmn-test-cases/.../0115-sum-collect-hitpolicy-test-01.xml:4-11`).
19. **Add a `decider2 check --tables` subcommand** wired to item 3, with per-check granularity
    (`--gaps`, `--overlaps`, `--missing-rows`, `--all`, `--off`) **so that** table analysis is a
    CI gate rather than a library call. *Effort:* S once item 3 exists. *Motivated by:* Drools'
    `<validateDMN>` per-check configuration.
20. **Close the `str`-input hole in the equivalence ladder** at
    `src/decider2/testing/equivalence.py:194-197`, which currently skips the whole `score()`
    rung for any pipeline with a `str` input, **so that** string decision tables — normal in
    credit, and already individually tested at `tests/test_tables.py:242` — are covered by the
    generic assertion. *Effort:* M (needs `runtime.invoke.score` to grow the dictionary-code
    path doc 05 §1.5 describes). *Motivated by:* nothing external — it is the one place where
    decider2's own strongest property (batch ≡ realtime, which no surveyed engine has) is not
    fully asserted.
21. **Add a short `docs/09-standards-and-interop.md`** stating: DMN is the interchange target
    for decision tables at CL2, PMML for scorecards and trees, ONNX explicitly out of scope and
    why (kernel split, EXPERIMENTS §B), and the mapping tables from §1 above. **So that** the
    next reader does not have to rediscover that the vocabulary came from decider 1 rather than
    from a standard. *Effort:* S. *Motivated by:* the zero-hit grep for DMN/PMML/ONNX across
    all of `decider2/docs` and `decider2/example_projects`.

---

## 12. Genuine alternatives — could an existing engine replace part of `decider2`?

**Could jDMN-style transpilation from DMN replace decider2's authoring layer?**
Partly, and only for tables. In favour: DMN is a real standard with free editors, jDMN proves
DMN → Python is mechanical, and `jdmn-python-runtime` is Apache-2.0 and installable today.
Against, decisively:

- **The generated code cannot be compiled.** jDMN's Python emits classes extending
  `DefaultDMNBaseDecision`, `typing.Optional` everywhere, `decimal.Decimal` numbers, and
  per-rule listener calls (`Strategy.py:46-137`). numba cannot compile any of it, and
  `Decimal` cannot even cross decider2's boundary — "it raises a Rust panic" (doc 03 §1.2 citing
  doc 05 §1.5). A DMN-authored pipeline would run interpreted at Python-object speed, and
  EXPERIMENTS §B measured that per-row Python inside a driver is 23.5× and `objmode` 77×.
- **No short-circuit.** jDMN evaluates every rule and then applies the policy
  (`Strategy.py:96-99`; `apply.ftl:101-112`) even for FIRST, which DMN explicitly permits
  halting on (`dmn15.txt:3536-3537`). decider2's `return r` on first match
  (`tables/codegen.py:217-218`) and its real branches in trees (doc 03 §8.2, "only the taken
  arm executes ... the entire source of the 7.8× advantage at depth 50") are the opposite
  design.
- **No batch.** Nothing in DMN or jDMN is columnar. decider2 runs 14.2 M clients
  (`example_projects/04` §1) through the same kernel as a 20 ms request, and *tests* that they
  agree (`testing/equivalence.py:218`). Reaching that from jDMN means writing the columnar
  engine anyway.
- **FEEL exceeds what decider2 can accept.** Contexts, lists, `for`/`some`/`every`, temporal
  arithmetic, external Java/PMML functions and generalized unary tests are all either
  uncompilable or explicitly banned from config (doc 03 §10). Importing DMN at CL2 is feasible
  (§9a); adopting DMN as *the* authoring language is not.
- **Reviewer readability is not obviously better.** Doc 04 §6.6 records that decider 1's
  declarative form failed worse than its Python: "~1000 lines of serialised AST with UUID node
  ids and `result_idx: -1` indirection to express roughly thirty rules". Raw DMN XML is the same
  genus (see `0115-sum-collect-hitpolicy.dmn`: 4 rules, ~45 lines of XML with 9 generated ids).
  The gain from DMN is the *editors*, not the format.

**Verdict:** adopt DMN as an **interchange format at CL2, in and out** (recommendations 13-14).
Do not adopt it as the authoring language or jDMN as the engine.

**Could Drools/Kogito replace decider2's engine?** No, on four independent counts.
(1) *Language*: JVM. decider2's steps are Python functions and its boundary is polars/Arrow;
adopting Drools means a JVM service and a serialisation boundary per record, against a 20 ms
budget where decider2 currently spends 0.07% of budget on all kernel dispatch
(`02-architecture.md:132-133`). (2) *Execution model*: PHREAK working memory with
`insertLogical()` truth maintenance and `salience` has, in GoRules' words, "no structural
equivalent in a stateless DAG" *(landscape pass)* — and stateless per-record scoring is the
whole requirement. (3) *Batch*: no columnar path; 14.2 M records means 14.2 M evaluations and
no batch ≡ realtime proof. (4) *No published latency numbers on any Drools page fetched*, so
the 20 ms budget cannot even be argued.

**What an existing engine *could* genuinely replace:**

- **The reviewer/editor surface.** `dmn-js` (bpmn.io licence) and the kie-tools DMN editor
  (Apache-2.0) are real, mature table editors *(landscape pass)*. Given recommendation 14,
  decider2 gets an authoring and review UI it has twice failed to build (doc 04 §6.1, §6.3b).
  This is the strongest "buy, don't build" conclusion in the survey.
- **The table-analysis algorithms.** jDMN's sweep validators are Apache-2.0 and cite their
  published algorithms; porting three classes is cheaper than inventing gap analysis
  (recommendation 3).
- **Nothing on the execution path.** No surveyed engine vectorises, none compiles to machine
  code, none proves batch ≡ realtime, and none publishes a comparable latency figure. On its own
  axes decider2 has no open-source alternative.

---

## 13. Unverified / unreachable

- **jDMN and Drools latency / compile-time numbers.** None published. jDMN has a
  `dmn-performance` module with `run-performance.bat` but no committed results. Any
  performance comparison with decider2's figures would be invented.
- **Drools DMN listener event class names and payloads.** The docs page fetched gives only the
  system property `org.kie.dmn.runtime.listeners.$LISTENER_NAME`; the `org.kie.dmn.api.core.event`
  interfaces were not read.
- **Drools DMN codegen (executable model for DMN specifically).** The fetched page says nothing;
  §5's "interpreted per request" is an inference from the absence of a codegen statement, not a
  quoted fact.
- **Camunda 7 DMN history/audit class names** (e.g. a decision-table evaluation event type).
  The dmn-engine page fetched is a navigation shell for those subsections.
- **pyDMNrules licence and DMN conformance level.** The README fetch returned neither; the
  GPL-3.0 and 1.4.5 facts come from the landscape pass, not re-verified here.
- **DMN TCK per-vendor result table.** Not re-fetched in this pass; all TCK pass counts are
  from `part_oss_notes.md` (2026-09-20) via `dmn-tck.github.io/tck`. The *format* of a test case
  is verified directly from the fixtures vendored in jDMN.
- **Whether `jdmn-python-runtime` is numba-hostile in practice.** Argued from the generated
  source (classes, `Optional`, `Decimal`), not measured.
- **DMN 1.5 §8.2 numbering.** The PDF's TOC and body disagree by one after §8.2.5 (see the
  header note). Section numbers here follow the body headings.
- **PMML 4.2 clause-level citations.** PMML itself was not fetched in this pass; the
  `Scorecard`/`TreeModel`/`RuleSetModel` element and attribute names in §9c, §A and
  recommendations 15-16 are from prior knowledge and must be checked against
  `dmg.org/pmml/v4-2-1/` before implementation.
- **`both_inclusive` interval import.** Recommendation 13's `[a..b]` → `[a..b+ε)` rewrite is
  only sound for integer-typed inputs; the float case has no safe rewrite and must be refused.
  Not prototyped.
- **Anything marked *(landscape pass)*** — licences, versions, dormancy dates and vendor
  benchmark claims for the long-tail engines — is from the 2026-09-20 fetch recorded in
  `scratchpad/part_oss_notes.md`, not re-verified today.

---

## 14. Sources

Local primary artefacts (in
`/tmp/claude-1000/.../1835b6ef-.../scratchpad/`):
`dmn15.pdf` / `dmn15.txt` (OMG DMN 1.5, `formal/24-01-01`, from `https://www.omg.org/spec/DMN/1.5/PDF`);
`jdmn/` (`https://github.com/goldmansachs/jdmn`, Apache-2.0, shallow clone),
in particular `dmn-core/src/main/java/com/gs/dmn/validation/Sweep*.java` and
`validation/table/`, `dmn-core/src/main/java/com/gs/dmn/transformation/native_/PythonFactory.java`,
`dmn-core/src/main/resources/templates/dmn/python/`,
`dmn-runtime/src/main/java/com/gs/dmn/runtime/RuleOutputList.java`,
`dmn-runtime-api/src/main/java/com/gs/dmn/runtime/annotation/HitPolicy.java`,
`dmn-test-cases/standard/tck/1.1/cl3/0004-lending/translator/expected/python/`,
`dmn-test-cases/standard/tck/1.5/cl2/0115-sum-collect-hitpolicy/`,
`docs/articles/Semantics*.pdf`; `part_oss_notes.md` (landscape fetch pass).

Web: `https://docs.drools.org/latest/drools-docs/drools/DMN/index.html`;
`https://docs.camunda.org/manual/latest/user-guide/dmn-engine/`;
`https://docs.camunda.org/manual/latest/reference/dmn/decision-table/hit-policy/`;
`https://github.com/camunda/feel-scala` (README);
`https://github.com/russellmcdonell/pyDMNrules` (README);
`https://github.com/dmn-tck/tck` (README);
`https://pypi.org/pypi/jdmn-python-runtime/json`.

`decider2`: `docs/README.md`, `docs/00-BUILD.md`, `docs/02-architecture.md`,
`docs/03-authoring-api.md`, `docs/04-observability-and-governance.md`,
`docs/05-boundary-and-compilation.md`, `docs/08-configuration-and-lifecycle.md`,
`docs/research/decision-engine-landscape.md`;
`src/decider2/{tables,trees}/{schema,codegen,build}.py`,
`src/decider2/boundary/{nulls,dtypes}.py`, `src/decider2/compile/cache.py`,
`src/decider2/graph/{interface,pipeline,module}.py`, `src/decider2/params.py`,
`src/decider2/testing/equivalence.py`, `tests/test_tables*.py`;
`example_projects/04-campaign-targeting-trees.md`.
