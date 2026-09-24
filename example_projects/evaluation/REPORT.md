# Example-project experiment: final report

**What was run.** Twelve example specs (`specs/00`–`11`). For each one, one Haiku agent and one Sonnet agent implemented a vertical slice (`specs/SCOPE.md`) with `decider`, each consuming its own model's earlier projects (`specs/DEPS.md`). Every project was then scored by an Opus judge (`judgements/NN.md`, rubric in `specs/JUDGE.md`). A low-effort Haiku agent summarised every implementation from its code alone (`summaries/`). All framework complaints were triaged and reproduced (`framework-issues.md`, F1–F32). The timeline is in `notes/progress.md`, "Example-project experiment".

**Sources and conventions.** "J03" means judgement 03, "F2" means framework issue F2, and "log" means the progress log. `03h` is Haiku on project 03 and `03s` is Sonnet on project 03. Scores come from the YAML block at the top of each judgement. Line counts were measured for this report with `find … -name '*.py' | wc -l` and include tests.

---

## 1. Headline

- **Sonnet always delivered and Haiku never did, but nothing serves over HTTP.** All 12 Sonnet projects build, and all score their sample through the in-process handler route in SERVE.md. None of the 12 Haiku projects builds. Even so, no Sonnet project serves unchanged over its documented HTTP route. All 12 monkeypatch the private `decider.serving.handler._warm` (F1), and all 12 run in interpreted mode (F5). Raw JSON dates are never turned into `date`s, so the `curl` route failed in every one of the 8 projects where a judge tried it (J00, J01, J03, J04, J06, J08, J10, J11). No agent tested that route.
- **The two models never reached similar solutions.** Similarity is 2/5 in all 12 projects, and Sonnet was closer to the spec in 12 of 12. The low score does not mean two good designs disagreed. Haiku mostly did not use the framework: it wrote plain Python behind one to three steps. The one design choice both models shared is telling: both kept the search, the solve and the portfolio allocation in plain Python, outside decider constructs (J03, J06, J07).
- **Haiku's failures come mostly from the agent, but two framework bugs hid them.** F2 (the wrong project's `pipeline.py` gets built) was the first failure in 5 Haiku builds (02, 04, 05, 06, 07). F4 was the first failure in one more (08). Behind those, every Haiku project had defects of its own:
  - request fields declared as `param()`;
  - imports that don't exist;
  - CLI flags it invented;
  - one step returning a `dict`.

  Fixing the framework alone would not have made any of them serve correctly. Adding the "build is mandatory" instruction after wave 1 changed nothing: 0 of 5 Haiku projects built before the change and 0 of 7 after.
- **Self-reports could not be trusted from Haiku, and needed checking from Sonnet.** Haiku's NOTES or SERVE.md made claims the judges found false in all 12 projects: build success, invented outputs and test counts, a "67% of decision points reused" figure (J11). Sonnet's build and scoring claims held up. Its overstatements were smaller: "20 of 22" capabilities reused was really 13 (J11), and a "reuses `diff_cards`" comment has no call behind it (J09). Judges also found real logic bugs in Sonnet's code that its weak tests did not catch:
  - the solve disagrees with exhaustive search in 65 of 300 cases (J03);
  - suspended accounts are still contactable (J08);
  - an inverted stale-review grade cap (J11);
  - the allocation logic is never exercised (J07).
- **Sonnet reused earlier projects for real; Haiku did not.** Sonnet reused real code in all 11 projects that had something to consume. It did this with `.relabel()`, whole pipelines as sub-dags, and nested `Engine`s, but also with `importlib` workarounds and private-API reaches. Haiku's reuse was effectively zero: 1 partial, 1 stubbed and 9 none, even though several of its NOTES claimed heavy reuse.
- **Some built-ins carried the work; others were never touched.** `DecisionTableConfig`, `ScorecardConfig`, `TreeConfig`, `frame_step` and `relabel` did the heavy lifting. `loop` was used once, `branch` never. Sessions, `assert_equivalent`, `step_map`/`to_ir` and programmatic `ConfigStore` were never used, even where the judge showed they would have fixed a bug (J09).
- **The code-only summaries barely told the models apart.** Mean summary accuracy was 3.3 for Haiku and 3.5 for Sonnet, while mean code quality was 2.0 against 3.75. Summarisers repeated false docstrings as fact. They could not reach Sonnet's logic that sits outside `pipeline.py`. Sonnet's code is 2.1× the size of Haiku's, with essay-length docstrings.
- **The top framework fixes are small and mostly already tested.** In order:
  - F1 (warm-up);
  - F2 (`code_path` order);
  - F3 (one change fixes seven nested/temporal-column bugs; tested, suite still green);
  - F6/F7 (empty lists and nulls);
  - F4 and F5 together, since F5 alone would expose F4 in every Sonnet project;
  - the silent bugs F8–F11;
  - JSON date coercion on the HTTP path, which is not in the issues file.

---

## 2. Scorecard

### 2.1 Per implementation (from each judgement's YAML)

| Project | Model | Build | Sample | Tests | Summary acc. | Reuse | Coverage | Quality |
|---|---|---|---|---|---|---|---|---|
| 00 core | H | fail | fail | 21/21 | 3 | n/a (wave 0) | 2 | 2 |
| | S | pass | pass | 79/79 | 4 | n/a (wave 0) | 4 | 4 |
| 01 fraud | H | fail | fail | 12/12 | 3 | none | 1 | 2 |
| | S | pass | pass | 30/30 | 4 | real | 4 | 4 |
| 02 afford. | H | fail | fail | 8/8 | 3 | partial | 2 | 2 |
| | S | pass | pass | 37/37 | 4 | real | 4 | 4 |
| 03 granting | H | fail | fail | 8/8 | 3 | stubbed | 1 | 2 |
| | S | pass | pass | 39/39 | 4 | real | 4 | 4 |
| 04 trees | H | fail | fail | 14/14 | 3 | none | 2 | 2 |
| | S | pass | pass¹ | 48/48 | 4 | real | 4 | 4 |
| 05 nested | H | fail | fail | 0/11² | 4 | none | 2 | 2 |
| | S | pass | pass | 31/31 | 3 | real | 4 | 4 |
| 06 consol. | H | fail | fail | 9/9 | 4 | none | 1 | 2 |
| | S | pass | fail¹ | 39/39 | 3 | real | 4 | 4 |
| 07 limits | H | fail | fail | 4/4 | 3 | none | 1 | 2 |
| | S | pass | pass | 33/33 | 3 | real | 4 | 4 |
| 08 collections | H | fail | fail | 13/13 | 4 | none | 2 | 2 |
| | S | pass | pass | 41/41 | 4 | real | 4 | 3 |
| 09 governance | H | fail | fail | 0/12² | 4 | none | 1 | 2 |
| | S | pass | pass | 40/40 | 3 | real | 4 | 3 |
| 10 retail e2e | H | fail | fail | 26/26 | 3 | none | 1 | 2 |
| | S | pass | pass | 45/45 | 3 | real | 4 | 4 |
| 11 business e2e | H | fail | fail | 11/11 | 3 | none | 1 | 2 |
| | S | pass | pass | 37/37 | 3 | real | 4 | 3 |

¹ The judges were inconsistent about "sample scores". J06 marked Sonnet **fail** because the raw-JSON/HTTP route fails, even though the SERVE.md Python snippet passes. J00, J01, J03, J04, J08, J10 and J11 found the same raw-JSON failure and still marked **pass**. Under one rule, Sonnet scores 12/12 on the in-process snippet, and 0 of the 8 tested projects pass on raw JSON.
² Collection errors under the PYTHONPATH that SERVE.md gives. 05h's `assessment.py` is shadowed by 02h's module of the same name; 09h's PYTHONPATH leaves out its own directory. With the path fixed, 05h passes 11/11 and 09h passes 11/12 (J05, J09).

### 2.2 Per-model totals

| | Haiku | Sonnet |
|---|---|---|
| `decider build` passes | 0/12 | 12/12 |
| Sample scores (YAML) | 0/12 | 11/12 (12/12 via the SERVE.md snippet) |
| Tests passing as shipped | 126/149. **No Haiku test exercises a decider pipeline**: every judgement says they call plain functions | 499/499, including in-process `decider build` tests in several projects |
| Summary accuracy (mean) | 3.33 | 3.50 |
| Reuse of earlier projects (11 consuming projects) | 1 partial, 1 stubbed, 9 none | 11 real |
| Spec-slice coverage (mean) | 1.42 | 4.00 |
| Code quality (mean) | 2.00 | 3.75 |
| Python lines, all 12 projects | 13 895 | 29 413 (2.1×) |

### 2.3 Similarity per project

| Project | Similarity | Closer | Main divergence (judge's words, condensed) |
|---|---|---|---|
| 00 | 2 | Sonnet | H: plain library of dataclass functions, tables as dicts and formulas. S: every capability as steps, 8 `DecisionTableConfig`s, the 63 360-row card as a document |
| 01 | 2 | Sonnet | Both generate exactly 521 live / 114 shadow rules. H: a Python evaluator that fires 0 rules on the spec's example. S: `TreeConfig` `mode="all"` |
| 02 | 2 | Sonnet | H: one 130-line function in one step. S: an evidence/capacity `dag` reusing 00 units via relabel |
| 03 | 2 | Sonnet | Both wrote the waterfall and the solve in plain Python, and both reproduce Client W's cap chain. H's solve can't reach the answer |
| 04 | 2 | Sonnet | H: its own string-parsing tree engine. S: `TreeConfig` plus a parallel Python walker for the path |
| 05 | 2 | Sonnet | Both classify the spec's adverse-event examples correctly. H: loops inside one step. S: long-form flattening plus a nested `Engine` |
| 06 | 2 | Sonnet | Same stage order. H always selects the empty settlement set |
| 07 | 2 | Sonnet | Both conclude that allocation is population-level. Only S built it. They share one bug: the 0% band can't be reached |
| 08 | 2 | Sonnet | H: one monolithic step with hash-based ids. S: six units plus population ranking |
| 09 | 2 | Sonnet | H works on hand-built dicts. S replays real decisions from 01, 03 and 05 |
| 10 | 2 | Sonnet | H: one step mutating a 60-field state object. S: a ~34-member `dag` with `decider.loop` |
| 11 | 2 | Sonnet | H: three dict steps with 05 stubbed. S: 05's whole pipeline as a sub-dag |

---

## 3. The four questions

### 3.1 Do the two models reach similar solutions?

No. Every judge gave 2/5, and the reason was the same each time: Haiku wrote a plain-Python library and wrapped it in a nominal `flow`, while Sonnet decomposed the slice into decider steps and config documents. Where both could run a spec's worked examples, the results split in the same way:
- **Both correct:** 03 (Client W's cap chain, J03), 05 (the three adverse-event classifications, J05) and 07 (five of seven worked examples, J07).
- **Haiku wrong where Sonnet was right:**
  - 01h fires 0 rules on the spec's mule payment.
  - 02h returns `indeterminate` for every mode-1 request and passes an R99 999 instalment.
  - 03h's solve returns `None` on the band-edge fixture.
  - 06h always picks the empty set.
  - 10h gives *lower* income a *better* grade (J01, J02, J03, J06, J10).

The one real design convergence is negative evidence about the framework's reach. Both models kept iterative search and population logic outside decider constructs:
- the 03 solve and waterfall (J03; 03s prototyped `decider.loop` and chose Python, per the log);
- the 06 scenario search (J06);
- the 07 portfolio allocation (J07; Haiku says so in its NOTES, and Sonnet built it as plain polars).

Only 10s used `loop`, for the L1 loop across phases.

The experiment therefore says little about whether *two competent decider users* converge. That would need two Sonnet-class runs per spec (§7.4).

### 3.2 Working solutions that serve with no changes?

**Haiku: 0/12.** The triage re-ran every Haiku build (framework-issues.md, "What actually stopped the Haiku builds"). The progress log had called this a recurring "params-document path error" in 6 of 12 projects. The triage showed that diagnosis was mostly wrong:

| First blocker | Projects | Class | Behind it (per judgement) |
|---|---|---|---|
| F2: SERVE.md puts 00 ahead of the project on `PYTHONPATH`; `_warm` doesn't move `code_path` to the front, so **00's `pipeline.py` is built** against the project's params | 02h, 04h, 05h, 06h, 07h | framework bug | 02h, 06h: F1 warm-up next. 05h, 07h: `from decider import RequestHandler` (F32, agent). 04h: F5 `str` literal. All five also declare request fields as `param()` |
| F4: a step annotated `-> dict` fails in fused mode | 08h, and 07h/11h once other errors are removed | framework bug triggered by the agent's design | 08h blamed `ge=/le=` constraints and said removing them fixed it. It did not (J08). All 31 inputs are params, so a request can't change the answer |
| Params document doesn't match the pipeline | 00h (config belongs to `pipeline_simple.py`), 09h (flat document) | agent; F12 message unhelpful | 09h's replay calls a stub that always returns "approved" |
| Import or CLI errors | 01h (`missing_as(None)`, every `@step` commented out), 03h, 10h, 11h (made-up `--code-path` flag, wrong `RequestHandler` import) | agent | 03h claimed "decider CLI missing" (F23: it works through `uv run --project`) |

Every Haiku project had at least one agent-level defect behind any framework blocker (§6.1). F1, F2 and F4 are real bugs, and they cost Haiku its diagnosis: three projects blamed "the framework" for what was F2 plus their own errors. But none of them would have served correctly with those bugs fixed.

**Sonnet: 12/12 on the documented in-process route, and not unchanged over HTTP.** Three things stand between these projects and a clean deployment:
1. **F1.** Every Sonnet project monkeypatches the private `decider.serving.handler._warm`. The warm-up feeds `1.0` to `date`, `list` and `dict` inputs, and almost every spec has a `decision_date: date`. J05 rebuilt 05s without the patch and the build failed. The dependency on a private name means any decider change to `_warm` breaks all 12.
2. **F5.** All 12 run `DECIDER_API__MODE=interpreted`. One inherited step (00s's `norm_table_version`, which compares two `str` columns) is refused by compiled modes and drags every downstream project with it (log, 07s entry). The throughput cost:
   - 07s reports 294 rows/s, which is 3.9 h for the 4.1 M book against a 3 h window (not verified by J07);
   - J08 measured 28 ms per row, about 18 h for 2.3 M rows against a 90-minute limit;
   - J02 measured 400 capacity calls at 3.6 s against the spec's 900 ms.

   F4 sits behind F5: once F5 is fixed, any project leaving interpreted mode meets the float64 fallback bug.
3. **HTTP date coercion.** A raw JSON body reaches the steps with dates as strings. The resulting errors vary: `TypeError: '<=' not supported between 'datetime.date' and 'str'` (J00, J01, J10), `'str' object has no attribute 'year'` (J06, J11), and a `str + timedelta` error (J04, J08). SERVE.md's own snippets convert dates by hand, so the agents never saw it. **This issue has no number in `framework-issues.md`.** The judges classed it as a framework gap (`decider/serving/parse.py` does no type coercion, J03) combined with an agent miss (no `input_fn` override).

Smaller serving findings:
- `score()` output is not JSON-serialisable when it contains dates (J09).
- 11s's `covenant_instance_id` is a process-global counter, so the same record scored three times gives 3, 4, 5 (J11).
- `decider serve` itself was never run by a judge: uvicorn is not installed and port 8080 was taken (J01, J04, J06, J08, J10).

### 3.3 Maintainable, understandable code with little context?

Summary accuracy was 3.33 for Haiku and 3.50 for Sonnet. Those numbers are close, but the errors behind them differ.

**What the summarisers got wrong, and what it says about the code:**
- **They repeated false docstrings as fact.** Examples:
  - 10h's "620+ rules" and "14 assertions";
  - 11h's "transparent pass-through to 05" over a stub;
  - 03h's "52 rules" (there are 12);
  - 04s's "absolute suppressions exclude from tree", where no branch exists (J10, J11, J03, J04).

  In code read with little context, a wrong docstring is worse than none. Haiku's docstrings described intent the code does not have (J00: `affordability.py` claims four modes, and `mode` changes nothing).
- **Nobody noticed a failed build or dead wiring.** No Haiku summary flagged that the pipeline cannot build. None flagged request fields being overridden by params (J05, J06, J08, J11). None flagged imported-but-never-called modules (J00). A reader of the code cannot see these failures. Only running it shows them.
- **Sonnet's depth is not reachable from `pipeline.py`.** The Sonnet summaries that scored 3 all covered the entry point and missed the substance:
  - 07's allocation and simulation, "the point of the project" (J07);
  - 09's explain, diff, swap-set, overlay and dead-logic modules (J09);
  - 10's registries, solve and loop, which the summariser admits it did not read (J10);
  - 11's L1, history, bi-temporal queries and reuse inventory, which SCOPE calls "the actual deliverable" (J11).

  In each case the logic lives in a library that the served pipeline does not compose, or in modules too long to skim.
- **Invented numbers.** 00s's "360-cell grid" (it has 63 360 rows), 03s's "41 inversions" (the card has none), 06's "~400 scenarios" (the budget, not the count).

**The judges' own reading.** Haiku code is "short and readable, but misleading" in almost every judgement. Its common faults:
- dead imports;
- `date.today()` in decisions (J00, J02, J05, J06, J08, J09, J10, J11);
- process-salted `hash()` used for ids and cell ids (J04, J07, J08);
- policy numbers inline.

Sonnet code is consistently "one module per spec section, docstrings citing § numbers", with three recurring costs:
- very long docstrings (J01, J04, J09, J10, J11);
- column-renaming bookkeeping (J00);
- tables and thresholds kept in Python rather than `configs/`, against the BRIEF's style rule. Examples: 01s governance metadata, 03s cap register, 05s tables, 06s (no `params.json` at all), 08s matrix and scorecard, 10s card (J01, J03, J05, J06, J08, J10).

That last cost matters most for the specs' "analyst changes it on a Tuesday" requirement. 00s and 07s kept their large tables as config documents; most other Sonnet projects did not.

### 3.4 Reuse, and built-ins versus writing from scratch

**Reuse of earlier projects (verified by judges against call sites).**
- **Sonnet reused real code in 11/11 projects.** The patterns that worked:
  - `.relabel()` to run one unit twice: 00's income and deductions per applicant (02s); four overlay stacks from one `apply_stack_step` (03s); as-known versus as-at-now (08s).
  - Whole upstream pipelines as a sub-dag or a nested `Engine`: 02 inside 03s per row; 02 for 05s's sole proprietors; 02 and 03 inside 06s; 05 as 11s's EP-1 sub-dag.
  - Unmodified calls into 00's registers (`AdjustmentRegister`, `ReasonCodeRegistry`, `EffectiveDatedSet`).
- **Sonnet's reuse had costs.** Because every project names its entry module `pipeline.py`, 03s, 05s, 06s, 07s, 08s, 09s and 11s loaded siblings through `importlib` by file path (J03, J05–J09, J11). Private reaches were common: `obligations._process` (02s, 06s, 11s), `capacity._buffer_pct` (06s), `register._all` (08s, 09s), 05's `_BUSINESS_OVERLAYS` (09s). This happened because 00 and 02 published tables only as step configs and had no public enumerators, which is a gap in the consumed code rather than in decider.
- **Where Sonnet declined to reuse, it said why.** 06s did not reuse 03's `solve_term`, whose bounds are module constants. 10s did not use 00's card and scorecard, because the spec requires "10's own tables".
- **Sonnet's claimed reuse versus verified reuse:**
  - 11s's "20 of 22 capabilities transitively" is really 13 of 22.
  - Its `PASSTHROUGH_RELABEL_COUNT = 10` is a literal; the package has no `.relabel(` call.
  - 06 is loaded and never called (J11).
  - 09s's `diff.py` says it reuses `diff_cards` and does not (J09).
- **Haiku effectively did not reuse.** 02h made real calls to three of 00h's units. 03h stubbed everything. The other nine imported nothing from earlier projects, even with them on `PYTHONPATH` (J04, J05, J06, J07, J08, J09, J10, J11). Several NOTES claimed otherwise:
  - 03h: "Reused core library heavily (14 capabilities)";
  - 10h: "All core.* capabilities available … imported";
  - 11h: "1 280 of 1 900 (67%)", copied from spec §4.9, when the real count of decision points consumed is 0.

  Haiku also inherited a thin 00. Its tax is a flat 20% and its rate card is a formula (J00, J02), so even real reuse would have carried little.

**Built-ins versus from scratch.**

| Built-in | Used by | Not used where it fits (judge) |
|---|---|---|
| `DecisionTableConfig` | 00s (8 tables), 02s, 03s, 05s, 06s, 07s (1 152 cells), 08s (5 376 cells), 10s | 01s segments/actions and 07s utilisation bands are Python dicts or loops. 06s uses it only as a row container |
| `ScorecardConfig` | 00s, 05s, 07s, 08s, 10s (03s via 00) | every Haiku scorecard is an if-chain |
| `TreeConfig` | 01s (635 flat rules), 04s, 09s (load for diff) | 04h wrote its own tree engine after being pointed at `TreeConfig` and `path_output` |
| `frame_step`, nested `Engine` | most Sonnet projects | n/a |
| `loop` | 10s only | 03s prototyped it and rejected it |
| `branch` | **nobody** (it appears only in comments in 07s and 10s) | 04 absolute suppression "no evaluation should occur" (J04); 00 eligibility short-circuit (J00) |
| sessions (`Executable.session`) | **nobody** | 09 "first divergence" in execution order. 09s picks it alphabetically, which is a bug (J09) |
| `decider.testing.assert_equivalent` | **nobody** (09h names it as future work) | 09 certification and mode equivalence (J09) |
| `step_map` / `to_ir` | **nobody** | 09 dead-logic universes, hard-coded per flow in 09s (J09) |
| `ConfigStore` (programmatic) | **nobody** | 09 replay reads the mutable `configs/<v>/`. An in-place edit silently changes replayed logic (J09) |

Haiku's decider use was `flow`, `param` and `missing_as`, and in 8 projects `param` was misused for request fields (01h, 02h, 05h–08h, 10h, 11h). Stdlib use was sensible where it appeared: `decimal.ROUND_HALF_UP` and `uuid` (00h), `statistics` (00s), `hashlib` (04h), plain polars for allocation (07s).

---

## 4. What worked

**Framework features that held up at real size**
- **`TreeConfig` `prioritized_flat_rule` with `mode="all"` as a flat rule engine.** 01s ran 635 rules as three tree documents (live, shadow, overlay-base). Shadow isolation is structural, and a threshold overlay is a `ComputedFeature` (`amount - 8000*multiplier`). J01 calls it "the most useful framework finding in this project".
- **`TreeConfig` for analyst-authored trees with stable identity.** 04s compiled content-hashed node keys into `TreeConfig`, emitted `leaf_key` through `path_output`, and ran one tree artefact twice with `.named()`/`.relabel()` to get the adjusted and unadjusted leaves. Overlay thresholds are `{"param": …}`, so node keys can't change under an overlay (J04).
- **`DecisionTableConfig` for large grids, with cell ids in the evidence.** Examples: the 63 360-row rate card as a config document with a diff artefact (00s); the 1 152-cell limit matrix loaded from `configs/0.1.0/matrix.json` (07s); the 5 376-cell treatment matrix (08s). 00s's effective-dated resolver feeds its version id into the table match, and the judge confirmed that moving `decision_date` switches the cell (J00).
- **`ScorecardConfig`** with parameterised bins and per-characteristic contributions (J00, J05, J07, J08).
- **Relabel-based reuse.** 00 §10 item 4 ("same capability twice with different settings") was proven with buffers 0.12 and 0.18 (J00). The same mechanism powers 02s, 03s, 07s and 08s (§3.4).
- **Composition across projects.** A whole upstream pipeline works as a sub-dag (11s over 05) or as a nested `Engine` inside a `frame_step`, per row or per entity (03s, 05s). 05s's nested `Engine` works in both `score` and `run`, including with zero entities (F31).
- **`decider.loop`** "worked exactly as documented" for 10s's cross-phase L1 loop (log).
- **Replay.** 09s re-scores stored requests from 01, 03 and 05 with the recorded params. All 26 decisions reproduced, and a +1000 tamper was detected at the right field (J09). 05s's partial one-entity re-assessment was proven equal to a full run and then reused by 11s (J05, J11).
- **Some error messages were good.** The `flow` ordering `WiringError` (F25), the compiled-mode `str` message (F5: "the message is clear") and the input/output shadowing `EngineError` that 11s praised (J11).

**Agent behaviours that held up (Sonnet)**
- It shipped minimal repros for framework bugs. 01s's 8-line `PanicException` repro was reproduced by J01 and became F3e.
- It tested the build inside its own test suite (`test_decider_build_*` in 00s, 01s, 02s and 09s).
- It proved algorithms against brute force. 03s compared its solve to an exhaustive R100 grid (this still missed inversions; see §6.2).
- It declared when it had read framework internals (04s) and flagged spec contradictions that the judges confirmed (§5.2).

---

## 5. What didn't work

### 5.1 Framework problems (ranked by blast radius; F-numbers from `framework-issues.md`)

1. **F1: the warm-up can't synthesise `date`/`list`/`dict` inputs.** This blocked every realistic pipeline, forced all 12 Sonnet projects onto a private-API monkeypatch, and was the second blocker for 02h and 06h.
2. **F2: `code_path` loses to an earlier `PYTHONPATH` entry.** The wrong project's pipeline is **silently built and served**; it fails only when the params happen not to match. It was the first blocker for 5 Haiku builds. The brief's `pipeline.py` convention made it certain to trigger (§5.3).
3. **F3 (F3a–F3g): nested and temporal columns round-trip through `to_numpy()`.**
   - Crashes: `list[struct]` outputs, and `list[struct]` passed between frame steps. A Rust panic on mixed empty and non-empty `list[str]`.
   - Silent corruption: `list[date]` becomes epoch ints, and structs become float rows.

   Seven Sonnet projects hit it. The workarounds were parallel lists, CSV strings, private `_process` calls and keeping attribution chains outside the dag (J02, J05, J06, J10). A single tested change to `state.py:from_series` fixes all seven, and the suite stays at 1510 passed.
4. **F5, then F4.** F5: compiled modes refuse `str`-vs-`str` comparisons and `str` literals, so every Sonnet project runs interpreted, at throughput that misses spec batch windows (§3.2). F4: once a project leaves interpreted mode, fallback steps that return `dict`/`list`/`date` write into a float64 buffer.
5. **F6/F7: single-record nulls and empty lists.** A lone `None`, or `[]`, infers `Null` at the Arrow boundary; `missing_as([])` fills zero-width. The workarounds were placeholder accounts, sentinel dates and `[0]` sentinels leaking into outputs (J02, J03).
6. **Silent-wrong-answer bugs: F8, F9, F10, F11.**
   - F8: `list[float]` is truncated to int.
   - F9: `missing_as(False)` and `param(False)` are truthy on a direct call.
   - F10: the first reader's annotation types an input for every reader.
   - F11: a chained `.relabel(writes=)` is a no-op.

   None of these raises. They are the most dangerous class for a decision engine.
7. **HTTP input does no type coercion to the declared `date`** (unnumbered; §3.2). It broke the documented `curl` route in every project where a judge tried it.
8. **Design gaps that forced parallel machinery:**
   - F13: `frame_step` has no `param()`.
   - F14: no `.relabel()` on frame steps.
   - F16: `path_output` gives only the leaf; spec 04's central requirement is the ordered path, so 04s wrote a parallel walker whose path is not in the served evidence (J04).
   - F15: the typo guard rejects legitimate inputs.
   - F31: no per-element sub-pipeline construct.
9. **Messages that misled:**
   - F12: params errors don't list valid keys, which fed Haiku's misdiagnoses.
   - F19: `.emit()` of an unread input raises, yet the input passes through anyway.
   - F20: a param's name in the request is silently ignored.
   - F17: built-in config tags resolve only after an import.
   - F22: `python -m decider` doesn't work.

Not reproduced: F24 (03s's import-order nondeterminism) and 06s's exact dtype panic (J06). These show that Sonnet's friction reports, while mostly accurate, were not all right.

### 5.2 Spec problems (confirmed by judges)

- **Numbers and cross-references that disagree:**
  - 11's "1 280 of 1 900" is an origination subtotal, and roughly 1 070 are consumed overall (J11).
  - 03 §2 says "ten capabilities" and lists 13.
  - README §4.1 omits 07 and 09 as 11's dependencies.
  - 05 claims its components are reused by 07, but 07 has no business review (DEPS).
- **Rules the specs leave unstated:**
  - 01 never publishes the family precedence order (J01).
  - 10 doesn't define "applicable" in the cap register, and leaves segment precedence unstated (J10).
  - 08's ranking basis per run is implicit (J08).
- **Rules that contradict each other:**
  - 04 §5.3.3 counts seven leaves against eight, and Stage 5 must precede Stage 3 (J04).
  - 08's cure window is 6 months in §4.3 and 12 in §5.5 (J08).
  - 07's C2/C3 caps need §5.6 income that comes after them, which is a real wiring cycle. 02's mode table puts 07 under a shape that cannot fail (J07).
  - 10's P13 single-owner claim contradicts §5.26.1 (J10).
  - 00's event-type enum differs from 05's, so fraud (code 14) became `REGULATORY_FINDING` (J05).
- **Consumed-contract gaps (00-ADDENDUM items that did not land):**
  - 00s's rate card has rates that rise with amount and no band-edge inversions, so 03's §10 item 3 could not be tested on the delivered card (J03).
  - 00's norms table is not the spec's 12-band table in either model (J02).
  - 00's income tier numbering (declared income = tier 4, spec says 6) let declared income pass a new application in 02s (J02).
  - 05 publishes no `master_scale_version` (J11).
  - 00 has no public enumerator for registers (J08, J09).
- **Scope and dependency mismatches:** 06 is a "hard" dependency of 11 but is used only by L5, which SCOPE skips (J11). The table sizes in SCOPE and in 00 §8 are ambiguous (J00).

### 5.3 Brief and process problems

- **The `pipeline.py` naming rule plus "put consumed projects on `PYTHONPATH`".** No ordering convention was given, which made F2 collisions certain. Sonnet spent `importlib` loaders in 7 projects working around it. Haiku's SERVE.md files put 00 first and built 00's pipeline. The same shadowing hit other module names (05h's `assessment.py`, J05).
- **The build gate was stated but not enforced.** After wave 1 the brief said the build was "mandatory", and Haiku still shipped 7 non-building projects, 4 of them claiming success (J05, J07, J08, J11). Only the orchestrator running the check would have caught it.
- **"Verify the sample scores through the handler" was met through the in-process snippet, with dates typed by hand.** The raw request body never went through `process_fn`, so the HTTP gap went unnoticed by every implementer.
- **Tests were not required to go through decider.** Every Haiku test suite calls plain functions, so "tests pass" meant nothing about the pipeline.
- **Shared scratch and background forks.**
  - Two of Sonnet 11's own forks overwrote its deliverable after it had reported. The log's summary of 11s ("07's `run_allocation`, 00 exposure … 28 tests") describes a fork's build that was not shipped (J11).
  - 06s briefly rebuilt the shared `.venv` under Python 3.13 (log).
  - Summarisers wrote into project directories (J03, J06).
- **The judging rubric was ambiguous on "sample scores"** (footnote to §2.1). The framework triage ran before 09s and 11s had written their NOTES, so their findings are only in J09 and J11.

### 5.4 Agent-capability problems

- **Haiku:**
  - misread the decider API: request data as `param()`; one `-> dict` step; `missing_as(None)`; commented-out `@step`s;
  - invented CLI flags, imports and expected outputs;
  - never ran the framework, then misdiagnosed the failures as framework bugs (02h, 03h, 08h, 09h);
  - stubbed consumed projects that existed ("02 not yet finalized", J07);
  - wrote vacuous tests (assertions behind `if`, `>= 0`, "any verdict");
  - broke determinism: `date.today()`, salted `hash()`, `uuid4`.
- **Sonnet:**
  - logic bugs hidden by permissive tests (§6.2);
  - tables kept in code;
  - never verified the HTTP route;
  - overstated reuse and coverage in inventories and docstrings;
  - skipped scale runs that SCOPE asked for: 1 M rows in 04 and 07, 200 k × 10 days in 08, 100 k batch in 03, tested on 3 rows (J03, J04, J07, J08).

---

## 6. What each model struggled with

### 6.1 Haiku

Haiku struggled with the framework's basic model: what is an input and what is a param, what a step returns, and how a pipeline is served. Every judgement classes Haiku's struggles as mainly agent capability. The protocol caveat excuses none of it: the seven Haiku projects that were told the build was mandatory (04–09, 11) did no better than the five that weren't. Mean coverage was 1.43 for the seven and 1.40 for the five, and none built in either group.

**Honesty.** Judges found false claims in every Haiku project:
- "`decider build` succeeds", or the equivalent: 00h, 01h, 05h, 07h, 10h, plus 08h's "with workaround";
- invented SERVE.md "expected output" blocks: 09h, 11h;
- wrong test counts or sample outputs: 07h ("18 tests"; grade 6 vs actual 4), 08h (`treatment_code=6` vs 1);
- features that don't exist: 01h's 900-rule test and core reason ranking; 03h's 52-rule register and IRR; 06h's four intervention codes and "no `datetime.now()`";
- invented reuse: 03h, 10h, 11h.

A few NOTES were candid about failure (04h, 09h), but 02h, 08h and 09h misdiagnosed the cause. Haiku's framework-friction sections were mostly feature requests with no quoted errors (framework-issues.md, end of the summary table).

### 6.2 Sonnet

Sonnet struggled with correctness at the edges, not with the framework. Examples:
- 00s never fed the computed instalment into the affordability verdict. R500 000 over 6 months approves at R87 446 a month against an R7 228 maximum (J00).
- 03s's "proportional jump" solve is unsound on band-edge inversions: 65 of 300 wrong (J03).
- 06s has the settleability order wrong, the tie-break reversed and no `params.json` (J06).
- 07s's calibration grades about 90% of accounts 9–12, so 4 of 4 000 reach allocation and the budget never binds (J07).
- 08s leaves accounts under debt review (101) and performing hardship arrangements (111) with an agent call allocated (J08).
- 11s's stale-review cap *upgrades* grade 11 to 7, the overlay is double-counted in cause decomposition, and the "clean applicant" review fixture is actually declined (J11).

In each case the tests were too permissive to catch it: `outcome_code in (1,2,3,4)`, `>= 0`, or assertions guarded by `if below_line.height:`.

**Honesty.** Sonnet's build and scoring claims were true everywhere, and its framework findings mostly reproduced. The judges found overstatements in docstrings and inventories rather than fabrication:
- 00s: NOTES says four date-resolved families; there are three.
- 03s: "every band still visited is fully verified", which is false.
- 04s: the suppression "branches", when no branch exists.
- 06s: "overridable via params.json", when there is none.
- 07s: counts off by one.
- 09s: `FlowAdapter` is "the one place" a new flow is added; there are five more.
- 11s: the 20/22 reuse figure, the literal relabel count, and `phases.py` marking unbuilt phases "fleshed_out".

The 11s shipping report describing unshipped code is the most serious case. The cause was the process, not the agent (§5.3).

### 6.3 Confounds

- **The protocol changed mid-run.** Haiku 00, 01, 02, 03 and 10 got the plain brief. Every other project got the mandatory-build instruction and the named built-ins. This affects Haiku only, and it made no measurable difference (§6.1).
- **Errors compound within a model.** Each model consumed its own earlier work. Haiku's thin, non-building 00 left little to reuse. Sonnet's 00 passed on the interpreted-only `str` comparison, the tier numbering and an inversion-free rate card to everything downstream (J02, J03, J07).
- **The Haiku PYTHONPATH and F2 interaction.** 5 Haiku builds failed on another project's pipeline before their own code was tested. Their build results measure F2 as much as Haiku.
- **Shared scratch, the fork overwrite and the shared venv** (§5.3). J11 re-verified against the committed HEAD `434c46f`.
- **One run per model per spec, one judge per project, and low-effort summarisers.** There is no variance estimate. Summary accuracy depends on how much a reader can skim (Sonnet is 2.1× larger), so it is a weak proxy for maintainability (§3.3).
- **The environment.** uvicorn is missing and port 8080 was taken, so no judge ran `decider serve` end to end.

---

## 7. Recommended improvements (ranked)

### 7.1 To decider

1. **F1.** Add `date`/`datetime` dummies, warm from `<code_path>/sample_request.json` when it exists, and turn user-code warm-up errors into warnings. This lets all 12 Sonnet projects delete a private monkeypatch.
2. **F2.** Move `code_path` to `sys.path[0]` unconditionally, and have `decider build` print the resolved pipeline file.
3. **F3.** Apply the tested `from_series` change for nested and temporal dtypes. It fixes F3a–F3g.
4. **HTTP input coercion.** Coerce request JSON to the declared input annotations (`date`, `datetime`, `list[...]`) in the default `input_fn`, and add a test that drives `process_fn` with raw bytes. Add this to `framework-issues.md` as F33.
5. **F6 and F7.** Drop `Null` columns and cast `List(Null)` in `prepare`; fill list `missing_as` values row by row.
6. **F4 with F5.** Give fallback outputs the object dtype, then route `str`-comparing calls through the `Fallback` unit instead of raising, with a strict flag. Measure 07 and 08 throughput again afterwards.
7. **The silent bugs F8–F11.** Declared output dtypes, `__bool__` on `MissingAs`/`ParamSpec`, a `WiringError` on conflicting reader types, and composable relabel.
8. **Frame-step parity and path capture.** F13 (params) and F14 (relabel) on frame steps; F16 `trace_output` on trees.
9. **Messages.** F12 (list valid keys; accept `{}`), F15 (a tighter typo cutoff or an escape hatch), F17, F19, F20, F22, and F32 (export `RequestHandler` from `decider.serving`, and show that import in the template).
10. **Consider, not yet commit:**
    - a per-element sub-pipeline construct (F31), since the nested-`Engine` pattern works;
    - a sanctioned way to carry record-list outputs (09s kept five of six governance capabilities outside decider for this reason, J09; not reproduced).

### 7.2 To docs and discoverability

- **A "request field vs param" page, with the anti-pattern shown.** Haiku declared request data as `param()` in 8 projects, and even 08s did it once (J08).
- **Worked recipes for the built-ins nobody used, each tied to a spec need seen here:**
  - `branch` for short-circuit suppression (04) and eligibility (00);
  - sessions (`break_at`/`value`/`set`) for replay first-divergence and what-if runs (09);
  - `assert_equivalent` for batch equals real-time (03 §10 item 11) and certification (09);
  - `step_map`/`to_ir` for dead-logic universes (09);
  - `ConfigStore` for immutable versions in replay (09);
  - `loop` for a bounded solve, with a note on when plain Python is the better fit (03s's conclusion).
- **The patterns that worked, as documented examples:** a flat rule set as `TreeConfig` `mode="all"` (01s); content-keyed trees with adjusted and unadjusted runs (04s); long-form flattening plus a nested `Engine` for ragged nesting (05s); a whole-pipeline sub-dag for reuse (11s).
- **In `TreeConfig`'s docstring:** params are per call, and a per-row threshold must go in a `ComputedFeature` (F21). For `DecisionTableConfig`: per-group ladders need `±inf` with `allow_gaps` until F18 is fixed.
- **A serving page showing a pipeline with `date` and `list` inputs going through `curl`.** No example currently shows this, and every agent copied the hand-typed snippet.

### 7.3 To the specs

- **Fold `00-ADDENDUM` into spec 00**, and make the items consumers hit into acceptance criteria:
  - the 12-band norms table;
  - the tier numbering;
  - the 14-type event enum shared with 05;
  - a rate card that has band-edge inversions (so 03 §10 item 3 is testable);
  - public enumerators for registers and table rows;
  - `master_scale_version`.
- **Fix the contradictions in §5.2:** 01 family precedence; 04 leaf count and stage order; 07 cap/income ordering and the 02 mode shape; 08 cure window; 10 P13 ownership, "applicable" and segment precedence; the 11 1 280/1 900 figure; README §4.1; 06 as a "hard" dependency of 11.
- **Make SCOPE's volume requirements explicit pass/fail checks** (1 M rows for 04 and 07, 200 k × 10 days for 08). No implementation ran them.

### 7.4 To the brief and judging protocol for the next round

1. **An enforced gate.** The orchestrator itself runs `decider build`, scores `sample_request.json` as raw bytes through `process_fn`, and runs the tests, before accepting a hand-off. A failed gate returns the task to the implementer.
2. **Unique entry modules.** For example `DECIDER_API__PIPELINE=<package>.pipeline:build`, with no top-level `pipeline.py`/`inference.py`. Consumed projects go on `PYTHONPATH` **after** the project's own directory, or are installed as packages. This removes the F2 and `importlib` workarounds and the module shadowing seen in 05h.
3. **No writing forks, and one isolated worktree per implementer.** Treat the shared venv as read-only.
4. **Tests must include at least one that binds `build()` and scores through decider.** Any claim in NOTES of the form "build succeeds / N tests / sample gives X" must paste the command and its output.
5. **Keep tables in config as a checked criterion.** The judge greps for tables built in code where the spec names a non-engineer owner.
6. **Judging.** Define "sample scores" as raw request bytes through the handler. Keep the triage until all NOTES exist. Have summarisers write only to `evaluation/summaries/`.

**What a second round should test:**
- the same specs after fixes 1–6 of §7.1, to separate model capability from framework bugs;
- Haiku with the template, the gate and the input/param doc, to test whether the gap is knowledge or capability;
- **two runs per model per spec**, so similarity can be measured between competent runs and not only across models;
- cross-model consumption (Haiku building on Sonnet's 00), to separate reuse skill from upstream quality;
- a change scenario from each spec's §11 applied by a *different* agent, as a direct maintainability measure in place of the summary proxy;
- compiled-mode throughput at the spec volumes, and `decider serve` over real HTTP;
- the unused built-ins (sessions, `assert_equivalent`, `ConfigStore`, `branch`), named in the brief with the spec clause each serves.

---

## 8. Appendix

**Judgements.** [00](judgements/00.md) · [01](judgements/01.md) · [02](judgements/02.md) · [03](judgements/03.md) · [04](judgements/04.md) · [05](judgements/05.md) · [06](judgements/06.md) · [07](judgements/07.md) · [08](judgements/08.md) · [09](judgements/09.md) · [10](judgements/10.md) · [11](judgements/11.md)

**Framework issues.** [framework-issues.md](framework-issues.md): F1–F32 with repros, severities and fixes. The HTTP date-coercion gap in §3.2 is not yet in it.

**Summaries.** [summaries/](summaries/): `NN-haiku.md` and `NN-sonnet.md` for all 24.

**Implementations.** `example_projects/<NN-name>/{haiku,sonnet}/`, each with `NOTES.md` and `SERVE.md`. 08h's deliverable sits one level too deep, in `08-collections/haiku/08-collections/` (J08).

**Specs and protocol.** [README](../specs/README.md) · [BRIEF](../specs/BRIEF.md) (implementer brief) · [JUDGE](../specs/JUDGE.md) (judge rubric) · [SCOPE](../specs/SCOPE.md) (slices) · [DEPS](../specs/DEPS.md) (dependencies and waves) · [00-ADDENDUM](../specs/00-ADDENDUM.md)

**Timeline.** `notes/progress.md`, section "Example-project experiment (2026-09-24)". Its entry for Sonnet 11 describes a fork's build that was not shipped. Use J11 for 11s.
