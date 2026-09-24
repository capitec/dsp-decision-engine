# Usability loop on steering and tracing a run

Five stories in `tools/vscode-decider/test/e2e/features.e2e.ts` (`pnpm test:features`), four on the
1,000-step bank flow and one on `loan.py`'s loop:

- F1: compare two products (branch arms) for one applicant.
- F2: force one applicant down another product mid-run and see the outcome.
- F3: a value breakpoint limited to part of the flow.
- F4: find which step set a value, see its history, and go back there.
- F5: compare 5 and 10 loop iterations, then pause at iteration 3.

The stop rule changed twice. Round 1 stopped on a change under 0.5. Round 3 stopped with three rounds within 0.5.
The scores were still climbing, so the rule became three rounds without a new best.

Each round, a fresh judge agent scores the screenshots alone, 1 to 10 per story. A story whose goal isn't
visibly met caps at 5. The loop stops at the first of these:

- an overall score of 8 or more;
- 10 rounds;
- three rounds in a row without a new best overall score.

| Round | Overall | F1 arms | F2 force | F3 value bp | F4 history | F5 loop |
|---|---|---|---|---|---|---|
| 0 | 6.4 | 6 | 5 | 7 | 7 | 7 |
| 1 | 6.6 | 7 | 7 | 7 | 6 | 6 |
| 2 | 7.0 | 7 | 7 | 8 | 7 | 6 |
| 3 | 7.1 | 7.5 | 7 | 8 | 6 | 7 |
| 4 | 7.4 | 8 | 7 | 8 | 6 | 8 |
| 5 | 7.2 | 7.5 | 6.5 | 7.5 | 7.5 | 7 |
| 6 | 6.9 | 7 | 7 | 7.5 | 6 | 7 |

## Round 0 (baseline)

The judge's problems, most damaging first:

1. F2: the offer_rate explanation contradicts itself. `product_arm = 1` is shown as computed from
   `product_code = 0`, with no mention of the force.
2. F2: the final offer is never shown. Wants the outcome (decision, reason, amount, rate) after a forced re-run.
3. F1: an applicant that turns into a decline still shows offer amount and rate as if they were an offer.
4. F1: the comparison is a diff (old struck through → new), not side by side per arm.
5. The branch controls sit below the fold, and they also show on every step inside the branch.
6. Branch edges are labelled with raw codes (`product_arm = 2`); the focused record's arm isn't highlighted.
7. Compare: 430 "downstream" rows of "wrote empty" from the arm not taken. The plurals are wrong ("1 steps").
   In a loop comparison, the changed step itself is called downstream.
8. F4: after going back, later steps still look as if they ran in the graph.
9. F4: the history leaves out pl_rate_floor, which wrote pl_rate without changing it. "not set" and
   "empty →" say the same thing twice.
10. F3: the chip says 25%, the pause banner says 0.25. Adding a breakpoint gives no confirmation, and the
    pause doesn't focus the matching record.
11. F5: the iteration input still shows its old value after a breakpoint is added. Number formats are mixed.
    Nothing says how many iterations the loop runs by itself.
12. The details pane is cluttered. The "0 edited · 1 knock-on" toolbar text is jargon.

## Round 1: 6.6

Changes:

- A branch's controls show at the top of the details, and only on its condition step. A loop's controls show on
  every step inside the loop.
- The explanation marks a forced value: "product_arm = 1, forced by you; product_arm gave 0". The value history
  shows the force as its own change.
- The pause banner gives the focused record's outcome from the flow's own emits and top-level branches. A decline
  shows the decision and reason, then "no offer is made".
- Compare result cards are a two-column table. The headers are the words that tell the runs apart. A side that
  declines has its figures greyed and struck through.
- Steps that only changed because records switched arm fold into two lines: "213 steps in personal_loan no longer
  ran…" and "213 in credit_card ran instead". Plurals are fixed, and a force isn't called "downstream".
- The value history shows steps that kept a value ("kept 25.2% by pl_rate_floor"). "not set" is gone.
- After a rewind, the graph ticks only what has run since (the bridge reports it).
- Breakpoint text matches its chip (25%, not 0.25). A breakpoint that fires focuses the first matching record.
  Adding one says so in the form. The iteration box starts empty.
- `offer` formats as money. The navigator says "changed steps: 1 edited, 3 as a result", not "knock-on".
- "Run to" is one-shot, so it no longer leaves a breakpoint that stops the next run.

Backlog, the judge's problems most damaging first:

1. The VS Code Variables pane and the "explain a value" list still show a declined record's offer figures as
   plain values.
2. "Which step gave the rate" credits pl_regulated_rate, which passed pl_raw_rate through unchanged. Wants the
   origin: the cap didn't bind, so the rate came from pl_raw_rate.
3. WRITES shows the previous iteration's value for a step that hasn't run yet in this iteration.
4. After going back, the history drops the later entries. Wants them kept, greyed, as "will re-run".
5. It's unclear whether picking an arm forces it already or only after "Re-run".
6. Forcing and comparing share one block. It's unclear whether "only client_id" narrows the comparison.
7. The focused record's arm can be off canvas, and it isn't highlighted.
8. Wants the breakpoints listed on the step, and in VS Code's Breakpoints view.
9. Units in the value box: show "25% or 0.25" and echo the parsed value.
10. The records a breakpoint matched ("and 5 more") aren't clickable.
11. Details open scrolled down. The action row varies between steps.
12. A loop comparison says "as a result of the force". The current iteration isn't shown on the loop. An edge
    label overlaps a node.

## Round 2: 7.0

Changes:

- Decline marking:
  - The VS Code Variables view marks a declined record's offer figures "(not offered: declined)".
  - The "explain a value" list marks them the same way.
- Pause banner:
  - It shows the loop iteration.
  - A value breakpoint's records are links that focus them.
- History and WRITES:
  - Going back keeps the undone changes in the history, greyed, as "will re-run".
  - A step that passed its input through unchanged names where the value came from.
  - WRITES on a loop step not yet run in this iteration says whose value it shows.
- Branch controls:
  - They split into "Force it in the debug run" and "Compare two ways".
  - They say when a force applies, and that Compare runs the whole flow twice.
- Breakpoint form:
  - It lists this step's breakpoints.
  - It hints at units ("e.g. 25% or 0.25") and echoes the parsed value.
- Going back stops with the `goto` reason, not "breakpoint".
- The test harness allows more time to launch and to take screenshots. The machine ran at a load of about 30 from
  other sessions.

The judge's problems, most damaging first:

1. F5: "It passed offer through unchanged, so the value comes from shrink_offer" is wrong inside a loop. shrink
   multiplies by 0.8, and shrink_offer is the loop.
2. F1: "40 records" for a one-client comparison.
3. F1: the columns don't say which is the baseline. The note says "greyed" but the figures are struck through.
4. F2: after "Re-run product forced", the graph shows the parent with no sign of the forced arm.
5. Arms show as numbers ("product_arm = 1 … gave 0") and edges read "product_arm = 2".
6. F4: two answers to "which step" (pl_regulated_rate set it, the value comes from pl_raw_rate). Wants one
   headline.
7. F4: "Go back to that moment" is still offered at that moment. Wants "you are here".
8. An edge label overlaps a node title.
9. The explain list truncates "(not offered: declined)".
10. The loop controls look disabled, and the iteration box's placeholder is "k".
11. The graph has no legend for its node styles.
12. Wording differs between places:
    - the scope reads "only in product / personal_loan" in the form but "in personal_loan" on the chip;
    - "applies when the run reaches product" is shown while the run is inside product;
    - the hit list is long.

## Round 3: 7.1

Changes:

- History headline: one line that names the step that computed the value, then the steps that passed it on or kept
  it. The passed-through line isn't used inside loops, where the breakdown shows the last iteration.
- History position: "◀ you are here" marks the moment the run is paused at, with no go-back link on it.
- A forced condition names its arms: "forced by you (down credit_card); product_arm gave 0 (down personal_loan)".
- Compare:
  - The baseline column is marked.
  - The decline note matches the strike-through.
  - A one-record force says that all records ran.
- "Re-run forced" pauses before the condition step, so the branch's controls stay in view.
- Wording:
  - Forcing says it applies "the next time product_arm runs".
  - A breakpoint's scope reads the same on the chip and in the form.
  - The loop boxes have example placeholders.
- A long list of hit records collapses to three and "and N more".

Backlog, the judge's problems most damaging first:

1. F4: the numbered history still starts at pl_regulated_rate. The headline credits pl_raw_rate. "Go back" goes
   to pl_regulated_rate, not to the step that computed the value. The timeline tracks one name, so it can't list
   a value that arrived under another name (pl_raw_rate → pl_rate) as its first row.
2. F2: after "Re-run forced", nothing says which arm the record now takes, and "applies the next time" still
   shows.
3. F1: "for client_id 20400" in the control against "all 40 records" in the result. Wants "the other 39 are
   unchanged".
4. F1: 430 changed steps against 213 + 213 folded; the other 4 aren't accounted for.
5. Comparison titles like "product down credit_card for client_id 20400" read as internal names.
6. The arm the record takes can be off canvas; wants the branch fitted and the taken arm highlighted.
7. The sticky action row in the details covers the top of the content.
8. Compare highlights and the changed-step navigator stay on during a later debug run. The graph has no legend.
9. An edge label overlaps a node, and an edge runs into a clipped box.
10. "Break when a value…" sits below the fold. After adding, the empty form reads as a failed add.
11. The step's action row varies ("Run through…" vs "Change the run ▾"). The step description sits below the
    branch controls.
12. A loop comparison doesn't say how many iterations each record runs by itself, or that a force overrides
    the while condition.

## Round 4: 7.4 (best so far)

Changes:

- Value history:
  - "Go back" can target a step by path. Where a step passed a value on, the history starts with the step that
    computed it, and going back lands there.
  - The history stays in view after going back past the step that first set the value.
- Branch and loop controls:
  - The force chip reads "product → credit_card".
  - Comparison sides read "client_id 20400: credit_card (at product)".
  - Paused just before the condition, forcing says what will happen when it runs.
  - Compare says every record runs and only the named one is forced.
  - A loop says a forced count overrides its condition.
- Compare:
  - It says "Only client_id 20400 was forced; the other 39 ran as they are".
  - The step count splits into stopped, ran-instead and changed values.
- Graph and details:
  - The graph has a "Key" for its node styles.
  - A new debug run hides an earlier comparison's colours.
  - The breakpoint form sits with the step's controls and closes after adding, leaving its confirmation.
  - The details header casts a shadow, and scrolled-to sections clear it.

The judge's problems, most damaging first:

1. F4: after going back to pl_raw_rate, the history says "Nothing has set pl_rate yet" and lists pl_regulated_rate
   as "set to", contradicting the screen before. Wants the same steps and wording at every pause.
2. F4: "pl_raw_rate computed it (as pl_raw_rate)" mixes names. Wants "the value came from pl_raw_rate;
   pl_regulated_rate copied it into pl_rate unchanged".
3. F2: "Re-run product forced" when already paused before the condition looks like a no-op, and its note reads as
   still in progress.
4. F3: "(1 on this step)" implies the breakpoint belongs to one step. Its scope is the whole personal_loan
   branch.
5. F1/F2: the "only client_id" checkbox is easy to miss, and it scopes both actions. Wants the scope inside each
   action.
6. F1: the comparison header repeats the client and reads like code. The step-by-step block is noise for an
   analyst.
7. An edge label overlaps a node.
8. Disabled buttons look like plain text.
9. The step's action row varies between steps.
10. The Variables pane truncates "(no offer)" so the figures look like an offer.
11. The record's arm can be off canvas.
12. "Will re-run" colours are explained only behind the collapsed Key.

## Round 5: 7.2 (1 round without a new best)

Changes:

- The timeline records, for each change and record, an input the step copied unchanged, and which step computed
  that input. The history reads the same at every pause:
  - "the number comes from pl_raw_rate; pl_regulated_rate copied it into pl_rate unchanged; pl_rate_floor kept it";
  - after going back: "Nothing has set pl_rate yet. The number will come from pl_raw_rate…".
- "Re-run forced" is hidden when the run is already paused just before the condition.
- The record scope is a picker inside each action ("Send [every record / client_id 20400] down …").
- The breakpoint summary counts active breakpoints.
- Disabled buttons look disabled.
- The Variables pane puts "(no offer)" first.
- A forced comparison folds its step-by-step list.
- F2 now starts paused at the last step, then forces and re-runs.

The judge's problems, most damaging first:

1. F2: "stop forcing" reads as plain text. Rename the re-run button "Re-run product with the force".
2. F2: the status text starts with an unexplained "On.".
3. F2: re-running from the last step leaves the user before product_arm. Wants to come back to where they were.
4. F2: the outcome banner shows internal flags and empty fields ("declined = false · decline_reason = ").
5. F3/F4: "Paused after" while the editor arrow sits on the step's def, as if it hadn't run.
6. F5: a loop comparison doesn't show what the loop does by itself.
7. Money is formatted inconsistently: R 49,152 next to R 16,106.13.
8. An edge label overlaps a node.
9. F1: the force and compare record pickers change together.
10. F1: the header and summary are noisy. Wants one sentence: "As a credit card, 20400 would be declined: …;
    as a personal loan: approved …".
11. The details panel is crowded:
    - four go-back links;
    - "copied R 150,000 into offer unchanged by offer" reads as circular;
    - "How X changed" should come first.
12. The action row varies. Breakpoint chips have a red, error-like border. Arcs and teal nodes aren't explained
    on screen.

## Round 6: 6.9 (2 rounds without a new best)

Changes:

- "Re-run with the force" goes back, re-runs and returns to where the run was paused, in one bridge command
  (`rerun`). Two requests (a rewind, then a continue) left VS Code's Variables view showing the values from before
  the re-run.
- "Stop forcing" is a button. The status text reads "Force is set: it applies the next time product_arm runs".
- The outcome line leaves out true/false flags and empty values.
- Money always shows two decimals.
- Paused after a step, the editor points into its body.
- The force and compare record pickers are separate.
- A one-record forced comparison leads with one sentence.
- In the value history:
  - it comes first in the details;
  - each line names its step first ("pl_regulated_rate copied pl_raw_rate into pl_rate unchanged");
  - each step has one go-back link.
- Breakpoint chips have a neutral border.

The judge's problems, most damaging first:

1. F4: after going back, the breakdown says pl_rate "= empty… It is an input", which is wrong.
2. The breakdown's "last written by pl_rate_floor" contradicts the history's "comes from pl_raw_rate".
3. The same answer three times: the summary, the list and the "Why" tree.
4. The sticky step header covers the loop controls.
5. The loop's Break section doesn't show the iteration breakpoint that's set.
6. Paused after pl_regulated_rate, the editor highlights its docstring.
7. The compare scope ignores the focused record.
8. Variables truncates "(no offer) R 118,…".
9. The comparison's title, lead sentence and baseline line repeat each other. The decision row isn't emphasised.
10. The step counts ("430 steps changed (213…)") are noise.
11. A loop comparison doesn't show the iteration count per side.
12. The re-run button's label is long, and the selection jumps away from the force controls after it.
