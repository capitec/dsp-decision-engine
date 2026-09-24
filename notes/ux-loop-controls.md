# Usability loop on steering and tracing a run

Five stories in `tools/vscode-decider/test/e2e/features.e2e.ts` (`pnpm test:features`), four on the
1,000-step bank flow and one on `loan.py`'s loop:

- F1: compare two products (branch arms) for one applicant.
- F2: force one applicant down another product mid-run and see the outcome.
- F3: a value breakpoint limited to part of the flow.
- F4: find which step set a value, see its history, and go back there.
- F5: compare 5 and 10 loop iterations, then pause at iteration 3.

Round 1 first stopped the loop on a change under 0.5. The rule is now three rounds in a row within 0.5.

Each round, a fresh judge agent scores the screenshots alone, 1 to 10 per story. A story whose goal isn't
visibly met caps at 5. The loop stops at the first of these:

- an overall score of 8 or more;
- 10 rounds;
- three consecutive rounds within 0.5 of each other.

| Round | Overall | F1 arms | F2 force | F3 value bp | F4 history | F5 loop |
|---|---|---|---|---|---|---|
| 0 | 6.4 | 6 | 5 | 7 | 7 | 7 |
| 1 | 6.6 | 7 | 7 | 7 | 6 | 6 |
| 2 | 7.0 | 7 | 7 | 8 | 7 | 6 |

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
