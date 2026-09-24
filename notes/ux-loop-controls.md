# Usability loop on steering and tracing a run

Five stories in `tools/vscode-decider/test/e2e/features.e2e.ts` (`pnpm test:features`), four on the
1,000-step bank flow and one on `loan.py`'s loop:

- F1: compare two products (branch arms) for one applicant.
- F2: force one applicant down another product mid-run and see the outcome.
- F3: a value breakpoint limited to part of the flow.
- F4: find which step set a value, see its history, and go back there.
- F5: compare 5 and 10 loop iterations, then pause at iteration 3.

**Result: stopped after round 1 (6.4 → 6.6). The change was under 0.5, which meets the stop rule.**

Each round, a fresh judge agent scores the screenshots alone, 1 to 10 per story. A story whose goal isn't
visibly met caps at 5. The loop stops at the first of these:

- an overall score of 8 or more;
- 10 rounds;
- two consecutive rounds that differ by less than 0.5.

| Round | Overall | F1 arms | F2 force | F3 value bp | F4 history | F5 loop |
|---|---|---|---|---|---|---|
| 0 | 6.4 | 6 | 5 | 7 | 7 | 7 |
| 1 | 6.6 | 7 | 7 | 7 | 6 | 6 |

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
