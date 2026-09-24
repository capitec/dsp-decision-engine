# Usability loop on steering and tracing a run

Five stories in `tools/vscode-decider/test/e2e/features.e2e.ts` (`pnpm test:features`), four on the
1,000-step bank flow and one on `loan.py`'s loop:

- F1: compare two products (branch arms) for one applicant.
- F2: force one applicant down another product mid-run and see the outcome.
- F3: a value breakpoint limited to part of the flow.
- F4: find which step set a value, see its history, and go back there.
- F5: compare 5 and 10 loop iterations, then pause at iteration 3.

Each round, a fresh judge agent scores the screenshots alone, 1 to 10 per story. A story whose goal isn't
visibly met caps at 5. The loop stops at the first of these:

- an overall score of 8 or more;
- 10 rounds;
- two consecutive rounds that differ by less than 0.5.

| Round | Overall | F1 arms | F2 force | F3 value bp | F4 history | F5 loop |
|---|---|---|---|---|---|---|
| 0 | 6.4 | 6 | 5 | 7 | 7 | 7 |

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
