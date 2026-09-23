# Usability loop on a 1,000-step credit flow

The extension on `tools/vscode-decider/examples/bank` (about 1,000 steps in 40 files, 1,400 params,
14 lookup tables with shared rows, 3 trees, 40 applications). Each round, `pnpm test:large` screenshots six
user stories and a fresh judge agent scores them from the screenshots alone (1-10 per story; a story whose
goal isn't visibly met caps at 5). The loop stops above 8/10 or after 40 rounds.

| Round | Overall | L1 find | L2 table | L3 why rate | L4 param | L5 sweep | L6 commit |
|---|---|---|---|---|---|---|---|
| 0 | 3.3 | 5 | 2 | 2 | 3 | 2 | 6 |

## Round 0 (baseline)

Judge's problems, most damaging first:

1. Lookup tables show as "empty" one-line text boxes marked "default": no rows to edit. What-if runs without
   the module's PARAMS. Wants an inline grid per table, filled from the real values.
2. repo_rate 7.75% to 7.5% reports "0 steps changed" with no explanation (the cap never binds).
3. The results list under "No result changes" shows every feature column for every client: noise.
4. "Run to pl_regulated_rate" drops the selection; State says "No session running"; no sign anything ran.
5. Nothing explains one applicant's rate: which table row matched, loadings, discounts, cap.
6. No find box on the graph for 1,000 steps.
7. Scenario param picker is a plain dropdown over 1,400 params; a half-filled knob row is silently ignored.
8. What-if params are one flat list of 1,400 with no filter or groups.
9. No "used by" list for a shared param.
10. "fit" on 1,000 steps shows a near-empty canvas; wants top-level groups collapsed with step counts.
11. Compare spends the view on "Unchanged"; the changed-step navigator is truncated and doesn't scroll the graph.
12. Panel title flips between the root flow and a sub-flow; data-link arcs hide the graph.
