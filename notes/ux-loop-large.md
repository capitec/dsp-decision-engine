# Usability loop on a 1,000-step credit flow

The extension on `tools/vscode-decider/examples/bank` (about 1,000 steps in 40 files, 1,400 params,
14 lookup tables with shared rows, 3 trees, 40 applications). Each round, `pnpm test:large` screenshots six
user stories and a fresh judge agent scores them from the screenshots alone (1-10 per story; a story whose
goal isn't visibly met caps at 5). The loop stops above 8/10 or after 40 rounds.

| Round | Overall | L1 find | L2 table | L3 why rate | L4 param | L5 sweep | L6 commit | L7 skip/swap |
|---|---|---|---|---|---|---|---|---|
| 0 | 3.3 | 5 | 2 | 2 | 3 | 2 | 6 | |
| 1 | 5.6 | 7 | 7 | 4 | 7 | 4 | 5 | 5 |
| 2 | 6.4 | 7 | 8 | 6 | 5 | 7 | 8 | 4 |
| 3 | 6.4 | 7 | 8 | 6 | 6 | 6 | 7 | 5 |
| 4 | 7.1 | 8 | 7 | 8 | 7 | 7 | 7 | 6 |
| 5 | 7.6 | 8 | 8 | 8 | 7 | 7 | 8 | 7 |

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

## Round 1

Changes: a find box (name, docstring, group; Ctrl+F); groups fold into boxes with step counts on flows over 80
steps and open around the selection, the pause and compared changes; What-if has a filter, folding groups, lookup
tables as editable grids filled from the module's PARAMS, "used by" for shared params; Compare lists changed results
only (changed records first) and says when a changed param's readers wrote the same values; scenario knobs are a
type-to-filter picker and a half-filled row blocks Run; a table step shows which row a record matched; the debugger
can skip a step or swap in edited code mid-run (new story L7); the pipeline title is the file's own flow, not an
imported sub-flow; the NCA cap in the example is repo + 21% so repo_rate moves offers; a paused step's source opens
left of the panel instead of over it.

Judge's problems, most damaging first:

1. L3: no single "why this rate" breakdown; paused before the cap so whether it applied is unknown.
2. L5: the sweep summary lacks approvals/offer amount; total_cost range identical in every row.
3. L6: the pricing.py table-row edit doesn't appear in the comparison at all.
4. L7: skip and swap never show their effect on offers.
5. L3: the table step is labelled "decision tree"; the matched row is below the fold.
6. Paused screens: the details pane is ~180px tall; the editor is squeezed.
7. L1: after Enter the hit isn't centred or highlighted; no breadcrumb.
8. L2: band edges overlap with no stated inclusivity; rates as decimals not %.
9. L4/L2: results pad with unchanged records; "150 unchanged results" is confusing next to 40 records.
10. What-if results land under "Compare with a git revision…".
11. "used by 8 steps" tiny and far from the value.
12. Scenario builder: truncated param field, raw money formatting.

## Round 2

Changes: the "how it was computed" card is a tree of each step's formula filled with the record's values
(`min(0.252, 0.0775 * 1 + 0.21)`), with a lookup table's matched row and why (`49 ≤ requested_term 60 < 85`);
lookup tables have their own label and icon; a paused run with edits offers "Compare with the flow as started";
the sweep summary counts outcomes (approve / decline) and averages numbers with their change; a revision's PARAMS
differences show as "Changed params" row by row; What-if results are titled with the change; results list only
changed records; table grids show rates as percentages and which band edge is inclusive; the details pane gets more
height; the example emits decision and offer_amount.

Judge's problems, most damaging first:

1. L7: comparing mid-run says "no final output changes" (outputs not computed yet) and shows empty values as changes.
2. L7: the focused record filters the changed offers out of Results; no "caused by".
3. L4: "used by" is tiny and collapsed; the steps never show.
4. Units change between screens: % in What-if, raw decimals elsewhere.
5. L3: whether the cap applied is left to arithmetic; zero terms clutter the raw-rate sum.
6. L3: the breakdown loses its expansion when drilling in; action buttons clipped.
7. L1: the found node is off screen after Enter; no breadcrumb.
8. L5: sweep cells wrap; averages over 40 records dilute a 3-record change; unchanged columns waste space.
9. L7: the rewind to the edited step happens silently.
10. L6: "Changed params" omits the code-declared limit change.
11. The "Changed:" prose is hard to scan; "default → 0.075" hides the old value.
12. The editor is squeezed; Variables shows 40 values of noise.

## Round 3

Changes: rates, loadings, discounts and margins show as percentages wherever the column is known, and params accept
"7.5%"; the breakdown hides zero terms and says whether a cap or floor applied; clicking a value opens it in place;
step actions stay pinned; a found step is centred with a breadcrumb; skip/swap leave a note in the pause banner and
the edits comparison is titled with them; a skipped step is "removed", not "values emptied"; the focused record is
pinned first in results with the changed ones after it; the changed-steps prose is a table; code-declared param
changes join "Changed params"; sweep cells say "3 of 40 · avg −0.53 pp" over the changed records, and outcomes that
never change move to a note; filtered params show their readers as chips.

Judge's problems, most damaging first:

1. L7: comparing while paused reads as if the runs had finished (they do run to the end, but it doesn't say so).
2. L7: the skip and the edit are compared only together, not one at a time.
3. L6: changed records are capped at 4 columns; one that matters is hidden.
4. L3: the breakdown is crammed under the graph; the story's click collapsed it (a test bug: it hit the outer item).
5. The editor beside the panel cuts code at ~45 characters.
6. Units still mixed: the matched-row table and the debugger's Variables show raw decimals; scenario values are fractions.
7. L4: reader chips run off the edge; the value box is clipped.
8. L3: "why this rate" takes four expert steps.
9. L5: the knob picker is three cramped boxes.
10. L5: summary cells don't say what the average is over; no currency.
11. L6/L7: no decision column next to reason_code changes.
12. Money formatting inconsistent; "default → 7.5%" in the What-if title while the body says 7.75%.

## Round 4

Changes: each mid-run edit can be compared on its own or all together, and the comparison says both runs went start
to end; the bridge compiles every load from source in a fresh bytecode cache (an edit saved within the same second at
the same length used to run the old code); results list every changed record as a row with decision first; money
shows in rand; empty values drop out of previews; one picker for a scenario knob (parameters and input fields),
values accept percentages; the breakdown gets more room; the editor split is 50/50; the adapter no longer fails on a
custom request without arguments.

Judge's problems, most damaging first:

1. L7: the focused, unchanged record is the only visible result row; the changed ones are below the fold.
2. Compare: rows of "follows from the above" push the results off screen.
3. Declined applicants appear as changed offers.
4. Result tables clip on the right in the narrow panel.
5. Lookup tables: the value column is clipped while band inputs take the space.
6. L7: "No result changes" doesn't say why (the floor never applied).
7. The graph viewport is tiny under a 4-line pause banner.
8. Scenario cells verbose; "on average" of what?
9. A few raw decimals left (a table result, node params).
10. No one-line summary on the rate breakdown.
11. After "Use edited code" nothing shows what changed in the code.
12. Whether the relaxed rule changed any decision is unclear.

## Round 5

Changes: Compare leads with decisions ("No decision changed · offers changed for 3 applicants · 2 declined records
changed values only"), then one card per changed record (approved first, declined grouped), then what changed
(params, inputs, the edited steps; downstream steps folded into one line), then step by step; when nothing changed it
says the edit made no difference; lookup tables put the output column first; the breakdown opens with a one-line
summary; "Use edited code" shows the lines that changed (diffed against the text as it was loaded, since
`inspect.getsource` reads the file as it is now); scenario cells read "3 · −R 5,303.95".

Judge's problems, most damaging first:

1. The pause banner (status, diff, edit pills) plus details crush the graph to 170-280px.
2. "Offers changed for 0 applicants" above an empty Results section looks broken.
3. Only 2 of repo_rate's 8 readers are listed; the other 6 aren't said to have had no effect.
4. The focused record is unaffected, with no route to the affected ones.
5. The editor beside the panel cuts the rule's condition; table columns clip.
6. Sweep cells cryptic; approvals only a footnote.
7. Knob pickers cramped; the "from the pause" option shows when nothing is paused.
8. Decline-reason changes are buried.
9. Step counts differ between the one-edit and all-edits comparisons.
10. "What changed" table: no header on the count, duplicate lines, wrapping headings.
11. Run-changing buttons sit inside the read-only explanation.
12. The find query clears after Enter, losing the other hits.
