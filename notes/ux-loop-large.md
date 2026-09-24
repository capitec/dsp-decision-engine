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
| 6 | 7.3 | 8 | 8 | 8 | 7 | 6 | 7 | 7 |
| 7 | 7.4 | 8 | 8 | 8 | 7 | 6 | 8 | 7 |
| 8 | 7.4 | 8 | 8 | 7 | 8 | 6 | 8 | 7 |
| 9 | 7.4 | 8 | 8 | 7 | 8 | 7 | 8 | 6 |
| 10 | 7.6 | 8 | 8 | 7 | 8 | 7 | 8 | 7 |
| 11 | 7.8 | 8 | 9 | 7 | 8.5 | 6.5 | 8 | 7.5 |
| 12 | 7.6 | 8 | 8 | 8 | 7 | 7 | 8 | 7 |
| 13 | 7.7 | 8 | 8 | 8 | 8 | 7 | 8 | 7 |
| 14 | 7.6 | 8 | 8 | 7 | 8 | 7 | 8 | 7 |
| 15 | 7.7 | 8 | 8 | 7 | 8 | 7 | 8 | 8 |
| 16 | 7.8 | 8 | 8.5 | 7 | 8 | 7 | 8.5 | 7.5 |
| 17 | 7.8 | 8 | 8 | 7.5 | 8.5 | 6.5 | 8.5 | 7.5 |
| 18 | 7.7 | 8 | 8 | 7 | 8 | 7 | 8 | 8 |

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

## Round 6

Changes: the pause banner is one line ("2 edits ▾" and the compare button), with the note and code diff only right
after an edit; run-changing buttons moved into "Change the run ▾"; the find box keeps its hits ("1 of 4 ◀ ▶");
record names in results focus that record; every reader of a changed param is listed, with 0 when nothing moved;
"N declines now for a different reason" in the headline; the step's formula shows in its details; scenario knobs
read "pl_product_cap (cap) in personal_loan/limits"; outcome counts are sweep columns.

The score dipped (7.6 to 7.3) with a fresh judge; several points contradict round 5 (declined records expanded or
collapsed). One is a real bug: after "Use edited code" the details still show the old formula.

Judge's problems, most damaging first:

1. L7: details say "running your edited code" above the old formula.
2. L5: the sweep table clips offer_amount; no horizontal scroll.
3. L5: cells need a legend; no book-level total.
4. L6: "2 declines now for a different reason" never names them or the reasons.
5. L4: no reason given for why a repo move changes no approved offer.
6. Declined records' value changes clutter results next to "0 offers changed".
7. L7: one-edit and all-edit comparisons look the same; no scope chip.
8. The details pane takes half the panel; the graph is small.
9. L3: the breakdown opens scrolled to the middle.
10. L3: the cap and floor values and the headroom are never computed.
11. L5: picker narrow; the chosen param's current value not shown; values not echoed.
12. What-if tables clip their last column; filter placeholder cut off.

## Round 7

Changes: after "Use edited code" the details show the new formula, marked edited, with the old one struck through;
the breakdown computes a cap or floor from its formula (a small arithmetic evaluator, since the webview's CSP rules
out eval) and says "cap 28.75%, not reached (3.55 pp below)"; it opens at its heading; a changed decline reason gets
its own result card and declined-only changes collapse; edit comparisons say which edits they apply; the sweep scrolls
sideways and drops never-changing columns; knob pickers show the param's current value and echo the parsed values;
the adapter's launch options moved to their own file (it was over 500 lines).

Judge's problems, most damaging first:

1. Offer counts contradict: "offers changed for 0 applicants" next to "offer_amount: 1 of 40 records changed".
2. Declined records show offer fields in bold as if offered.
3. Sweep cells wrap to three lines; no up/down colour; no approve/decline column.
4. The knob's full path overlaps the values input.
5. "41 steps changed" counts readers that changed no record.
6. No reason given for why no approved offer moved.
7. The banner still carries the diff that the details repeat.
8. Step details crush the graph; no resizable split.
9. The breakdown weighs zero parts like real ones; the summary is one run-on line.
10. After running the edited code the focused record shows no before/after.
11. Lookup table columns clip; column order differs from the matched-row view.
12. Small labels: a git button inside the step list, a wrapping badge, "reads pl_base_rates" for the table itself.

## Round 8

Changes: declined records show only their decision and reason, other values as "internal values, not offered";
readers of a changed param that moved nothing collapse into one line; when no approved offer moved, one line says
where the change went; a table's own edit reads "row 3 edited"; the git action sits beside the title; sweep cells
read "−R 5,303.95 (3)", green down and red up, with decision counts as a column; a draggable split between the graph
and the details, remembered per viewer; the breakdown folds zero parts and splits its summary into lines; lookup
tables put the band first and fit the panel; the banner no longer repeats the code diff.

The score held at 7.4. The remaining problems are mostly about space: a ~500px panel beside the editor.

Judge's problems, most damaging first:

1. L5: the sweep scrolls sideways; offer columns hidden while half the panel is empty; the current setting unmarked.
2. L3: the summary drops the zero terms, so "risk loading" is missing from the answer.
3. The panel is cramped; the editor clips the edited line; What-if, Scenarios and Compare don't need the editor.
4. L7: the focused record's new value is below the fold.
5. The graph viewport is small: banner, two toolbar rows and details.
6. L5: the knob picker is squeezed and its path overlaps the values box.
7. L4/L7: "only declined records moved" is said four times.
8. L6: "What changed" lists flow paths, not the changed files.
9. L3: the step name scrolls away above the breakdown.
10. The banner's compare button looks like text; edit chips hidden.
11. What-if opens with the shared tables expanded; the sticky footer covers a title; headers truncated.
12. L1: no sense of position among 1,024 steps; params cut off.

## Round 9

Changes: the knob picker is a two-column grid (an older four-column rule was squeezing it); the sweep summary stacks
each scenario's results in one cell so nothing scrolls sideways, and tags the current setting; the panel widens to
70% on What-if, Scenarios and Compare; the view switches moved into a "View" popover; the step's name, actions,
written value and params are pinned at the top of its details; "only declined moved" is said once; "What changed"
has a file column; the edits sit inline in the banner; What-if opens with every group folded; the breadcrumb says
"step 11 of 13 in pricing".

Judge's problems, most damaging first:

1. L7: "42 steps changed" but Step by step lists one: the focused record filter hides steps that didn't change it.
2. L5: green/red by sign has no clear meaning (a smaller offer is green).
3. L5: a 3 × 2 sweep is six stacked rows; wants a grid.
4. L3: the answer lands at the pane's bottom edge.
5. L3: the summary is machine-like; wants a labelled waterfall; "* 1" is noise.
6. L7: the banner grows to four lines.
7. The panel width jumps between tabs (the round 8 judge asked for wider tabs; this one dislikes the jump).
8. L7: the edit isn't visible in the editor.
9. L6: a client id is cut off in Step by step.
10. The file column says "PARAMS", not the file.
11. Changed records truncated at 3 with "(and 1 more)".
12. L4: the floor is tagged "changed" though it only passed its input through.

## Round 10

Changes: the step list no longer hides steps when the focused record is unchanged (the round 9 contradiction); the
panel keeps one width on every tab; the rate breakdown shows a sum as a waterfall (each part, its value, and for a
lookup table the row that matched), with the summary in words; "* 1" drops out of filled formulas; a two-knob sweep
is a grid, one metric at a time, with ▲▼ instead of red/green; the banner's edits sit behind one chip; "and N more"
in the step list expands.

Judge's problems, most damaging first:

1. L7: the edits menu overflows the panel sideways and stays open over the report.
2. The graph shows 3-4 nodes; the editor beside it holds an irrelevant file.
3. L3: the breakdown is cramped; its +/− read as expand toggles.
4. L3: the matched row is below the fold; the find box keeps a stale query.
5. L7: Skip is hidden in "Change the run".
6. L5: nothing says the cap made no difference; the hint's example doesn't match the metric.
7. L4: the floor is badged "changed" though it only passed its input through.
8. L7: a step about to re-run still shows its ✓; the edited cap's result doesn't say whether it bound.
9. Compare headers: which side is the baseline; process chatter above the answer.
10. L1: the opened rule is cut off in the narrow editor.
11. L5: knob hints out of order; "(cap)" jargon; money unformatted.
12. L6: "PARAMS" instead of params.json; duplicate "in its code"; "working tree" means little to an analyst.

## Round 11

Changes: the edits list opens inline under the banner and closes when a comparison is picked; the panel opens at 65%
of the width and the graph gets more height; the waterfall's signs are part of its numbers; the find box clears when
the selection moves elsewhere and a table's matched row scrolls into view; paused at a step, Skip and "Use edited
code" are plain buttons; steps that will re-run after an edit lose their ✓; a step that only reads a changed param
is badged "reads repo_rate"; an edited cap says whether it bound; the Compare header names the baseline; knob rows
read name, values, remove with one hint line; the file column names params.json; a knob that changes nothing says so.

Judge's problems, most damaging first:

1. L5: one metric at a time; approvals a footnote; offer_amount absent.
2. L3: the explanation opens scrolled with its heading cut off; the graph takes space the answer needs.
3. L3: the matched row sits at the bottom edge.
4. L1: the opened rule is cut at ~35 characters in the editor.
5. L7: "edited" uses the same orange as "changed"; the compare legend lingers.
6. L7: per-edit compare hidden in the dropdown.
7. The debug toolbar covers the panel title; the detail pane is cut mid-line.
8. L6: a lone param line above the table reads as the only change.
9. Wording: "offers changed for 0 applicants" reads like a bug.
10. Node labels truncate while the graph has empty bands.
11. L5: axis notation and "(3 rec.)" read as notation, not language.
12. L2: an offer amount rises after a rate cut with no "why".

## Round 12

Changes: each sweep grid cell shows every result in words ("▼ −0.53 pp on 3 applicants") with named axes; the
explanation opens at its top with the graph shrunk; a lookup table's matched row is a card above the table; short
steps show their code in the details; mid-run edits are purple with ✎, apart from a comparison's orange; one or two
edits show as chips in the banner; "What changed" is one table with a files summary; the headline says "no offer
changed" and "declined applicants had internal values change"; wider graph nodes; "why?" beside a changed result.

Judge's problems, most damaging first:

1. L4: "What changed" credits pricing.py for a shared param edit and never lists the edit itself.
2. L5: approvals only in a footnote.
3. L5: every changed cell the same fill; which applicants isn't said.
4. The Graph tab stacks too much; the graph shrinks to a strip.
5. L3: the table's match sentence hides under the sticky button row.
6. L1: the opened rule is cut off in the narrow editor.
7. L7: what was edited isn't visible under the edited step's name.
8. L7: the banner grows to three lines.
9. L7: "Compare with the flow as started" vs "Compare all edits": two names for one action.
10. L3: the breakdown repeats cap/floor; zero parts take rows.
11. Node text cut off; unexplained lane lines.
12. Units: no "months" on term columns; cents on whole rand amounts.

## Round 13

Changes: "What changed" leads with each param edit (old → new, where it lives, how many steps read it, records
changed) with the steps that moved because of it indented under it; code changes follow; the sweep states approvals
up front ("Approvals unchanged in all 6 scenarios") and tints cells by which way offer_amount moved; a table's match
sentence scrolls clear of the sticky header; an edited step shows its new formula under its name; one name for the
compare action ("Compare with start"); whole rand amounts drop the cents.

Judge's problems, most damaging first:

1. L5: a cell with no offer_amount change still shows the changed fill, read as "down".
2. L7: the banner grows to three lines with the chips; the toolbar wraps.
3. L7: after running the edited step, its result for the focused record isn't on screen.
4. L3: the graph shrinks to a strip; details scroll under the sticky header.
5. L1: nothing says which of the four matching rules fired.
6. L1: no sense of position beyond the breadcrumb.
7. L1: the opened line is highlighted too faintly in a narrow editor.
8. The first view: spread-out groups, an empty details pane.
9. L6: in compared mode the change block is below the fold.
10. Headlines lead with a big "steps changed" count; declined counts don't add up.
11. L5: tiny axis labels; 22.855% precision.
12. L5: the two knob pickers render differently.

## Round 14

Changes: a grid cell is shaded only when the legend's metric moved; axis labels in normal text; percentages to two
decimals; a step's result for the focused record and a comparison's change block come first in its details; scrolled
content stops below the sticky header; no empty details pane before a step is picked; the step count moved from the
headline to Step by step, and declines with a new reason aren't counted again as "internal values only"; the opened
source line flashes; the banner is one line with the edits behind one control.

Scores have sat between 7.4 and 7.8 for eight rounds. Successive judges now ask for opposite things (edit chips inline
vs. behind one control; zero parts shown vs. folded), so part of the remaining gap is judge variance.

Judge's problems, most damaging first:

1. L3: the matched row is cut off; nothing scrolls to it.
2. L3: the summary hides which parts are 0.
3. L7: the per-edit compare is hidden behind "2 edits ▾".
4. L7: it isn't said that both versions ran to the end.
5. The banner, find bar and compare legend squeeze the graph.
6. First open: blank space below the graph.
7. L5: which applicants each cell means; shading not labelled.
8. L5: axis headers ambiguous; current values not marked on the axes.
9. Results looks empty when no offer changed.
10. L6: the declined card's internal values are tiny grey text.
11. Step actions differ by step type; the find box empties after a link.
12. L1: the opened source is cramped.

## Round 15

Changes: a lookup table step scrolls to its matched row once the pane has laid out (the pane's reset-to-top was
undoing it), and it gets the taller details layout; the breakdown's summary names the parts that are 0; up to three
edits show as chips with their own compare; the comparison says both versions ran to the end; the comparison legend
shrank to a "changes" toggle (the swatches moved into View); the graph fills the panel when no step is picked; sweep
axes are named over their values with the current values marked; Results says "No offer changed" and opens the
declined list; a declined card's internal values are normal lines.

Judge's problems, most damaging first:

1. L3: explaining a rate still takes a debug run, a known step, a focus pick and a click.
2. While paused the graph gets 190-400px: focus picker, banner, find bar, breadcrumb, details.
3. "Open source" lands in a ~35-character editor.
4. A skipped step can't be restored. (The engine's edit API has replace and delete, no insert.)
5. L5: cells average over unnamed applicants.
6. L5: grid typography: small labels, misaligned headers, a fill that reads as an error.
7. Counts clash: "offers changed for 1" beside "records changed 4"; "2 other declined" with no first group.
8. L4: no headroom per step for approved applicants.
9. L7: the edited node's border turns dotted after it runs, like a skipped one.
10. L3: the matched-row card drops the incl./excl. band labels.
11. L6: 49 changed steps paged one by one, not grouped by cause.
12. L7: the combined compare doesn't say the skip contributed nothing.

## Round 16

Changes: while paused, the record picker sits in the pause banner (a row fewer) with an "explain a value…" picker
beside it that opens a value's breakdown on the step that wrote it; one fixed shape for the counts ("Decisions
changed: 0 · Offers changed: 1 · …"); an edited step keeps its solid purple border after running; the matched-row card
uses the band labels; sweep cells show direction by a coloured edge; the details pane scrolls to an open breakdown
and folds the step's code while it is open.

Judge's problems, most damaging first:

1. L3: the rate explanation is a bold paragraph, not a breakdown table.
2. L3: the explain picker forgets the choice; no heading says what is being explained.
3. While paused the graph is a strip; no overview of where you are in 1,024 steps.
4. Code opens in a ~300px editor.
5. L7: the banner wraps over three rows.
6. L7: skips and code swaps can't be undone.
7. L5: averages over unnamed applicants; the colour cue barely visible.
8. L5: the two knob controls look different.
9. "Declined, internal values only" is jargon; the declined group opens inconsistently.
10. L1: the four find hits look identical but for grey path text.
11. Node labels spill out of their boxes.
12. Group boundaries nearly invisible on the graph.

## Round 17

Changes: the rate breakdown opens with a summary table (each part with its matched table row, the zeros folded into
one row, the raw total, each cap and floor with its headroom, the final value) under "Why pl_rate = 25.2% for
client_id 20400 · last written by pl_rate_floor"; the explain picker keeps its choice and fits the banner; a swapped
step can be put back ("Use original code", a new bridge `restore`); plainer names ("Still declined, internal values
changed"), that group folded; find hits carry their product as a tag; both knob pickers look the same.

Judge's problems, most damaging first:

1. L5: the sweep grid's offer figures include declined applicants (a correctness problem, not just layout).
2. The editor beside the panel is ~300px; opened code is unreadable.
3. L3: the explanation starts below the fold and is cut off on the right.
4. The graph is ~190px tall while explaining.
5. L7: a skipped step can't be restored.
6. L7: "nothing changed" with no reason.
7. Explain needs a debug pause.
8. Table changes say "row 3", not the term band.
9. Step by step is mostly declined records' internal values.
10. Dotted data edges unexplained.
11. No book-level totals per scenario.
12. The banner wraps with edit chips.

## Round 18

Changes: sweep cells count only applicants with an offer on either side ("no offer changed (3 declined only)"), and
shade by those; a skipped step can be restored (the bridge swaps its enclosing flow back in, rebuilt from the flow
as started with the other edits kept, since the engine has replace but no insert); the value being explained is
the first card in the details; a table row change names its band ("row 3 (min_term 49, max_term 85)"); steps that
only moved declined applicants' internal values fold behind "show them".

Scores have held at 7.6-7.8 for eleven rounds. The one complaint every judge repeats is space: a ~350px editor
beside a ~650px panel, so code lines are cut and tables clip.

Judge's problems, most damaging first:

1. L5: "no offer changed (3 declined only)" repeated in every metric line; the grid clips.
2. L3: explaining pl_rate moves the selection to its writer.
3. L3: risk loading is folded into "5 other parts".
4. L3: no way back to the breakdown after following a link.
5. The debug layout squeezes the panel and the code.
6. L1: "Open source" lands in a narrow editor, on the decorator.
7. Unlabelled data curves and group bands.
8. No overview of position among 1,000 steps.
9. L7: step counts disagree between compares; "the other 1 is left out" reads like a bug.
10. L7: restore not visible after skipping.
11. L2: which rows the applications hit isn't shown before running.
12. The summary line is a dense run-on.
