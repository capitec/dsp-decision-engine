# Experiment 02 findings — flow graph scale and interaction

**Status:** spike complete. Baselines measured headlessly; budgets proposed for review.
**Reproduce:** `uv run python notes/vscode-redesign/experimentation/02-flow-scale/measure.py` (structure/payload)
and `node notes/vscode-redesign/experimentation/02-flow-scale/layout-time.mjs` (dagre timing, needs `@dagrejs/dagre` installed).

## What the extension uses today

- **Layout library:** `@dagrejs/dagre` (`tools/decider-ui/package.json`, `src/layout.ts`). Layout only.
- **Rendering:** hand-rolled React SVG (`src/Graph.tsx`): one `<svg viewBox=…>` scaled by `width/height` CSS, nodes as `<g>`, edges as `<path>`. No graph UI library (no cytoscape/vis.js/d3).
- **Gestures:** none built. Zoom is a fixed CSS scale; pan is the host's native scrollbar / `scrollIntoView`. There is no pointer-centred zoom, no drag-to-pan, no fit-to-flow or fit-to-selection. The task-05 "confirm supported gestures in the selected library" step is moot: there is no gesture library, so every required gesture is a build, not a configure.
- **Data dependency representation:** already partially progressive. Node boxes always show a `reads …` / `→ …` sub-line; data edges are capped at `MAX_OUT = 4` outgoing per node (`Graph.tsx`), and the rest appear only when a node is selected or a column highlighted.

## Measured baseline

### Real fixture (`example_projects/10-retail-credit-e2e/sonnet/pipeline.py`)

| metric | value |
|---|---|
| call nodes | 94 |
| group nodes | 13 (12 sequence, 1 loop) |
| total nodes | 107 |
| max nesting depth (groups incl. root) | 3 |
| inputs / outputs | 316 / 183 |
| order edges | 94 |
| data edges | 181 |
| full `describe` payload | ~110 KB (IR alone ~103 KB) |

The retail-credit flow is **small** — an order of magnitude below the review's
"thousands of nodes". No project in `example_projects/` reaches the target size,
so the scale ceiling is set by generated dense graphs (below) plus library limits.

Per-node payload is ~1.1 KB in the real fixture (dominated by `table`, `formula`,
`body`, `python` per call node), not the ~0.4 KB of a bare node.

### Generated dense IR (`measure.py`)

| calls | total nodes | group nodes | depth | data edges | IR payload (minimal) |
|---|---|---|---|---|---|
| 1k | 2 341 | 1 341 | 6 | 3 996 | 349 KB |
| 2k | 5 093 | 3 093 | 8 | 7 996 | 771 KB |
| 5k | 13 280 | 8 280 | 9 | 19 996 | 2.0 MB |
| 10k | 26 719 | 16 719 | 10 | 39 996 | 4.1 MB |

Nesting inflates node count well past leaf count (a depth-10 fanout-3 tree holds
more groups than calls). Payload grows ~linearly; with real per-node weight the
5k flow is ~5 MB and 10k ~10 MB.

### Layout time (dagre, `layout-time.mjs`)

Chain plus ~2 forward data edges per node (the real edge density), network-simplex ranker:

| calls | layout time |
|---|---|
| 94 | ~330 ms |
| 500 | ~2.4 s |
| 1 000 | ~5.4 s |
| 2 000 | `RangeError: Maximum call stack size exceeded` (crash) |

Two separate ceilings, both measured:
1. **dagre's ordering pass recurses and stack-overflows around ~2 000 nodes.** The current renderer cannot draw "thousands of nodes" at all; it crashes.
2. **Layout is superlinear** (~5 s at 1 000 nodes) and re-runs synchronously on the webview main thread on every fold/unfold (`useMemo(() => layout(ir), [ir])` in `Graph.tsx`).

`dataEdges` in `layout.ts` is also O(inputs × nodes) with a full `[...seen]` array
copy per input — fine at 181 edges, a GC hotspot at tens of thousands.

### What could not be measured

Interaction latency and webview memory need the real VS Code webview, which this
spike cannot launch headlessly. The budgets below are therefore proposed from
measured structure + measured layout cost + documented SVG/postMessage limits,
not from a live profile. They must be confirmed in-webview before task 05 freezes.

## Decision 1 — fixture set and budgets (proposed for review)

**Fixture set** (kept in `example_projects/` + generated): the retail-credit flow
(94 calls) as the small case; generated dense trees at 1k/2k/5k/10k calls and
depth 6–10 as the scale cases.

**Budgets:**

| axis | target | hard ceiling | rationale |
|---|---|---|---|
| layout time (fold/unfold) | < 150 ms | 500 ms | measured 330 ms at 94 nodes; must stay interactive |
| unfolded (visible) call nodes | ≤ 500 | 1 000 | measured 2.4 s / 5.4 s layout; crash at ~2 000 |
| total call nodes per flow | ≤ 5 000 | 10 000 | beyond this payload + layout both fail |
| describe payload | ≤ 2 MB | 5 MB | webview `postMessage` structured-clone cost |
| visible data edges | ≤ 4/node by default | selection-only beyond | already `MAX_OUT = 4` |

The one lever that makes the ceiling work is **fold-by-default**: a 10k-node flow
is only ever unfolded a group at a time, so layout never sees more than a few
hundred nodes. Thousands of *stored* nodes are fine; thousands of *rendered*
nodes are not, with the current library.

## Decision 2 — graph-library gesture configuration and fallbacks

Keep `@dagrejs/dagre` for layout and the hand-rolled SVG for rendering; add the
gesture layer directly (there is nothing to "configure"). No new dependency —
all gestures are SVG transform / scroll manipulation:

- **Zoom:** `wheel` handler, scale factor ~1.1 per notch, clamped to `[0.1, 4]`, centred on the pointer (keep the point under the cursor fixed by adjusting `scrollTop/Left` and the CSS scale together).
- **Pan:** `pointerdown` on empty canvas + `pointermove` drag; `space`-drag as the accessibility alias.
- **Fit-to-flow:** set the `viewBox` (or scale + scroll) to the laid bounding box.
- **Fit-to-selection:** set it to the selected node's box.
- **Collapse/expand:** anchored on the selected node (Decision 3).

**Fallbacks** (required, because the webview is a browser and some hosts lack wheel/touch):

- no wheel/touch → native scroll + toolbar buttons for `+`/`−`/`fit` (the existing `scrollIntoView` reveal stays as the selection fallback);
- no pointer → keyboard navigation that keeps the focused node centred.

## Decision 3 — viewport/selection preservation algorithm

On any structural change (fold/unfold, re-describe, selection sync) that re-runs layout:

1. Before layout, record the selected node's centre `C_old` in **viewport** (client) coordinates.
2. Lay out; find the selected node's new centre `C_new` in **graph** coordinates.
3. Set `scrollTop/Left` so `C_new * scale` maps back onto `C_old` — the selected node stays exactly where it was, the viewport moves around it.
4. If the selected node disappeared (its group was folded), anchor on its nearest visible ancestor cluster instead and point selection at that cluster.
5. Only when there was no prior anchor, fall back to `scrollIntoView(center)`.

This is O(1) per change and needs no layout diff; it only requires the selected
node's coordinates before and after, both of which the layout already produces.

## Decision 4 — dependency information before selection

Keep the current minimal layer **always on**, and make the dense parts selection-only:

- **Always on:** node `reads … / → …` sub-line; execution-order edges; branch-condition edge labels. These answer "what order" and "what does a step read/write" without touching the dense data edges.
- **Column-per-edge labels: off by default.** Draw data edges as unlabelled arcs; show the column name only on edges incident to the selected node (or a highlighted column). With 20–40k data edges, per-edge labels are unreadable noise — this is the one clear win over "minimal edge-label everywhere".
- **Progressive disclosure (selection-gated):** selected step → its full in/out data edges with column labels; selected edge → the carried value (runtime only); selected value → every read/write/touch point highlighted. This matches task 05's three-tier design and reuses `lineage`/`highlightColumn` already wired in `Graph.tsx`.

## Plan change recommendation

1. **Do not bet task 05's graph on dagre alone.** Budget Decision 1 assumes fold-by-default and ≤1 000 unfolded nodes. Before task 05 freezes, either (a) confirm the ~2 000-node stack-overflow and pick a fold threshold, or (b) swap layout for one without the recursion limit (ELK, or a custom layered pass) if thousands of *unfolded* nodes are a hard requirement.
2. **Spike the `dataEdges` algorithm** — its `[...seen]` copy-per-input is the first thing to fix when the graph gets large, independent of the renderer.
3. **Fold-by-default is a product decision, not just a rendering trick** — it changes what a user sees first; surface it in task 05's "Done when".
