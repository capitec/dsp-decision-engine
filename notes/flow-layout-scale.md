# Flow layout at scale

`layout(ir)` in `ui/lib/src/layout.ts` is a dependency-free layout that replaced dagre. This note times it against the flow-scale budgets set in `notes/vscode-redesign/experimentation/02-flow-scale-and-gesture-findings.md`.

## Rerun

```
cd ui/lib && pnpm bench
```

`bench/layout.ts` builds a nested flow per size (groups of 3; every 5th group a branch, every 7th a loop; each call reads the previous two outputs) and prints a markdown table. Times are the median of 7 runs after one warm-up. Machine: Apple M2 Pro, Node 23.

## What is timed

- **layout**: `layout(ir)`, which includes `orderEdges`.
- **fold+layout**: `fold` with groups three levels down drawn as boxes and one of them opened, then `layout`. This is what a fold/unfold costs in `Graph.tsx`. The folded view lays out only 29 nodes whatever the flow size.
- **dataEdges**, **orderEdges**: the edge passes alone.
- **cull**: the viewport filter over `nodes`, `clusters` and `edges`, with the viewport sized to show about 200 nodes.

## Results

| calls | groups | data edges | order edges | layout | fold+layout | dataEdges | cull (nodes shown) |
|---|---|---|---|---|---|---|---|
| 100 | 59 | 186 | 109 | 0.2 ms | 0.1 ms | 0.1 ms | 0.0 ms (92) |
| 500 | 364 | 914 | 558 | 0.6 ms | 0.1 ms | 0.4 ms | 0.1 ms (192) |
| 1 000 | 635 | 1 876 | 1 100 | 0.8 ms | 0.1 ms | 0.8 ms | 0.2 ms (195) |
| 2 000 | 1 093 | 3 663 | 2 194 | 1.2 ms | 0.1 ms | 2.9 ms | 0.3 ms (182) |
| 5 000 | 3 280 | 9 219 | 5 546 | 3.5 ms | 0.2 ms | 14.5 ms | 0.6 ms (191) |
| 10 000 | 6 719 | 18 662 | 11 124 | 8.8 ms | 0.4 ms | 53.1 ms | 1.2 ms (164) |

Dagre, same machine, `02-flow-scale/layout-time.mjs` (a flat chain with two forward edges per node, one run, no groups, so only indicative): 100 calls 72 ms, 500 calls 462 ms, 1 000 calls 1.2 s, 2 000 and 5 000 calls `RangeError: Maximum call stack size exceeded`. The 02 findings recorded about 330 ms, 2.4 s and 5.4 s at 94, 500 and 1 000 on a slower run.

ELK (`elkjs` 0.10.2, `layered`, direction down, `hierarchyHandling: INCLUDE_CHILDREN`) on the same generated flows: groups as ELK compound nodes, leaves 250x58, the flow's order edges. Installed in a scratch directory, not the repo; one timed run per size after five warm-up layouts. Our column is the custom layout's time in the same process.

| calls | ELK | custom layout | ratio |
|---|---|---|---|
| 100 | 0.05 s | 0.2 ms | ~250x |
| 500 | 0.24 s | 0.6 ms | ~400x |
| 1 000 | 0.46 s | 0.8 ms | ~600x |
| 2 000 | 0.84 s | 1.6 ms | ~530x |
| 5 000 | 2.40 s | 7.5 ms | ~320x |
| 10 000 | 4.65 s | 8.9 ms | ~520x |

ELK never crashes and grows roughly linearly, but it misses the 150 ms target from about 500 calls and the 500 ms limit from about 1 000 calls unfolded. With fold-by-default only the visible nodes are laid out, so it would stay interactive there, and it can run in a Web Worker so the panel doesn't freeze. It does more than the custom layout: crossing minimisation and edge routing for any graph, where the custom layout places nodes from the flow's own tree. Layout quality was not compared.

## Verdict against the 02 budgets

- Layout time (target 150 ms, limit 500 ms): passes at every size, 8.8 ms at 10 000 calls fully unfolded. A fold/unfold is under 1 ms.
- Visible call nodes (target 500, limit 1 000) and total call nodes (target 5 000, limit 10 000): layout is no longer the constraint on either; even 10 000 unfolded calls lay out in under 10 ms. There is no stack overflow: recursion depth follows nesting, not node count.
- The remaining cost at 10 000 calls is `dataEdges` (53 ms), which `Graph.tsx` runs separately from `layout`. It is still inside the 150 ms target.
- Culling is about 1 ms at 10 000 calls, so it is not a cost. The renderer's cost at scale is DOM size, which this script does not measure.

## Surprises

- `dataEdges` is the only superlinear pass: 5 000 to 10 000 calls (2x) costs 3.7x. It scans backwards for each input's last writer and a flow whose columns are rarely rewritten scans far; an index of last writer per column would make it linear (the `ponytail:` comment in `layout.ts` already says so). Not fixed here.
- Layout itself grows roughly linearly (about 1 ms per 1 000 calls).
- The generator is not the real fixture: a real flow with wide branches or many frame steps with unknown (`null`) inputs may differ. Real payloads are also 5 to 10 MB at 5 000 to 10 000 calls (02 findings), so `postMessage` and JSON parse, not layout, are the next ceiling.
