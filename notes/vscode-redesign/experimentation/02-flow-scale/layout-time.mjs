// Experiment 02 layout-time harness: dagre (the layout.ts library) on a chain
// plus forward data edges, matching the webview's real edge density.
// Run: node notes/vscode-redesign/experimentation/02-flow-scale/layout-time.mjs
// Requires @dagrejs/dagre (install in any scratch dir, e.g. tools/decider-ui).
import dagre from "@dagrejs/dagre";

function build(calls, extra) {
  const g = new dagre.graphlib.Graph({ multigraph: true });
  g.setGraph({ rankdir: "TB", nodesep: 30, ranksep: 40, marginx: 16, marginy: 16 });
  g.setDefaultEdgeLabel(() => ({}));
  for (let i = 0; i < calls; i++) g.setNode(`n${i}`, { width: 250, height: 58 });
  let eid = 0;
  for (let i = 0; i < calls; i++) {
    if (i + 1 < calls) g.setEdge(`n${i}`, `n${i + 1}`, { width: 0, height: 0 }, `e${eid++}`);
    for (const k of [2, 3]) if (i + k < calls) g.setEdge(`n${i}`, `n${i + k}`, { width: 0, height: 0 }, `e${eid++}`);
  }
  return g;
}

function time(calls) {
  const g = build(calls, 2);
  const t0 = process.hrtime.bigint();
  dagre.layout(g);
  return Number(process.hrtime.bigint() - t0) / 1e6;
}

time(20); // warm the JIT

for (const c of [94, 500, 1000, 2000]) {
  try {
    console.log(`${c} calls: ${time(c).toFixed(0)}ms`);
  } catch (e) {
    console.log(`${c} calls: ${e.constructor.name}: ${e.message}`);
  }
}
