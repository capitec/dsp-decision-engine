// Run: pnpm bench   (from ui/lib)
import { dataEdges, fold, layout, orderEdges } from "../src/layout.ts";
import { callNodes, walk, type IRNodeJson } from "../src/model/protocol.ts";

const FANOUT = 3;

// Nested groups of 3, with every 5th group a branch and every 7th a loop, so the shape has
// the depth and the mix of a real flow. Each call reads the previous output and the one before.
function build(calls: number): IRNodeJson {
  let i = 0;
  let g = 0;
  const call = (path: string): IRNodeJson => {
    const n = i++;
    return {
      path, source: "", file: null, line: null, kind: "call", callKind: "scalar",
      inputs: n ? [`c${n - 1}`, ...(n > 1 ? [`c${n - 2}`] : [])] : [], outputs: [`c${n}`],
      params: {}, python: null, code: "x",
    } as IRNodeJson;
  };
  const group = (path: string, size: number): IRNodeJson => {
    if (size <= 1) return call(path);
    const k = ++g;
    const parts = Math.min(FANOUT, size);
    const kids: IRNodeJson[] = [];
    for (let a = 0, left = size; a < parts; a++) {
      const share = Math.ceil(left / (parts - a));
      kids.push(group(`${path}/${a}`, share));
      left -= share;
    }
    const base = { path, source: "", file: null, line: null };
    if (k % 5 === 0 && kids.length === 3) return { ...base, kind: "branch", modifies: [], children: kids } as IRNodeJson;
    if (k % 7 === 0 && kids.length === 2) return { ...base, kind: "loop", carries: [], maxIterations: 20, children: kids } as IRNodeJson;
    return { ...base, kind: "sequence", children: kids } as IRNodeJson;
  };
  return group("root", calls);
}

const groupsOf = (ir: IRNodeJson) => {
  const out: string[] = [];
  walk(ir, (n) => n.kind !== "call" && out.push(n.path));
  return out;
};
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[xs.length >> 1];
function time(f: () => unknown, runs = 7): number {
  f();
  const ts: number[] = [];
  for (let r = 0; r < runs; r++) {
    const t = performance.now();
    f();
    ts.push(performance.now() - t);
  }
  return median(ts);
}

const rows: string[] = ["| calls | groups | data edges | order edges | layout | fold+layout (laid nodes) | dataEdges | orderEdges | cull |", "|---|---|---|---|---|---|---|---|---|"];
for (const target of [100, 500, 1000, 2000, 5000, 10000]) {
  const ir = build(target);
  let groups = 0;
  walk(ir, (n) => n.kind !== "call" && groups++);
  const calls = callNodes(ir).length;
  try {
    const laid = layout(ir);
    // A typical unfold: groups three levels down are boxes; one of them opens.
    const depth = (p: string) => p.split("/").length;
    const unfoldedPath = groupsOf(ir).find((p) => depth(p) === 4)!;
    const open = (p: string) => depth(p) <= 3 || p === unfoldedPath;
    // Cull to ~200 nodes: the nodes sharing the first one's column nearest the top.
    const sorted = [...laid.nodes].sort((a, b) => a.y - b.y || a.x - b.x);
    const v = { x1: sorted[0].x - 1, x2: sorted[0].x + 800, y1: sorted[0].y - 1, y2: sorted[Math.min(199, sorted.length - 1)].y + 60 };
    const hit = (x: number, y: number, w: number, h: number) => x + w >= v.x1 && x <= v.x2 && y + h >= v.y1 && y <= v.y2;
    const cull = () => {
      laid.nodes.filter((n) => hit(n.x, n.y, n.width, n.height));
      laid.clusters.filter((c) => hit(c.x, c.y, c.width, c.height));
      laid.edges.filter((e) => {
        const xs = e.points.map((p) => p.x), ys = e.points.map((p) => p.y);
        return hit(Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
      });
    };
    const shown = laid.nodes.filter((n) => hit(n.x, n.y, n.width, n.height)).length;
    const f = (x: number) => x.toFixed(1) + " ms";
    rows.push(`| ${calls} | ${groups} | ${dataEdges(ir).length} | ${orderEdges(ir).length} | ${f(time(() => layout(ir)))} | ${f(time(() => layout(fold(ir, open))))} (${layout(fold(ir, open)).nodes.length}) | ${f(time(() => dataEdges(ir)))} | ${f(time(() => orderEdges(ir)))} | ${f(time(cull))} (${shown} nodes) |`);
  } catch (e) {
    rows.push(`| ${calls} | ${groups} | - | - | ${(e as Error).constructor.name}: ${(e as Error).message} | | | | |`);
  }
}
console.log(rows.join("\n"));
