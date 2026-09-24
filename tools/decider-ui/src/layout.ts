import dagre from "@dagrejs/dagre";
import { callNodes, lastSegment, walk, type CallNodeJson, type IRNodeJson } from "./model/protocol";

export interface LaidNode {
  path: string;
  kind: IRNodeJson["kind"];
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  node: IRNodeJson;
}

export interface LaidEdge {
  id: string;
  from: string;
  to: string;
  kind: "order" | "back";
  label?: string;
  points: { x: number; y: number }[];
}

export interface Layout {
  width: number;
  height: number;
  nodes: LaidNode[];
  clusters: LaidNode[];
  edges: LaidEdge[];
}

const NODE_W = 250;
const NODE_H = 58;

/** Which branch arms a node sits in, so alternatives never feed each other. */
function armKeys(ir: IRNodeJson): Map<string, Map<string, number>> {
  const keys = new Map<string, Map<string, number>>();
  const visit = (n: IRNodeJson, inherited: Map<string, number>) => {
    keys.set(n.path, inherited);
    if (n.kind === "call") return;
    n.children.forEach((c, i) => {
      const own = n.kind === "branch" && i > 0 ? new Map([...inherited, [n.path, i]]) : inherited;
      visit(c, own);
    });
  };
  visit(ir, new Map());
  return keys;
}

function exclusive(a: Map<string, number>, b: Map<string, number>): boolean {
  for (const [branch, arm] of a) {
    const other = b.get(branch);
    if (other !== undefined && other !== arm) return true;
  }
  return false;
}

/** Data-flow edges: every read is fed by the last compatible write before it. */
export function dataEdges(ir: IRNodeJson): { from: string; to: string; column: string }[] {
  const arms = armKeys(ir);
  const seen: CallNodeJson[] = [];
  const edges: { from: string; to: string; column: string }[] = [];
  for (const node of callNodes(ir)) {
    for (const input of node.inputs ?? []) {
      const writer = [...seen].reverse().find((w) => (w.outputs ?? []).includes(input) && !exclusive(arms.get(w.path)!, arms.get(node.path)!));
      if (writer) edges.push({ from: writer.path, to: node.path, column: input });
    }
    seen.push(node);
  }
  return edges;
}

const isLeaf = (n: IRNodeJson) => n.kind === "call" || n.folded !== undefined;

/** `ir` with every group `open` rejects drawn as one box; the root is always open. */
export function fold(ir: IRNodeJson, open: (path: string) => boolean, root = true): IRNodeJson {
  if (ir.kind === "call") return ir;
  if (!root && !open(ir.path)) return { ...ir, children: [], folded: callNodes(ir).map((c) => c.path) };
  return { ...ir, children: ir.children.map((c) => fold(c, open, false)) };
}

function firstLeaf(n: IRNodeJson): string {
  return isLeaf(n) ? n.path : firstLeaf((n as { children: IRNodeJson[] }).children[0]);
}

function lastLeaves(n: IRNodeJson): string[] {
  if (isLeaf(n) || n.kind === "call") return [n.path];
  if (n.kind === "branch") return n.children.slice(1).flatMap(lastLeaves);
  if (n.kind === "loop") return [n.children[0].path]; // the loop exits from its condition
  return lastLeaves(n.children[n.children.length - 1]);
}

/** Control-flow edges: siblings in order; a branch condition into each arm; a loop body back to its condition. */
export function orderEdges(ir: IRNodeJson): { from: string; to: string; label?: string; back?: boolean }[] {
  const edges: { from: string; to: string; label?: string; back?: boolean }[] = [];
  walk(ir, (n) => {
    if (isLeaf(n)) return;
    if (n.kind === "sequence") {
      for (let i = 1; i < n.children.length; i++) {
        for (const from of lastLeaves(n.children[i - 1])) edges.push({ from, to: firstLeaf(n.children[i]) });
      }
    } else if (n.kind === "branch") {
      const cond = n.children[0];
      const out = cond.kind === "call" ? cond.outputs?.[0] ?? "condition" : "condition";
      const arms = n.children.slice(1);
      // A bool condition picks arm 0 when true; an int one picks by index.
      arms.forEach((arm, i) => edges.push({ from: cond.path, to: firstLeaf(arm), label: arms.length === 2 ? `${out} = ${i === 0}` : `${out} = ${i}` }));
    } else if (n.kind === "loop") {
      const [cond, body] = n.children;
      edges.push({ from: cond.path, to: firstLeaf(body), label: "while" });
      for (const from of lastLeaves(body)) edges.push({ from, to: cond.path, label: "repeat", back: true });
    }
  });
  return edges;
}

export function layout(ir: IRNodeJson): Layout {
  const g = new dagre.graphlib.Graph({ compound: true, multigraph: true });
  g.setGraph({ rankdir: "TB", nodesep: 30, ranksep: 40, marginx: 16, marginy: 16 });
  g.setDefaultEdgeLabel(() => ({}));
  walk(ir, (n, parent) => {
    if (isLeaf(n)) g.setNode(n.path, { width: NODE_W, height: NODE_H });
    else g.setNode(n.path, { clusterLabelPos: "top" });
    if (parent) g.setParent(n.path, parent.path);
  });
  const edges: Omit<LaidEdge, "points">[] = orderEdges(ir).map((e, i) => ({ id: `o${i}`, from: e.from, to: e.to, kind: e.back ? ("back" as const) : ("order" as const), label: e.label }));
  for (const e of edges) g.setEdge(e.from, e.to, { width: e.label ? e.label.length * 6 : 0, height: e.label ? 12 : 0 }, e.id);
  dagre.layout(g);

  const nodes: LaidNode[] = [];
  const clusters: LaidNode[] = [];
  walk(ir, (n) => {
    const l = g.node(n.path);
    const laid = { path: n.path, kind: n.kind, label: lastSegment(n.path), x: l.x - l.width / 2, y: l.y - l.height / 2, width: l.width, height: l.height, node: n };
    (isLeaf(n) ? nodes : clusters).push(laid);
  });
  const laidEdges: LaidEdge[] = edges.map((e) => ({ ...e, points: g.edge(e.from, e.to, e.id).points }));
  const { width = 0, height = 0 } = g.graph();
  return { width, height, nodes, clusters, edges: laidEdges };
}
