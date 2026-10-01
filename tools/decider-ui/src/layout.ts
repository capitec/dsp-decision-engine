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
      // ponytail: a backward scan is O(n) per input; index outputs by name if edges at 100k+ nodes matter.
      for (let i = seen.length - 1; i >= 0; i--) {
        const w = seen[i];
        if ((w.outputs ?? []).includes(input) && !exclusive(arms.get(w.path)!, arms.get(node.path)!)) {
          edges.push({ from: w.path, to: node.path, column: input });
          break;
        }
      }
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

/** "step 7 of 89 in policy" for a path in a flow of `nodes`. */
export function positionIn(nodes: { path: string }[], path: string): string | null {
  const group = path.slice(0, path.lastIndexOf("/"));
  const siblings = nodes.filter((n) => n.path.slice(0, n.path.lastIndexOf("/")) === group);
  const i = siblings.findIndex((n) => n.path === path);
  return i < 0 || siblings.length < 2 ? null : `step ${i + 1} of ${siblings.length} in ${group.split("/").pop()}`;
}

/** Every group below the root: what "expand all" opens. */
export function groupPaths(ir: IRNodeJson): Set<string> {
  const out = new Set<string>();
  walk(ir, (n) => {
    if (n.kind !== "call" && n !== ir) out.add(n.path);
  });
  return out;
}

/** Every node that reads or writes `column`, in walk order. */
export function touchPoints(ir: IRNodeJson, column: string): CallNodeJson[] {
  return callNodes(ir).filter((n) => (n.inputs ?? []).includes(column) || (n.outputs ?? []).includes(column));
}

/** The node at `path`, or undefined. */
export function findNode(ir: IRNodeJson, path: string): IRNodeJson | undefined {
  let found: IRNodeJson | undefined;
  walk(ir, (n) => {
    if (n.path === path) found ??= n;
  });
  return found;
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

const GAP_X = 30;
const GAP_Y = 40;
const CLUSTER_TOP = 28;
const CLUSTER_SIDE = 12;
const CLUSTER_BOTTOM = 12;

type Size = { w: number; h: number };
const kidsOf = (n: IRNodeJson) => (n.kind === "call" ? [] : n.children);
const column = (s: Size[]): Size => ({ w: Math.max(0, ...s.map((x) => x.w)), h: s.reduce((a, x) => a + x.h, 0) + GAP_Y * Math.max(0, s.length - 1) });
const row = (s: Size[]): Size => ({ w: s.reduce((a, x) => a + x.w, 0) + GAP_X * Math.max(0, s.length - 1), h: Math.max(0, ...s.map((x) => x.h)) });

/**
 * `ir`'s own tree shape is the layout: a sequence stacks its children, a branch puts its
 * condition over a row of arms, a loop puts its condition over its body. No layout library:
 * recursion goes as deep as the nesting, not the node count, so it doesn't share dagre's
 * stack-overflow ceiling around ~2 000 nodes.
 */
export function layout(ir: IRNodeJson): Layout {
  const size = new Map<string, Size>();
  const measure = (n: IRNodeJson): Size => {
    let s: Size = { w: NODE_W, h: NODE_H };
    if (!isLeaf(n)) {
      const kids = kidsOf(n).map(measure);
      const inner = n.kind === "branch" ? column([kids[0], row(kids.slice(1))]) : column(kids);
      s = { w: inner.w + 2 * CLUSTER_SIDE, h: inner.h + CLUSTER_TOP + CLUSTER_BOTTOM };
    }
    size.set(n.path, s);
    return s;
  };
  measure(ir);

  const at = new Map<string, LaidNode>();
  const place = (n: IRNodeJson, x: number, y: number) => {
    const s = size.get(n.path)!;
    at.set(n.path, { path: n.path, kind: n.kind, label: lastSegment(n.path), x, y, width: s.w, height: s.h, node: n });
    if (isLeaf(n)) return;
    const kids = kidsOf(n);
    const cw = s.w - 2 * CLUSTER_SIDE;
    let cy = y + CLUSTER_TOP;
    const stack = kids.slice(0, n.kind === "branch" ? 1 : kids.length);
    for (const k of stack) {
      const ks = size.get(k.path)!;
      place(k, x + CLUSTER_SIDE + (cw - ks.w) / 2, cy);
      cy += ks.h + GAP_Y;
    }
    if (n.kind !== "branch") return;
    const arms = kids.slice(1);
    let cx = x + CLUSTER_SIDE + (cw - row(arms.map((a) => size.get(a.path)!)).w) / 2;
    for (const a of arms) {
      place(a, cx, cy);
      cx += size.get(a.path)!.w + GAP_X;
    }
  };
  place(ir, 0, 0);

  const laidEdges: LaidEdge[] = orderEdges(ir).map((e, i) => {
    const a = at.get(e.from)!;
    const b = at.get(e.to)!;
    const kind = e.back ? ("back" as const) : ("order" as const);
    if (e.back) {
      const x = Math.max(a.x + a.width, b.x + b.width) + 16;
      return { id: `o${i}`, from: e.from, to: e.to, kind, label: e.label, points: [{ x: a.x + a.width, y: a.y + a.height / 2 }, { x, y: a.y + a.height / 2 }, { x, y: b.y + b.height / 2 }, { x: b.x + b.width, y: b.y + b.height / 2 }] };
    }
    const [ax, bx, ay, by] = [a.x + a.width / 2, b.x + b.width / 2, a.y + a.height, b.y];
    const my = (ay + by) / 2;
    return { id: `o${i}`, from: e.from, to: e.to, kind, label: e.label, points: [{ x: ax, y: ay }, { x: ax, y: my }, { x: bx, y: my }, { x: bx, y: by }] };
  });

  const nodes: LaidNode[] = [];
  const clusters: LaidNode[] = [];
  walk(ir, (n) => (isLeaf(n) ? nodes : clusters).push(at.get(n.path)!));
  const root = size.get(ir.path)!;
  return { width: root.w, height: root.h, nodes, clusters, edges: laidEdges };
}
