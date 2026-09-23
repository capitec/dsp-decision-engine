// Shapes the Python bridge sends. Shared by the adapter, the tree and the webview.

export interface IRNodeBase {
  path: string;
  source: string;
  file: string | null;
  line: number | null;
}

export interface CallNodeJson extends IRNodeBase {
  kind: "call";
  inputs: string[];
  outputs: string[];
  params: Record<string, unknown>;
  fills: Record<string, unknown>;
  bodyLine: number | null;
}

export interface GroupNodeJson extends IRNodeBase {
  kind: "sequence" | "branch";
  modifies?: string[];
  children: IRNodeJson[];
}

export type IRNodeJson = CallNodeJson | GroupNodeJson;

export interface DescribeResult {
  pipelines: { name: string; line: number | null; kind: string }[];
  pipeline: string;
  ir: IRNodeJson;
  columns: string[];
}

export interface Checkpoint {
  path: string;
  phase: "start" | "end";
  depth: number;
}

export interface SessionEvent {
  event: string;
  path?: string;
  reason?: string;
  name?: string;
  producer?: string;
  outputs?: ColumnSummary[];
  [k: string]: unknown;
}

export interface Status {
  finished: boolean;
  current: Checkpoint | null;
  events: SessionEvent[];
}

export interface ColumnSummary {
  name: string;
  dtype: string;
  rows: number;
  nulls: number;
  preview: unknown[];
  producer: string;
  versions: number;
}

export interface Lineage {
  name: string;
  producer: string | null;
  inputs: Lineage[];
}

/** Messages between the extension and the graph webview. */
export type ToWebview =
  | { type: "describe"; describe: DescribeResult }
  | { type: "status"; current: Checkpoint | null; finished: boolean; finishedPaths: string[] }
  | { type: "state"; columns: ColumnSummary[] | null };

export type FromWebview = { type: "reveal"; path: string } | { type: "ready" };

export function walk(node: IRNodeJson, fn: (n: IRNodeJson, parent: IRNodeJson | null) => void, parent: IRNodeJson | null = null) {
  fn(node, parent);
  if (node.kind !== "call") for (const c of node.children) walk(c, fn, node);
}

export function callNodes(node: IRNodeJson): CallNodeJson[] {
  const out: CallNodeJson[] = [];
  walk(node, (n) => {
    if (n.kind === "call") out.push(n);
  });
  return out;
}

export function lastSegment(path: string): string {
  return path === "" ? "<root>" : path.slice(path.lastIndexOf("/") + 1);
}
