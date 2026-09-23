// Shapes the Python bridge sends. Shared by the adapter, the tree and the webview.

export interface IRNodeBase {
  path: string;
  source: string;
  file: string | null;
  line: number | null;
}

export interface CallNodeJson extends IRNodeBase {
  kind: "call";
  callKind: "scalar" | "row" | "frame";
  /** `null`: a frame step of unknown lineage. */
  inputs: string[] | null;
  outputs: string[] | null;
  params: Record<string, unknown>;
  /** The Python that runs in interpreted mode: `fn`, or a row node's `reference`. */
  python: { file: string; line: number; bodyLine: number | null } | null;
}

export interface GroupNodeJson extends IRNodeBase {
  kind: "sequence" | "branch" | "loop";
  modifies?: string[];
  carries?: string[];
  maxIterations?: number;
  children: IRNodeJson[];
}

export type IRNodeJson = CallNodeJson | GroupNodeJson;

export interface DescribeResult {
  pipelines: { name: string; line: number | null; kind: string }[];
  pipeline: string;
  ir: IRNodeJson;
}

export interface Checkpoint {
  path: string;
  when: "before" | "after";
}

export interface Summary {
  dtype: string;
  rows: number;
  nulls: number;
  preview: unknown[];
}

export interface SessionEvent {
  kind: string;
  origin?: { path: string; source: string; locator: string | null };
  [k: string]: unknown;
}

export interface Status {
  finished: boolean;
  current: Checkpoint | null;
  events: SessionEvent[];
  error?: string;
}

export interface ColumnSummary extends Summary {
  name: string;
  /** The focused record's value, when a record is focused. */
  value: unknown;
  producer: string;
  versions: number;
}

export interface Lineage {
  name: string;
  producer: string | null;
  value: unknown;
  via?: "merge" | "carry";
  inputs: Lineage[];
}

/** Tree positions a row node reached in its latest run: locator -> rows. */
export type Visits = Record<string, Record<string, number>>;

export interface RunStatus {
  current: Checkpoint | null;
  finished: boolean;
  finishedPaths: string[];
  visits: Visits;
  record: number | null;
}

/** Messages between the extension and the graph webview. */
export type ToWebview =
  | { type: "describe"; describe: DescribeResult }
  | ({ type: "status" } & RunStatus)
  | { type: "state"; columns: ColumnSummary[] | null; rows: number }
  | { type: "lineage"; lineage: Lineage | null };

export type FromWebview =
  | { type: "ready" }
  | { type: "reveal"; path: string }
  | { type: "record"; row: number | null }
  | { type: "lineage"; name: string };

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

export function kindLabel(n: IRNodeJson): string {
  return n.kind === "call" ? n.callKind : n.kind;
}

export function previewOf(c: Summary): string {
  const shown = c.preview.map((v) => JSON.stringify(v)).join(", ");
  return `[${shown}${c.rows > c.preview.length ? ", …" : ""}]${c.nulls ? ` ${c.nulls} null` : ""}`;
}
