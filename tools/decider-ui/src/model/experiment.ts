// The experiment result as `decider.experiments.run` serialises it, and the
// pure helpers the experiments view renders with. Kept JSON-shaped on purpose:
// the wire form is the contract, and these helpers are what the UI unit-tests.

export type ExperimentRunStatus =
  | "completed"
  | "non_reproducible"
  | "partial"
  | "failed"
  | "cancelled"
  | "timed_out";

export type ScenarioStatus = "completed" | "failed" | "resumed" | "skipped";

export interface Finding {
  kind: string;
  scenario?: string;
  location?: string;
  expected?: unknown;
  actual?: unknown;
  message?: string;
}

export interface ScenarioResult {
  name: string;
  status: ScenarioStatus;
  error?: string | null;
  divergences: Finding[];
  output_ref?: { name: string; location: string; row_count?: number | null } | null;
  row_count?: number | null;
  revision?: { authored: string; resolved?: string | null } | null;
  params?: Record<string, unknown>;
  overrides?: Record<string, unknown>;
  row?: number | null;
}

export interface ExperimentScenarioSummary {
  name: string;
  status: ScenarioStatus;
  error?: string | null;
  changed: Record<string, number[]>;
  unchanged: Record<string, number>;
  unique: Record<string, number>;
  path_counts: Record<string, Record<string, number>>;
  step_differences: string[];
  drill_down: Record<string, { row: number; expected: unknown; actual: unknown }[]>;
}

export interface ExperimentSummary {
  outputs: string[];
  first_divergence: { scenario: string; location: string } | null;
  scenarios: ExperimentScenarioSummary[];
}

export interface SankeyNode {
  id: string;
  label: string;
  kind: "input" | "arm" | "outcome";
}

export interface SankeyLink {
  source: string;
  target: string;
  value: number;
}

export interface Sankey {
  outcome: string;
  nodes: SankeyNode[];
  links: SankeyLink[];
  paths?: Record<string, Record<string, number>>;
}

export interface ExperimentResult {
  name?: string;
  status: ExperimentRunStatus;
  nondeterministic?: boolean;
  manifest?: { revision?: { authored?: string; resolved?: string | null } | null };
  scenarios: ScenarioResult[];
  summary?: ExperimentSummary;
  sankey?: Sankey | null;
  job?: { message?: string; error?: string | null; done?: number; total?: number | null } | null;
}

/** One line that says what the run is, never letting a non-reproducible or partial run read as done. */
export function experimentHeadline(r: ExperimentResult): string {
  const scenarioCount = (r.scenarios ?? []).length;
  const divergences = (r.scenarios ?? []).reduce((t, s) => t + (s.divergences?.length ?? 0), 0);
  const bits = [`${scenarioCount} scenario${scenarioCount === 1 ? "" : "s"}`];
  if (divergences) bits.push(`${divergences} divergence${divergences === 1 ? "" : "s"}`);
  if (r.status === "non_reproducible") bits.push("non-reproducible: a baseline rerun diverged");
  if (r.status === "partial") bits.push("partial: some scenarios did not finish");
  if (r.status === "cancelled") bits.push("cancelled");
  if (r.status === "timed_out") bits.push("timed out");
  return bits.join(" · ");
}

/** The status as the badge reads: the three non-completed states are named, never "done". */
export function statusLabel(status: ExperimentRunStatus): string {
  return {
    completed: "completed",
    non_reproducible: "non-reproducible",
    partial: "partial",
    failed: "failed",
    cancelled: "cancelled",
    timed_out: "timed out",
  }[status] ?? status;
}

/** Whether the run counts as a deterministic, finished comparison. */
export function isDeterministicComplete(r: ExperimentResult): boolean {
  return r.status === "completed" && !r.nondeterministic;
}

/** A scenario is a candidate for the debug hand-off only when it actually ran. */
export function isRunnableScenario(s: ScenarioResult): boolean {
  return s.status === "completed" || s.status === "failed";
}

/** A flat list of nodes at `depth` (shortest path from the source), for a layered Sankey layout. */
export function sankeyDepths(s: Sankey): Map<string, number> {
  const depth = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const l of s.links) {
    if (!adj.has(l.source)) adj.set(l.source, []);
    adj.get(l.source)!.push(l.target);
    if (!adj.has(l.target)) adj.set(l.target, []);
  }
  const source = s.nodes.find((n) => n.kind === "input")?.id ?? s.links[0]?.source ?? "";
  const queue = [source];
  depth.set(source, 0);
  while (queue.length) {
    const id = queue.shift()!;
    for (const next of adj.get(id) ?? []) {
      if (!depth.has(next)) {
        depth.set(next, depth.get(id)! + 1);
        queue.push(next);
      }
    }
  }
  for (const n of s.nodes) if (!depth.has(n.id)) depth.set(n.id, 0);
  return depth;
}

export interface SankeyLayout {
  nodes: { id: string; x: number; y: number; w: number; h: number; label: string; kind: SankeyNode["kind"] }[];
  links: { source: string; target: string; value: number; x1: number; y1: number; x2: number; y2: number }[];
  width: number;
  height: number;
}

const NODE_W = 120;
const NODE_H = 26;
const GAP = 18;
const MARGIN = 10;

/** Lay out a Sankey in columns by depth, one node per row, sized by its flow. */
export function sankeyLayout(s: Sankey): SankeyLayout {
  const depth = sankeyDepths(s);
  const maxDepth = Math.max(0, ...depth.values());
  const total = s.links.reduce((t, l) => t + l.value, 0);
  const byDepth = new Map<number, SankeyNode[]>();
  for (const n of s.nodes) {
    const d = depth.get(n.id) ?? 0;
    if (!byDepth.has(d)) byDepth.set(d, []);
    byDepth.get(d)!.push(n);
  }
  const height = Math.max(200, (Math.max(...[...byDepth.values()].map((v) => v.length), 1)) * (NODE_H + GAP) + MARGIN * 2);
  const width = (maxDepth + 1) * (NODE_W + GAP) + MARGIN;
  const flow = (id: string) => s.links.filter((l) => l.target === id).reduce((t, l) => t + l.value, 0) || total;
  const nodes = [] as SankeyLayout["nodes"];
  const pos = new Map<string, { x: number; y: number; w: number; h: number }>();
  for (const [d, list] of byDepth) {
    const x = MARGIN + d * (NODE_W + GAP);
    const colH = list.length * (NODE_H + GAP) - GAP;
    let y = (height - colH) / 2;
    for (const n of list) {
      const h = Math.max(NODE_H, (flow(n.id) / Math.max(total, 1)) * height * 0.6);
      pos.set(n.id, { x, y, w: NODE_W, h });
      nodes.push({ id: n.id, x, y, w: NODE_W, h, label: n.label, kind: n.kind });
      y += h + GAP;
    }
  }
  const links = s.links.map((l) => {
    const a = pos.get(l.source);
    const b = pos.get(l.target);
    return { ...l, x1: a ? a.x + a.w : 0, y1: a ? a.y + a.h / 2 : 0, x2: b ? b.x : 0, y2: b ? b.y + b.h / 2 : 0 };
  });
  return { nodes, links, width, height };
}
