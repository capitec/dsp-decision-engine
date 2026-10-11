import * as path from "node:path";
import type { FlowRef } from "./flows";

export interface Group {
  root?: string;
  main: FlowRef[];
  sub: FlowRef[];
}

/** The nearest ancestor directory of `file` that holds `configs/`. */
export function projectRoot(file: string, isDir: (p: string) => boolean): string | undefined {
  for (let d = path.dirname(file); ; d = path.dirname(d)) {
    if (isDir(path.join(d, "configs"))) return d;
    if (path.dirname(d) === d) return undefined;
  }
}

/** Flows per project (main flows and subflows), projects by path, flows outside any project last. */
export function groupFlows(flows: FlowRef[], rootOf: (file: string) => string | undefined): Group[] {
  const groups = new Map<string | undefined, Group>();
  const buildIn = new Set(flows.filter((f) => f.pipeline === "build" && path.basename(f.file) === "pipeline.py").map((f) => f.file));
  for (const f of flows) {
    const root = rootOf(f.file);
    const g = groups.get(root) ?? groups.set(root, { root, main: [], sub: [] }).get(root)!;
    const main =
      root !== undefined &&
      (path.basename(f.file) === "pipeline.py" && path.dirname(f.file) === root
        ? f.pipeline === "build" || !buildIn.has(f.file)
        : !f.pipeline && path.dirname(f.file) === root);
    (main ? g.main : g.sub).push(f);
  }
  return [...groups.values()].sort((a, b) => (a.root === undefined ? 1 : b.root === undefined ? -1 : a.root.localeCompare(b.root)));
}
