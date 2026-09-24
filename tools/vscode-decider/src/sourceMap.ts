import * as path from "node:path";
import type { IRNodeJson } from "@decider/ui";

/**
 * The node a source breakpoint on `line` of `file` means: the nearest node
 * above it, counting both where a step is defined and the Python it runs
 * (so a line inside a tree's reference walker maps to the tree).
 */
export function nodeAtLine(nodes: Iterable<IRNodeJson>, file: string, line: number): { path: string; line: number } | undefined {
  let best: { path: string; line: number } | undefined;
  const consider = (at: number, p: string) => {
    if (at <= line && (!best || at > best.line)) best = { path: p, line: at };
  };
  for (const n of nodes) {
    if (n.file && n.line && samePath(n.file, file)) consider(n.line, n.path);
    if (n.kind === "call" && n.python && samePath(n.python.file, file)) consider(n.python.line, n.path);
  }
  return best;
}

function samePath(a: string, b: string): boolean {
  return path.resolve(a) === path.resolve(b);
}
