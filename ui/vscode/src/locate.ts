import * as path from "node:path";
import { callNodes, type CallNodeJson, type IRNodeJson } from "@decikit/ui";

/** Evaluated in a paused Python frame: the flow node running on that thread, or `None`. */
export const NODE_PATH = "__import__('decikit._tooling.bridge.run', fromlist=['node_path']).node_path()";

export interface SourceLine {
  file: string;
  /** 1-based. */
  line: number;
  condition?: string;
}

/** The steps whose Python contains `at`: more than one when a function is reused. */
export function callsAt(ir: IRNodeJson, at: SourceLine): CallNodeJson[] {
  return callNodes(ir).filter((n) => {
    const py = n.python;
    return py && samePath(py.file, at.file) && py.line <= at.line && at.line <= (py.endLine ?? py.bodyLine ?? py.line);
  });
}

/** A breakpoint condition that stops in a shared function only when it runs for `nodePath`. */
export function onlyAt(nodePath: string): string {
  return `${NODE_PATH} == ${JSON.stringify(nodePath)}`;
}

/** The node an `onlyAt` condition names. */
export function onlyAtPath(condition: string | undefined): string | undefined {
  if (!condition?.startsWith(NODE_PATH)) return undefined;
  const m = /==\s*("(?:[^"\\]|\\.)*")\s*$/.exec(condition);
  return m ? JSON.parse(m[1]) : undefined;
}

/** The nodes the breakpoints stop at: every step whose Python holds one, narrowed by an `onlyAt` condition. */
export function breakpointPaths(ir: IRNodeJson, bps: SourceLine[]): Set<string> {
  const out = new Set<string>();
  for (const bp of bps) {
    const only = onlyAtPath(bp.condition);
    for (const n of callsAt(ir, bp)) if (only === undefined || only === n.path) out.add(n.path);
  }
  return out;
}

/** The node a paused stack is in: the innermost frame inside a step's Python, when it is not shared. */
export function pausedAt(ir: IRNodeJson, frames: SourceLine[]): string | undefined {
  for (const f of frames) {
    const hits = callsAt(ir, f);
    if (hits.length) return hits[0].path;
  }
  return undefined;
}

/** The Python `repr` of a str, as an evaluate reply shows it, back to the string; `None` to undefined. */
export function fromRepr(result: string): string | undefined {
  const m = /^(['"])(.*)\1$/.exec(result);
  return m ? m[2].replace(/\\(.)/g, "$1") : undefined;
}

function samePath(a: string, b: string): boolean {
  return path.resolve(a) === path.resolve(b);
}
