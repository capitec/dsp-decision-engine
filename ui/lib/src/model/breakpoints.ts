import { callNodes, walk, type Hit, type IRNodeJson, type Watch } from "./protocol";

/** What kind of breakpoint is set at a node, for the graph's marker. */
export type BreakpointKind = "plain" | "value" | "iteration";

/**
 * The nodes a set of watches marks, so the graph can show breakpoint state.
 * A plain watch marks its step; a value watch marks every step that writes the
 * value (inside its scope); an iteration watch marks the loop's condition step.
 *
 * A value watch scoped to exactly one step also matches that step's own inputs,
 * not just values it writes — "pause before this step when its input X meets a
 * condition" is the same check as "...when X was last written to meet it", just
 * named from the reader's side instead of the writer's.
 */
export function breakpointNodes(ir: IRNodeJson, watches: Watch[], hit: Hit | null): Map<string, BreakpointKind> {
  const out = new Map<string, BreakpointKind>();
  for (const w of watches) {
    if (w.iteration !== undefined) {
      const loop = findNode(ir, w.path!);
      const cond = loop && loop.kind === "loop" ? loop.children[0] : undefined;
      if (cond && !out.has(cond.path)) out.set(cond.path, "iteration");
    } else if (w.condition !== undefined) {
      // A free-text expression is always scoped to the one step it's written against.
      const at = w.scope?.length === 1 ? w.scope[0] : undefined;
      if (at !== undefined && !out.has(at)) out.set(at, "value");
    } else if (w.name === undefined) {
      if (w.path && !out.has(w.path)) out.set(w.path, "plain");
    } else {
      for (const n of callNodes(ir)) {
        const ownScope = w.scope?.length === 1 && w.scope[0] === n.path;
        const matches = (n.outputs ?? []).includes(w.name) || (ownScope && (n.inputs ?? []).includes(w.name));
        if (!matches) continue;
        if (w.scope?.length && !ownScope && !w.scope.some((s) => n.path === s || n.path.startsWith(`${s}/`))) continue;
        if (!out.has(n.path)) out.set(n.path, "value");
      }
    }
  }
  if (hit?.path && !out.has(hit.path)) out.set(hit.path, "plain");
  return out;
}

function findNode(node: IRNodeJson, path: string): IRNodeJson | undefined {
  let found: IRNodeJson | undefined;
  walk(node, (n) => {
    if (n.path === path) found ??= n;
  });
  return found;
}
