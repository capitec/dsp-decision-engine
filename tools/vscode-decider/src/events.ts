import { previewOf, type IRNodeJson, type SessionEvent, type Summary, type Visits } from "@decider/ui";

export interface Read {
  /** Lines for the Debug Console, with their output category. */
  lines: [string, "stdout" | "stderr" | "console"][];
  paused?: string;
}

/** Fold session events into `visits` and `finishedPaths`, and render what the console shows. */
export function readEvents(events: SessionEvent[], nodes: Map<string, IRNodeJson>, visits: Visits, finishedPaths: string[], edits: Record<string, "delete" | "replace"> = {}): Read {
  const read: Read = { lines: [] };
  for (const ev of events) {
    const p = ev.origin?.path ?? "";
    switch (ev.kind) {
      case "node_started":
        if (nodes.get(p)?.kind === "call") visits[p] = {};
        break;
      case "node_visited":
        (visits[p] ??= {})[ev.origin!.locator!] = ev.rows as number;
        break;
      case "node_finished": {
        const outs = Object.entries(ev.outputs as Record<string, Summary>);
        if (nodes.get(p)?.kind === "call") finishedPaths.push(p);
        if (outs.length) read.lines.push([`${p}:  ${outs.map(([n, s]) => `${n} = ${previewOf(s, n)}`).join(";  ")}\n`, "stdout"]);
        break;
      }
      case "overridden":
        read.lines.push([`set ${ev.name} (${ev.producer})\n`, "console"]);
        break;
      case "warning":
        read.lines.push([`warning: ${ev.message}\n`, "console"]);
        break;
      case "error":
        read.lines.push([`error in ${ev.path ?? "?"}: ${ev.message}\n`, "stderr"]);
        break;
      case "run_finished":
        read.lines.push([`run finished; columns: ${Object.keys(ev.output as object).join(", ")}\n`, "console"]);
        break;
      case "paused":
        read.paused ??= ev.reason as string;
        break;
      case "edited": {
        edits[ev.path as string] = ev.action as "delete" | "replace";
        // Everything from the edited step on runs again, in the order the nodes were described.
        const order = [...nodes.keys()];
        const from = order.indexOf(ev.path as string);
        if (from >= 0) {
          const rerun = new Set(order.slice(from));
          for (let i = finishedPaths.length - 1; i >= 0; i--) if (rerun.has(finishedPaths[i])) finishedPaths.splice(i, 1);
        }
        read.lines.push([`${ev.action === "delete" ? "skipped" : "swapped in edited code for"} ${ev.path}; re-running from there\n`, "console"]);
        break;
      }
    }
  }
  return read;
}
