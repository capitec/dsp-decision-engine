import * as path from "node:path";
import { configAt } from "./agentPath";

/** 1-based, as an agent reads line numbers in a file. */
export interface Pos {
  line: number;
  column: number;
}

export interface ActiveEditor {
  file: string;
  text: string;
  /** Offset of the cursor in `text`. */
  cursor: number;
  start: Pos;
  end: Pos;
  selected: string;
}

export const DEBUG_CONTROLS = ["continue", "step_over", "step_in", "step_out", "stop"] as const;
export type DebugControl = (typeof DEBUG_CONTROLS)[number];

/** What the editor window offers an agent; `agentBridge.ts` backs it with VS Code. */
export interface AgentState {
  editor(): ActiveEditor | undefined;
  shown(): { file: string; pipeline?: string; described: boolean; error?: string } | undefined;
  structureSelection(): string[];
  graphSelection(): string | undefined;
  paused(): string | undefined;
  breakpoints(): string[];
  /** The innermost frame of the paused thread, if the debugger is stopped. */
  frame(): Promise<{ file: string; line: number; name: string } | undefined>;
  /** Select and reveal a node in the Structure view and graph; false when the shown flow has no such node. */
  highlight(node: string): Promise<boolean>;
  reveal(file: string, line: number): Promise<void>;
  showFlow(file: string, pipeline?: string): Promise<void>;
  /** Make the breakpoints match `nodes`; returns the nodes that now have one. */
  setBreakpoints(nodes: string[]): string[];
  /** Start a debug session; resolves with the node it first pauses at, if it pauses. */
  debugFlow(file: string, pipeline?: string, dataset?: string): Promise<string | undefined>;
  /** Resolves with the node the debugger next pauses at, if it does. */
  debugControl(action: DebugControl): Promise<string | undefined>;
  debugEvaluate(expression: string): Promise<string>;
}

export interface Action {
  kind: string;
  [key: string]: unknown;
}

const str = (a: Action, key: string, required = true): string | undefined => {
  const v = a[key];
  if (typeof v === "string" && v) return v;
  if (v == null && !required) return undefined;
  throw new Error(`${a.kind}: '${key}' must be a non-empty string`);
};

/** Answer one action from the MCP server, resolving relative files against the window's folder. */
export function makeHandler(state: AgentState, root: string): (a: Action) => Promise<unknown> {
  const file = (a: Action, key: string) => path.resolve(root, str(a, key)!);
  const optFile = (a: Action, key: string) => (a[key] == null ? undefined : file(a, key));
  return async (a) => {
    switch (a.kind) {
      case "selection":
        return selection(state);
      case "highlight": {
        const flow = optFile(a, "flow");
        if (flow && state.shown()?.file !== flow) await state.showFlow(flow);
        if (!state.shown()) throw new Error("no flow is shown: call show_flow first");
        const nodes = Array.isArray(a.nodes) ? a.nodes.map(String) : [];
        // The Structure view and the graph each hold one selection.
        let found: string | undefined;
        for (const n of nodes) if (await state.highlight(n)) { found = n; break; }
        if (!found) throw new Error(`none of ${JSON.stringify(nodes)} is a node of the shown flow`);
        return { highlighted: found, skipped: nodes.filter((n) => n !== found) };
      }
      case "reveal":
        await state.reveal(file(a, "path"), Number(a.line) || 1);
        return { ack: true };
      case "show_flow": {
        const f = file(a, "file");
        await state.showFlow(f, str(a, "pipeline", false));
        return state.shown();
      }
      case "set_breakpoints": {
        const wanted = Array.isArray(a.nodes) ? a.nodes.map(String) : [];
        if (!state.shown()?.described) throw new Error("no described flow is shown: call show_flow first");
        const now = state.setBreakpoints(wanted);
        return { breakpoints: now, not_set: wanted.filter((n) => !now.includes(n)) };
      }
      case "debug_flow":
        return { started: true, paused: (await state.debugFlow(file(a, "file"), str(a, "pipeline", false), str(a, "dataset", false))) ?? null };
      case "debug_control": {
        const action = str(a, "action") as DebugControl;
        if (!DEBUG_CONTROLS.includes(action)) throw new Error(`debug_control: action must be one of ${DEBUG_CONTROLS.join(", ")}`);
        return { paused: (await state.debugControl(action)) ?? null };
      }
      case "debug_evaluate":
        return { result: await state.debugEvaluate(str(a, "expression")!) };
      default:
        throw new Error(`unknown action '${a.kind}'`);
    }
  };
}

async function selection(state: AgentState) {
  const e = state.editor();
  const config = e && /\.(json|ya?ml)$/.test(e.file) ? configAt(e.text, e.cursor) : undefined;
  return {
    file: e?.file ?? null,
    selection: e && { start: e.start, end: e.end, text: e.selected },
    config: config ?? null,
    flow: state.shown() ?? null,
    structure_selection: state.structureSelection(),
    graph_selection: state.graphSelection() ?? null,
    debug: { paused: state.paused() ?? null, frame: (await state.frame()) ?? null, breakpoints: state.breakpoints() },
  };
}
