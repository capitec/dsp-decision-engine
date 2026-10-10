import * as vscode from "vscode";
import type { IRNodeJson } from "@decikit/ui";
import { makeHandler, type AgentState, type DebugControl } from "./agentActions";
import { listen } from "./agentSocket";
import { currentBreakpoints, debugFlow, setBreakpointNodes } from "./debugging";
import type { Item } from "./outline";
import type { StructureProvider } from "./structure";
import { GraphPanel } from "./graphPanel";

/** What the extension already holds, handed in so the bridge reads it rather than keeping a copy. */
export interface Host {
  shown(): { file: string; pipeline?: string; describe?: unknown; error?: string } | undefined;
  paused(): string | undefined;
  graphSelection(): string | undefined;
  ir(): IRNodeJson | undefined;
  structure: StructureProvider;
  tree: vscode.TreeView<Item>;
  open(file: string, line: number | null): Promise<void>;
}

const COMMANDS: Record<Exclude<DebugControl, "stop">, string> = {
  continue: "workbench.action.debug.continue",
  step_over: "workbench.action.debug.stepOver",
  step_in: "workbench.action.debug.stepInto",
  step_out: "workbench.action.debug.stepOut",
};
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Let the MCP server reach each workspace folder's window: selection, navigation and debugging. */
export function startAgentBridge(host: Host): vscode.Disposable {
  const state = vscodeState(host);
  const stops = new Map<string, () => void>();
  let disposed = false;
  const add = async (f: vscode.WorkspaceFolder) => {
    if (f.uri.scheme !== "file") return;
    try {
      const stop = await listen(f.uri.fsPath, makeHandler(state, f.uri.fsPath));
      if (disposed) stop();
      else stops.set(f.uri.fsPath, stop);
    } catch (e) {
      console.warn(`decikit: editor bridge unavailable for ${f.uri.fsPath}: ${(e as Error).message}`);
    }
  };
  vscode.workspace.workspaceFolders?.forEach(add);
  const changes = vscode.workspace.onDidChangeWorkspaceFolders((e) => {
    e.added.forEach(add);
    for (const f of e.removed) stops.get(f.uri.fsPath)?.();
  });
  return new vscode.Disposable(() => {
    disposed = true;
    changes.dispose();
    for (const stop of stops.values()) stop();
  });
}

function vscodeState(h: Host): AgentState {
  /** The paused thread's top frame, with the session to ask about it. */
  async function topFrame() {
    const session = vscode.debug.activeDebugSession;
    if (!session) return undefined;
    const item = vscode.debug.activeStackItem;
    try {
      const threadId = item?.threadId ?? (await session.customRequest("threads")).threads[0]?.id;
      const { stackFrames } = await session.customRequest("stackTrace", { threadId, startFrame: 0, levels: 1 });
      const top = stackFrames[0];
      return top && { session, id: item instanceof vscode.DebugStackFrame ? item.frameId : (top.id as number), file: top.source?.path as string, line: top.line as number, name: top.name as string };
    } catch {
      return undefined; // running, not stopped
    }
  }

  // ponytail: polls for the next stop; the debug adapter tracker in debugging.ts has no "stopped" event to wait on.
  async function nextStop(ms = 15000) {
    await sleep(150);
    for (const end = Date.now() + ms; Date.now() < end; await sleep(100)) {
      if (!vscode.debug.activeDebugSession) return undefined;
      if (h.paused()) return h.paused();
    }
  }

  return {
    editor() {
      const e = vscode.window.activeTextEditor;
      if (!e) return undefined;
      const s = e.selection;
      const pos = (p: vscode.Position) => ({ line: p.line + 1, column: p.character + 1 });
      return { file: e.document.uri.fsPath, text: e.document.getText(), cursor: e.document.offsetAt(s.active), start: pos(s.start), end: pos(s.end), selected: e.document.getText(s) };
    },
    shown() {
      const s = h.shown();
      return s && { file: s.file, pipeline: s.pipeline, described: !!s.describe, error: s.error };
    },
    structureSelection: () => h.tree.selection.map((i) => i.path),
    graphSelection: h.graphSelection,
    paused: h.paused,
    breakpoints() {
      const ir = h.ir();
      return ir ? [...currentBreakpoints(ir)] : [];
    },
    async frame() {
      const f = await topFrame();
      return f && { file: f.file, line: f.line, name: f.name };
    },
    async highlight(node) {
      const item = h.structure.find(node);
      if (!item) return false;
      // Revealing opens the Structure view when it is closed, so the user sees what is pointed at.
      await h.tree.reveal(item, { select: true, focus: false });
      GraphPanel.post({ type: "select", path: node });
      return true;
    },
    reveal: (file, line) => h.open(file, line),
    async showFlow(file, pipeline) {
      await vscode.commands.executeCommand("decikit.showFlow", vscode.Uri.file(file), pipeline);
    },
    setBreakpoints(nodes) {
      const ir = h.ir();
      if (!ir) throw new Error("no flow is shown in VS Code: call show_flow first");
      setBreakpointNodes(ir, new Set(nodes));
      return [...currentBreakpoints(ir)];
    },
    async debugFlow(file, pipeline, dataset) {
      await debugFlow(file, pipeline, dataset, undefined, false);
      return nextStop();
    },
    async debugControl(action) {
      if (!vscode.debug.activeDebugSession) throw new Error("no debug session is running: start one with debug_flow");
      if (action === "stop") return void (await vscode.commands.executeCommand("workbench.action.debug.stop"));
      await vscode.commands.executeCommand(COMMANDS[action]);
      return nextStop();
    },
    async debugEvaluate(expression) {
      const f = await topFrame();
      if (!f) throw new Error("the debugger is not paused: set_breakpoints and debug_flow, or wait for debug_control to report a pause");
      return (await f.session.customRequest("evaluate", { expression, frameId: f.id, context: "repl" })).result;
    },
  };
}
