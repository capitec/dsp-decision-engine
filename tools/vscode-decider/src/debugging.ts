import * as path from "node:path";
import * as vscode from "vscode";
import { callNodes, type CallNodeJson, type IRNodeJson } from "@decider/ui";
import { ask, configuredExecutable } from "./python";
import { breakpointPaths, callsAt, fromRepr, NODE_PATH, onlyAt, pausedAt, type SourceLine } from "./locate";

const PYTHON_DEBUGGERS = new Set(["debugpy", "python"]);

export interface DebugHooks {
  /** The shown flow's described IR. */
  ir(): IRNodeJson | undefined;
  /** Show this flow, when a session for another one starts. */
  show(file: string, pipeline?: string): void;
  paused(path: string | undefined): void;
  breakpoints(paths: Set<string>): void;
}

/** Run a pipeline to the end under debugpy, in plain Python, so breakpoints in its steps stop. */
export async function debugFlow(file: string, pipeline?: string, dataset?: string, row?: number): Promise<void> {
  let rows: string | undefined;
  if (!dataset) {
    const named = await ask<{ datasets: { name: string; type: string; error?: string }[] }>(path.dirname(file), "datasets", { file })
      .then((r) => r.datasets.filter((d) => !d.error))
      .catch(() => []);
    const data = await vscode.window.showQuickPick(
      [
        { label: "$(beaker) Sample", detail: "The module's SAMPLE, or sample_request.json beside configs/", value: undefined, name: undefined },
        ...named.map((d) => ({ label: `$(database) ${d.name}`, detail: d.type, value: undefined, name: d.name })),
        { label: "$(file) Choose a file…", detail: "JSON, CSV or Parquet rows", value: "pick", name: undefined },
      ],
      { placeHolder: "Rows to debug the flow on" },
    );
    if (!data) return;
    dataset = data.name;
    if (data.value) {
    const picked = await vscode.window.showOpenDialog({ filters: { Rows: ["json", "csv", "parquet"] }, defaultUri: vscode.Uri.file(path.dirname(file)) });
    if (!picked?.length) return;
    rows = picked[0].fsPath;
    }
  }
  const python = configuredExecutable(path.dirname(file));
  await vscode.debug.startDebugging(vscode.workspace.getWorkspaceFolder(vscode.Uri.file(file)), {
    type: "debugpy",
    request: "launch",
    name: `decider: ${pipeline ?? path.basename(file)}${dataset ? ` · ${dataset}${row == null ? "" : `#${row}`}` : ""}`,
    module: "decider._tooling.bridge.run",
    args: [file, ...(pipeline ? ["--pipeline", pipeline] : []), ...(rows ? ["--data", rows] : []), ...(dataset ? ["--dataset", dataset] : []), ...(row == null ? [] : ["--row", String(row)])],
    cwd: path.dirname(file),
    justMyCode: true,
    console: "integratedTerminal",
    ...(python ? { python } : {}),
    deciderFlow: file,
    deciderPipeline: pipeline,
  });
}

/**
 * Mirror the Python debugger onto the flow: where it is paused, and which nodes
 * have breakpoints. `relocate` maps the current pause again, once a flow is described.
 */
export function watchDebugger(hooks: DebugHooks): { disposables: vscode.Disposable[]; relocate: () => void } {
  let asked = 0;
  const onStack = async (item: vscode.DebugThread | vscode.DebugStackFrame | undefined) => {
    const ask = ++asked;
    const ir = hooks.ir();
    if (!(item instanceof vscode.DebugStackFrame) || !PYTHON_DEBUGGERS.has(item.session.type) || !ir) return hooks.paused(undefined);
    const at = await locate(item, ir);
    if (ask === asked) hooks.paused(at);
  };
  const disposables = [
    vscode.debug.onDidStartDebugSession((s) => {
      if (s.configuration.deciderFlow) hooks.show(s.configuration.deciderFlow, s.configuration.deciderPipeline);
    }),
    vscode.debug.onDidChangeActiveStackItem(onStack),
    vscode.debug.registerDebugAdapterTrackerFactory("*", {
      createDebugAdapterTracker: (s) =>
        PYTHON_DEBUGGERS.has(s.type)
          ? { onWillReceiveMessage: (m) => void (["continue", "next", "stepIn", "stepOut"].includes(m.command) && hooks.paused(undefined)) }
          : undefined,
    }),
    vscode.debug.onDidTerminateDebugSession(() => hooks.paused(undefined)),
    vscode.debug.onDidChangeBreakpoints(() => {
      const ir = hooks.ir();
      if (ir) hooks.breakpoints(breakpointPaths(ir, sourceLines()));
    }),
  ];
  return { disposables, relocate: () => void onStack(vscode.debug.activeStackItem) };
}

/** Add a breakpoint at a step's first line, or remove the ones that stop there. In a reused function it stops at this node only. */
export function toggleBreakpoint(ir: IRNodeJson, nodePath: string) {
  const node = callNodes(ir).find((n) => n.path === nodePath);
  const py = node && userCode(node) ? node.python : undefined;
  if (!py) return;
  const existing = sourceBreakpoints().filter((bp) => breakpointPaths(ir, [lineOf(bp)]).has(nodePath));
  if (existing.length) return vscode.debug.removeBreakpoints(existing);
  const line = py.bodyLine ?? py.line;
  const shared = callsAt(ir, { file: py.file, line }).length > 1;
  const location = new vscode.Location(vscode.Uri.file(py.file), new vscode.Position(line - 1, 0));
  vscode.debug.addBreakpoints([new vscode.SourceBreakpoint(location, true, shared ? onlyAt(nodePath) : undefined)]);
}

/** Make the breakpoints match the graph's dots: toggle each node whose dot changed. */
export function setBreakpointNodes(ir: IRNodeJson, wanted: Set<string>) {
  const now = breakpointPaths(ir, sourceLines());
  for (const p of new Set([...now, ...wanted])) if (now.has(p) !== wanted.has(p)) toggleBreakpoint(ir, p);
}

export function currentBreakpoints(ir: IRNodeJson): Set<string> {
  return breakpointPaths(ir, sourceLines());
}

/** A step whose Python is the user's: a tree or table runs decider's own walker, where `justMyCode` never stops. */
export function userCode(n: CallNodeJson): boolean {
  return !!n.python && n.callKind !== "row";
}

async function locate(item: vscode.DebugStackFrame, ir: IRNodeJson): Promise<string | undefined> {
  try {
    const r = await item.session.customRequest("evaluate", { expression: NODE_PATH, frameId: item.frameId, context: "watch" });
    const at = fromRepr(r.result);
    if (at) return at;
  } catch {
    // decider not importable in the debuggee: fall back to where the frames are.
  }
  const { stackFrames } = await item.session.customRequest("stackTrace", { threadId: item.threadId, startFrame: 0, levels: 50 });
  const frames: { id: number; line: number; source?: { path?: string } }[] = stackFrames;
  const from = Math.max(0, frames.findIndex((f) => f.id === item.frameId));
  return pausedAt(ir, frames.slice(from).flatMap((f) => (f.source?.path ? [{ file: f.source.path, line: f.line }] : [])));
}

function sourceBreakpoints(): vscode.SourceBreakpoint[] {
  return vscode.debug.breakpoints.filter((b): b is vscode.SourceBreakpoint => b instanceof vscode.SourceBreakpoint);
}

function sourceLines(): SourceLine[] {
  return sourceBreakpoints().filter((b) => b.enabled).map(lineOf);
}

function lineOf(bp: vscode.SourceBreakpoint): SourceLine {
  return { file: bp.location.uri.fsPath, line: bp.location.range.start.line + 1, condition: bp.condition };
}
