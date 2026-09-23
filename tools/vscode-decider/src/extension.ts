import * as path from "node:path";
import * as vscode from "vscode";
import { DeciderDebugSession } from "./adapter";
import { analyse, PipelineCodeLens } from "./analysis";
import { GraphPanel } from "./graphPanel";
import type { CallNodeJson, ColumnSummary, DescribeResult, Lineage, RunStatus } from "./protocol";
import { debugpyLibs, pythonCommand } from "./python";
import { StructureProvider } from "./structure";

let lineageChannel: vscode.OutputChannel | undefined;

export function activate(ctx: vscode.ExtensionContext) {
  const structure = new StructureProvider();
  const tree = vscode.window.createTreeView("decider.structure", { treeDataProvider: structure, showCollapseAll: true });
  const showGraph = (describe: DescribeResult) => {
    structure.setDescribe(describe);
    GraphPanel.show(ctx, describe, async (m) => {
      const s = deciderSession();
      if (m.type === "reveal") {
        const node = findNode(describe, m.path);
        if (node?.file) reveal(node.file, node.line);
      } else if (m.type === "record" && s) {
        await s.customRequest("decider.setRecord", { row: m.row });
      } else if (m.type === "lineage") {
        const lineage = s ? ((await s.customRequest("decider.lineage", { name: m.name })) as Lineage) : null;
        GraphPanel.current?.post({ type: "lineage", lineage });
      }
    });
  };

  ctx.subscriptions.push(
    tree,
    vscode.languages.registerCodeLensProvider({ language: "python" }, new PipelineCodeLens()),
    vscode.debug.registerDebugAdapterDescriptorFactory("decider", {
      createDebugAdapterDescriptor: () =>
        new vscode.DebugAdapterInlineImplementation(new DeciderDebugSession({ python: pythonCommand(), debugpyLibs: debugpyLibs() })),
    }),

    vscode.commands.registerCommand("decider.reveal", reveal),

    vscode.commands.registerCommand("decider.visualise", async (uri?: vscode.Uri, pipeline?: string) => {
      const doc = await pickDocument(uri);
      if (!doc) return;
      try {
        const d = await analyse(doc);
        showGraph(pipeline ? { ...d, pipeline } : d);
      } catch (e) {
        vscode.window.showErrorMessage(`decider: ${(e as Error).message}`);
      }
    }),

    vscode.commands.registerCommand("decider.runFlow", async (uri?: vscode.Uri, pipeline?: string) => {
      const doc = await pickDocument(uri);
      if (!doc) return;
      const data = await pickData();
      if (data === undefined) return;
      const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
      await vscode.debug.startDebugging(folder, {
        type: "decider",
        request: "launch",
        name: `decider: ${pipeline ?? path.basename(doc.fileName)}`,
        program: doc.fileName,
        pipeline,
        data: data === "SAMPLE" ? undefined : data,
        stopOnEntry: true,
      });
    }),

    vscode.commands.registerCommand("decider.debugStep", debugStep),

    vscode.commands.registerCommand("decider.focusRecord", async () => {
      const s = deciderSession();
      if (!s) return vscode.window.showInformationMessage("decider: no pipeline session is running");
      const text = await vscode.window.showInputBox({ prompt: "Record (row number) to follow; empty for the whole batch" });
      if (text === undefined) return;
      await s.customRequest("decider.setRecord", { row: text.trim() === "" ? null : Number(text) });
    }),

    vscode.commands.registerCommand("decider.trace", async () => {
      const s = deciderSession();
      if (!s) return vscode.window.showInformationMessage("decider: no pipeline session is running");
      const { columns } = (await s.customRequest("decider.state")) as { columns: ColumnSummary[] | null };
      const name = await vscode.window.showQuickPick((columns ?? []).map((c) => ({ label: c.name, description: `${c.dtype}, from ${c.producer}` })), {
        placeHolder: "Column to trace",
      });
      if (!name) return;
      const lineage = (await s.customRequest("decider.lineage", { name: name.label })) as Lineage;
      lineageChannel ??= vscode.window.createOutputChannel("decider lineage");
      const out = lineageChannel;
      out.clear();
      out.appendLine(renderLineage(lineage));
      out.show(true);
    }),

    vscode.debug.onDidStartDebugSession(async (s) => {
      if (s.type !== "decider") return;
      const d = (await s.customRequest("decider.describe")) as DescribeResult;
      showGraph(d);
    }),

    vscode.debug.onDidReceiveDebugSessionCustomEvent(async (e) => {
      if (e.session.type !== "decider" || e.event !== "decider.status") return;
      const body = e.body as RunStatus;
      structure.setStatus(body.current, body.finishedPaths);
      GraphPanel.current?.post({ type: "status", ...body });
      const { columns } = (await e.session.customRequest("decider.state")) as { columns: ColumnSummary[] | null };
      GraphPanel.current?.post({ type: "state", columns, rows: columns?.[0]?.rows ?? 0 });
    }),

    vscode.debug.onDidTerminateDebugSession((s) => {
      if (s.type !== "decider") return;
      structure.setStatus(null, []);
      GraphPanel.current?.post({ type: "status", current: null, finished: true, finishedPaths: [], visits: {}, record: null });
    }),
  );
}

/**
 * Drop into the Python of the current node: attach debugpy to the bridge
 * process, break once on the function's first statement, then step the flow.
 */
async function debugStep() {
  const s = deciderSession();
  if (!s) return vscode.window.showInformationMessage("decider: no pipeline session is running");
  const info = (await s.customRequest("decider.info")) as { debugpyPort?: number; node: CallNodeJson | null; current: { when: string } | null };
  if (!info.debugpyPort) return vscode.window.showErrorMessage("decider: debugpy is not available (install the ms-python.debugpy extension)");
  const py = info.node?.kind === "call" ? info.node.python : null;
  if (!py?.bodyLine || info.current?.when !== "before") {
    return vscode.window.showInformationMessage("decider: stop just before a step that runs Python (a function, frame or tree reference) first");
  }
  // ponytail: stops on the first row the step runs; a per-record stop needs the row index passed into the call.
  const location = new vscode.Location(vscode.Uri.file(py.file), new vscode.Position(py.bodyLine - 1, 0));
  const bp = new vscode.SourceBreakpoint(location, true, undefined, "1");
  vscode.debug.addBreakpoints([bp]);
  const attached = await vscode.debug.startDebugging(
    s.workspaceFolder,
    { type: "debugpy", request: "attach", name: `python: ${info.node!.path}`, connect: { host: "127.0.0.1", port: info.debugpyPort }, justMyCode: false },
    { parentSession: s, compact: true },
  );
  if (!attached) {
    vscode.debug.removeBreakpoints([bp]);
    return;
  }
  await s.customRequest("next", { threadId: 1 });
  const sub = vscode.debug.onDidReceiveDebugSessionCustomEvent((e) => {
    if (e.session === s && e.event === "decider.status") {
      vscode.debug.removeBreakpoints([bp]);
      sub.dispose();
    }
  });
}

async function pickDocument(uri?: vscode.Uri): Promise<vscode.TextDocument | undefined> {
  if (uri) return vscode.workspace.openTextDocument(uri);
  const doc = vscode.window.activeTextEditor?.document;
  if (doc?.languageId === "python") return doc;
  vscode.window.showInformationMessage("decider: open a Python file that defines a pipeline");
  return undefined;
}

async function pickData(): Promise<unknown | "SAMPLE" | undefined> {
  const choice = await vscode.window.showQuickPick(
    [
      { label: "SAMPLE", description: "the module's SAMPLE rows" },
      { label: "JSON file…", description: "a file holding a list of records" },
      { label: "Type rows…", description: "paste JSON records" },
    ],
    { placeHolder: "Input rows for the run" },
  );
  if (!choice) return undefined;
  if (choice.label === "SAMPLE") return "SAMPLE";
  if (choice.label === "JSON file…") {
    const picked = await vscode.window.showOpenDialog({ canSelectMany: false, filters: { JSON: ["json"] } });
    return picked?.[0]?.fsPath;
  }
  const text = await vscode.window.showInputBox({
    prompt: "JSON list of records",
    value: '[{"net_income": 9000.0, "expenses": 4000.0}]',
    validateInput: (v) => {
      try {
        return Array.isArray(JSON.parse(v)) ? null : "expected a JSON list";
      } catch (e) {
        return (e as Error).message;
      }
    },
  });
  return text === undefined ? undefined : JSON.parse(text);
}

async function reveal(file: string, line: number | null) {
  const doc = await vscode.workspace.openTextDocument(file);
  const editor = await vscode.window.showTextDocument(doc, { preserveFocus: true, preview: true, viewColumn: vscode.ViewColumn.One });
  if (line) {
    const pos = new vscode.Position(line - 1, 0);
    editor.revealRange(new vscode.Range(pos, pos), vscode.TextEditorRevealType.InCenter);
    editor.selection = new vscode.Selection(pos, pos);
  }
}

function findNode(d: DescribeResult, p: string) {
  let found: CallNodeJson | { file: string | null; line: number | null } | undefined;
  const visit = (n: DescribeResult["ir"]) => {
    if (n.path === p) found = n;
    else if (n.kind !== "call") n.children.forEach(visit);
  };
  visit(d.ir);
  return found;
}

function deciderSession(): vscode.DebugSession | undefined {
  const s = vscode.debug.activeDebugSession;
  return s?.type === "decider" ? s : undefined;
}

function renderLineage(l: Lineage, indent = ""): string {
  const value = l.value === null || l.value === undefined ? "" : ` = ${JSON.stringify(l.value)}`;
  const line = `${indent}${l.name}${value}  ←  ${l.producer ?? "<input column>"}${l.via ? ` (${l.via})` : ""}`;
  return [line, ...l.inputs.map((i) => renderLineage(i, indent + "    "))].join("\n");
}
