import * as path from "node:path";
import * as vscode from "vscode";
import { DeciderDebugSession } from "./adapter";
import { analyse, PipelineCodeLens } from "./analysis";
import { GraphPanel } from "./graphPanel";
import type { CallNodeJson, Checkpoint, ColumnSummary, DescribeResult, Lineage } from "./protocol";
import { debugpyLibs, pythonCommand } from "./python";
import { StructureProvider } from "./structure";

let lineageChannel: vscode.OutputChannel | undefined;

export function activate(ctx: vscode.ExtensionContext) {
  const structure = new StructureProvider();
  const tree = vscode.window.createTreeView("decider.structure", { treeDataProvider: structure, showCollapseAll: true });
  let lastDescribe: { uri: vscode.Uri; describe: DescribeResult } | undefined;

  const showGraph = (uri: vscode.Uri, describe: DescribeResult) => {
    lastDescribe = { uri, describe };
    structure.setDescribe(describe);
    GraphPanel.show(ctx, describe, (m) => {
      if (m.type === "reveal") {
        const node = findNode(describe, m.path);
        if (node?.file) reveal(node.file, node.line);
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
        showGraph(doc.uri, pipeline ? { ...d, pipeline } : d);
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

    vscode.commands.registerCommand("decider.trace", async () => {
      const s = vscode.debug.activeDebugSession;
      if (s?.type !== "decider") return vscode.window.showInformationMessage("decider: no pipeline session is running");
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
      showGraph(vscode.Uri.file(s.configuration.program), d);
    }),

    vscode.debug.onDidReceiveDebugSessionCustomEvent(async (e) => {
      if (e.session.type !== "decider" || e.event !== "decider.status") return;
      const body = e.body as { current: Checkpoint | null; finished: boolean; finishedPaths: string[] };
      structure.setStatus(body.current, body.finishedPaths);
      GraphPanel.current?.post({ type: "status", ...body });
      const { columns } = (await e.session.customRequest("decider.state")) as { columns: ColumnSummary[] | null };
      GraphPanel.current?.post({ type: "state", columns });
    }),

    vscode.debug.onDidTerminateDebugSession((s) => {
      if (s.type !== "decider") return;
      structure.setStatus(null, []);
      GraphPanel.current?.post({ type: "status", current: null, finished: true, finishedPaths: [] });
    }),
  );
}

/**
 * Drop into the Python of the current node: attach debugpy to the bridge
 * process, break once on the function's first statement, then step the flow.
 */
async function debugStep() {
  const s = vscode.debug.activeDebugSession;
  if (s?.type !== "decider") return vscode.window.showInformationMessage("decider: no pipeline session is running");
  const info = (await s.customRequest("decider.info")) as { debugpyPort?: number; node: CallNodeJson | null };
  if (!info.debugpyPort) return vscode.window.showErrorMessage("decider: debugpy is not available (install the ms-python.debugpy extension)");
  if (!info.node || info.node.kind !== "call" || !info.node.file || !info.node.bodyLine) {
    return vscode.window.showInformationMessage("decider: the current node is not a Python function; step into a call node first");
  }
  const location = new vscode.Location(vscode.Uri.file(info.node.file), new vscode.Position(info.node.bodyLine - 1, 0));
  const bp = new vscode.SourceBreakpoint(location, true, undefined, "1");
  vscode.debug.addBreakpoints([bp]);
  const attached = await vscode.debug.startDebugging(
    s.workspaceFolder,
    { type: "debugpy", request: "attach", name: `python: ${info.node.path}`, connect: { host: "127.0.0.1", port: info.debugpyPort }, justMyCode: false },
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

function renderLineage(l: Lineage, indent = ""): string {
  const line = `${indent}${l.name}  ←  ${l.producer ?? "<input column>"}`;
  return [line, ...l.inputs.map((i) => renderLineage(i, indent + "    "))].join("\n");
}
