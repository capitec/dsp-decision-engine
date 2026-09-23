import * as path from "node:path";
import * as vscode from "vscode";
import { DeciderDebugSession } from "./adapter";
import { analyse, PipelineCodeLens } from "./analysis";
import { listRefs, materialise, repoRoot } from "./git";
import { GraphPanel } from "./graphPanel";
import type { CallNodeJson, ColumnHistory, ColumnSummary, DescribeResult, FromWebview, Lineage, RecordKey, RunStatus, ToWebview } from "./protocol";
import { debugpyLibs, pythonCommand } from "./python";
import { runComparison, type Side } from "./compareRuns";
import { withBridge } from "./bridge";
import { summariseSweep, type Scenario, type SweepResponse } from "./sweep";
import { StructureProvider } from "./structure";

let lineageChannel: vscode.OutputChannel | undefined;
/** The pipeline the views show: its file and name. */
let shown: { file: string; pipeline: string } | undefined;

export function activate(ctx: vscode.ExtensionContext) {
  const structure = new StructureProvider();
  const tree = vscode.window.createTreeView("decider.structure", { treeDataProvider: structure, showCollapseAll: true });

  const showGraph = (file: string, describe: DescribeResult) => {
    shown = { file, pipeline: describe.pipeline };
    structure.setDescribe(describe);
    return GraphPanel.show(ctx, describe, (m) => onWebview(m, describe).catch((e) => vscode.window.showErrorMessage(`decider: ${(e as Error).message}`)));
  };

  ctx.subscriptions.push(
    tree,
    vscode.languages.registerCodeLensProvider({ language: "python" }, new PipelineCodeLens()),
    vscode.debug.registerDebugAdapterDescriptorFactory("decider", {
      createDebugAdapterDescriptor: () =>
        new vscode.DebugAdapterInlineImplementation(new DeciderDebugSession({ python: pythonCommand(), debugpyLibs: debugpyLibs() })),
    }),

    vscode.commands.registerCommand("decider.reveal", reveal),

    vscode.commands.registerCommand("decider.selectStep", async (nodePath: string, file: string | null, line: number | null) => {
      post({ type: "select", path: nodePath });
      if (file) await reveal(file, line);
    }),

    vscode.commands.registerCommand("decider.visualise", async (uri?: vscode.Uri, pipeline?: string) => {
      const doc = await pickDocument(uri);
      if (!doc) return;
      try {
        const d = await analyse(doc);
        await showGraph(doc.fileName, pipeline ? { ...d, pipeline } : d);
      } catch (e) {
        vscode.window.showErrorMessage(`decider: ${(e as Error).message}`);
      }
    }),

    vscode.commands.registerCommand("decider.runFlow", async (uri?: vscode.Uri, pipeline?: string) => {
      const doc = await pickDocument(uri);
      if (!doc) return;
      const data = await pickData();
      if (data === undefined) return;
      await vscode.debug.startDebugging(vscode.workspace.getWorkspaceFolder(doc.uri), {
        type: "decider",
        request: "launch",
        name: `decider: ${pipeline ?? path.basename(doc.fileName)}`,
        program: doc.fileName,
        pipeline,
        data: data === "SAMPLE" ? undefined : data,
        stopOnEntry: true,
        // The flow panel shows what the console would; keep the editor's height for the code.
        internalConsoleOptions: "neverOpen",
      });
    }),

    vscode.commands.registerCommand("decider.debugStep", debugStep),

    vscode.commands.registerCommand("decider.compareRevision", async (uri?: vscode.Uri) => {
      if (uri || !shown) await vscode.commands.executeCommand("decider.visualise", uri);
      await compareRevision();
    }),

    vscode.commands.registerCommand("decider.whatIf", async (uri?: vscode.Uri) => {
      if (uri || !GraphPanel.current) await vscode.commands.executeCommand("decider.visualise", uri);
      GraphPanel.post({ type: "tab", tab: "params" });
    }),

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
      lineageChannel.clear();
      lineageChannel.appendLine(renderLineage(lineage));
      lineageChannel.show(true);
    }),

    vscode.debug.onDidStartDebugSession(async (s) => {
      if (s.type !== "decider") return;
      const d = (await s.customRequest("decider.describe")) as DescribeResult;
      showGraph(s.configuration.program, d);
    }),

    vscode.debug.onDidReceiveDebugSessionCustomEvent(async (e) => {
      if (e.session.type !== "decider" || e.event !== "decider.status") return;
      const body = e.body as RunStatus;
      structure.setStatus(body.current, body.finishedPaths);
      GraphPanel.post({ type: "status", ...body });
      const { columns, key } = (await e.session.customRequest("decider.state")) as { columns: ColumnSummary[] | null; key: RecordKey };
      GraphPanel.post({ type: "state", columns, rows: columns?.[0]?.rows ?? 0, key });
    }),

    vscode.debug.onDidTerminateDebugSession((s) => {
      if (s.type !== "decider") return;
      structure.setStatus(null, []);
      GraphPanel.post({ type: "status", current: null, finished: true, finishedPaths: [], visits: {}, record: null });
      GraphPanel.post({ type: "state", columns: null, rows: 0, key: null });
    }),
  );
}

function post(m: ToWebview) {
  GraphPanel.post(m);
}

async function onWebview(m: FromWebview, describe: DescribeResult) {
  const s = deciderSession();
  switch (m.type) {
    case "reveal": {
      const node = findNode(describe, m.path);
      if (node?.file) await reveal(node.file, node.line);
      break;
    }
    case "record":
      await s?.customRequest("decider.setRecord", { row: m.row });
      break;
    case "lineage": {
      if (!s) return post({ type: "lineage", lineage: null, history: null });
      const [lineage, history] = await Promise.all([
        s.customRequest("decider.lineage", { name: m.name }) as Thenable<Lineage>,
        s.customRequest("decider.column", { name: m.name }) as Thenable<ColumnHistory>,
      ]);
      post({ type: "lineage", lineage, history });
      break;
    }
    case "treePath":
      if (s) post({ type: "treePath", ...((await s.customRequest("decider.treePath", { path: m.path })) as { path: string; row: number; visited: string[]; result: unknown[] }) });
      break;
    case "rewind":
      await s?.customRequest("decider.rewind", { path: m.path });
      break;
    case "restartWith":
      await s?.customRequest("decider.restartWith", { params: m.params });
      break;
    case "whatIf": {
      if (!shown) return;
      await compare(
        { label: "defaults", file: shown.file, pipeline: shown.pipeline },
        { label: m.label, file: shown.file, pipeline: shown.pipeline, params: m.params, overrides: m.overrides, row: m.row },
      );
      break;
    }
    case "compareRevision":
      await compareRevision();
      break;
    case "sweep":
      await runSweep(m.scenarios, m.fromHere && !!s);
      break;
    case "runTo":
      await runTo(m.path);
      break;
    case "step":
      await s?.customRequest("next", { threadId: 1 });
      break;
    case "maximise":
      await vscode.commands.executeCommand("workbench.action.toggleMaximizeEditorGroup");
      break;
    case "openDiff":
      if (lastFiles) {
        const node = findNode(describe, m.path);
        await vscode.commands.executeCommand("vscode.diff", vscode.Uri.file(lastFiles.a), vscode.Uri.file(lastFiles.b), `${path.basename(lastFiles.b)}: ${lastFiles.label} ↔ working tree`, {
          selection: node?.line ? new vscode.Range(node.line - 1, 0, node.line - 1, 0) : undefined,
        });
      }
      break;
  }
}

/** Run scenarios next to the unchanged run: forked from the paused session, or from the start. */
async function runSweep(list: Scenario[], fromHere: boolean) {
  post({ type: "tab", tab: "scenarios" });
  post({ type: "sweep", sweep: null, busy: `Running ${list.length} scenario${list.length === 1 ? "" : "s"}…` });
  try {
    const s = deciderSession();
    const r = (fromHere && s
      ? await s.customRequest("decider.sweep", { scenarios: list })
      : await withBridge({ python: pythonCommand(), cwd: path.dirname(shown!.file) }, (b) =>
          b.request("sweep", { scenarios: list, from_here: false, file: shown!.file, pipeline: shown!.pipeline }),
        )) as SweepResponse;
    post({ type: "sweep", sweep: summariseSweep(r, list) });
  } catch (e) {
    post({ type: "sweep", sweep: null, error: (e as Error).message });
  }
}

/** The files of the latest revision comparison, for "view code diff". */
let lastFiles: { a: string; b: string; label: string } | undefined;

/** Pause before a step: a function breakpoint on its path, then run (or start) the flow to it. */
async function runTo(nodePath: string) {
  vscode.debug.addBreakpoints([new vscode.FunctionBreakpoint(nodePath)]);
  const s = deciderSession();
  if (s) await s.customRequest("continue", { threadId: 1 });
  else if (shown)
    await vscode.debug.startDebugging(undefined, { type: "decider", request: "launch", name: `decider: ${shown.pipeline}`, program: shown.file, pipeline: shown.pipeline, stopOnEntry: false, internalConsoleOptions: "neverOpen" });
}

async function compare(a: Side, b: Side) {
  post({ type: "tab", tab: "compare" });
  post({ type: "compare", comparison: null, busy: `Running ${a.label} and ${b.label}…` });
  try {
    const comparison = await runComparison(a, b, pythonCommand(), path.dirname(b.file));
    comparison.paramsDocs = { a: a.params ?? {}, b: b.params ?? {} };
    if (a.file !== b.file) {
      comparison.files = { a: a.file, b: b.file };
      lastFiles = { ...comparison.files, label: a.label };
    }
    post({ type: "compare", comparison });
  } catch (e) {
    post({ type: "compare", comparison: null, error: (e as Error).message });
  }
}

/** Pick a tag, branch or commit, and compare the shown pipeline there against the working tree. */
async function compareRevision() {
  if (!shown) return vscode.window.showInformationMessage("decider: visualise a pipeline first");
  const { file, pipeline } = shown;
  const root = await repoRoot(file);
  const pick = await vscode.window.showQuickPick(
    (await listRefs(root)).map((r) => ({ label: r.label, description: r.description, ref: r.ref })),
    { placeHolder: "Compare the working tree against…", matchOnDescription: true },
  );
  if (!pick) return;
  const tree = await materialise(root, pick.ref);
  await compare({ label: pick.label, file: path.join(tree, path.relative(root, file)), pipeline }, { label: "working tree", file, pipeline });
}

/**
 * Drop into the Python of the current node: attach debugpy to the bridge
 * process, break on the function's first statement (for the focused record
 * only, when one is focused), then step the flow.
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
  const { condition } = (await s.customRequest("decider.debugCondition", { path: info.node!.path })) as { condition: string | null };
  const location = new vscode.Location(vscode.Uri.file(py.file), new vscode.Position(py.bodyLine - 1, 0));
  const bp = new vscode.SourceBreakpoint(location, true, condition ?? undefined, condition ? undefined : "1");
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
  let found: { file: string | null; line: number | null } | undefined;
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
