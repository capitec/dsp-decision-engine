import * as path from "node:path";
import * as vscode from "vscode";
import { DeciderDebugSession } from "./adapter";
import { analyse, PipelineCodeLens } from "./analysis";
import { listRefs, materialise, repoRoot } from "./git";
import { GraphPanel } from "./graphPanel";
import type { CallNodeJson, ColumnSummary, Controls, DescribeResult, FromWebview, Lineage, RecordKey, RunStatus, ToWebview, ValueHistory } from "./protocol";
import { debugpyLibs, pythonCommand } from "./python";
import { runComparison, type Side } from "./compareRuns";
import { compareTraces, type TraceResult } from "./compare";
import { withBridge } from "./bridge";
import { summariseSweep, type Scenario, type SweepResponse } from "./sweep";
import { StructureProvider } from "./structure";

let lineageChannel: vscode.OutputChannel | undefined;
/** The pipeline the views show: its file and name. */
let shown: { file: string; pipeline: string } | undefined;
/** Forces and value breakpoints set in the flow panel; every decider launch starts with them. */
let controls: Controls = { forces: [], watches: [] };

export function activate(ctx: vscode.ExtensionContext) {
  const structure = new StructureProvider();
  const tree = vscode.window.createTreeView("decider.structure", { treeDataProvider: structure, showCollapseAll: true });

  const showGraph = (file: string, describe: DescribeResult) => {
    shown = { file, pipeline: describe.pipeline };
    structure.setDescribe(describe);
    return GraphPanel.show(ctx, describe, (m) => onWebview(m, describe).catch((e) => vscode.window.showErrorMessage(`decider: ${(e as Error).message}`)));
  };

  ctx.subscriptions.push(
    vscode.debug.registerDebugConfigurationProvider("decider", {
      resolveDebugConfiguration: (_folder, config) => ({ ...config, controls: config.controls ?? controls }),
    }),
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
      // From the palette: the flow in the active editor, not whichever one the panel last showed.
      const target = uri ?? vscode.window.activeTextEditor?.document.uri;
      if (!GraphPanel.current || (target && target.fsPath !== shown?.file)) await vscode.commands.executeCommand("decider.visualise", target);
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
      if (body.current) clearRunTo(body.current.path);
      structure.setStatus(body.current, body.finishedPaths);
      GraphPanel.post({ type: "status", ...body });
      const { columns, key } = (await e.session.customRequest("decider.state")) as { columns: ColumnSummary[] | null; key: RecordKey };
      GraphPanel.post({ type: "state", columns, rows: columns?.[0]?.rows ?? 0, key });
    }),

    vscode.debug.onDidTerminateDebugSession((s) => {
      if (s.type !== "decider") return;
      clearRunTo();
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
      // A value not written yet has no lineage, but its history still lists what will set it.
      const [lineage, history] = await Promise.all([
        Promise.resolve(s.customRequest("decider.lineage", { name: m.name }) as Thenable<Lineage>).catch(() => null),
        s.customRequest("decider.changes", { name: m.name }) as Thenable<ValueHistory>,
      ]);
      post({ type: "lineage", lineage, history });
      break;
    }
    case "treePath":
      if (s) post({ type: "treePath", ...((await s.customRequest("decider.treePath", { path: m.path })) as { path: string; row: number; visited: string[]; result: unknown[] }) });
      break;
    case "rerun":
      await s?.customRequest("decider.rerun", { path: m.path, back: m.back, when: m.when });
      break;
    case "goTo":
      await s?.customRequest("decider.goTo", { change: m.change });
      break;
    case "rewind":
      await s?.customRequest("decider.rewind", { path: m.path });
      break;
    case "compareEdits": {
      if (!s) return;
      post({ type: "tab", tab: "compare" });
      post({ type: "compare", comparison: null, busy: "Running the flow as started and as edited…" });
      try {
        const r = (await s.customRequest("decider.compareEdits", { path: m.path })) as { a: TraceResult; b: TraceResult };
        const comparison = compareTraces(r.a, r.b, "the flow as started", m.label);
        const name = ([p, a]: [string, string]) => `${p.split("/").pop()} ${a === "delete" ? "skipped" : "edited"}`;
        const others = Object.entries(m.edits).filter(([p]) => p !== m.path).map(name);
        const scope = m.path ? (others.length ? `Only ${m.label} is applied; ${others.join(", ")} ${others.length === 1 ? "is" : "are"} not.` : "") : others.length > 1 ? `All ${others.length} edits are applied.` : "";
        comparison.note = `${scope} Both versions ran from the start to the end; your debug run is still paused where it was.`.trim();
        // Both runs share one description, so a swapped step's new code shows only through the edits made.
        for (const st of comparison.steps) {
          if (m.path && m.path !== st.path) continue;
          if (m.edits[st.path] === "replace") st.structural.push("code");
          if (m.edits[st.path] === "delete" && st.status === "not taken") st.status = "removed";
        }
        post({ type: "compare", comparison });
      } catch (e) {
        post({ type: "compare", comparison: null, error: (e as Error).message });
      }
      break;
    }
    case "skip":
    case "reloadStep":
    case "restore":
      try {
        const command = m.type === "skip" ? "decider.skip" : m.type === "restore" ? "decider.restore" : "decider.reloadStep";
        const r = (await s?.customRequest(command, { path: m.path })) as { diff?: string[]; formula?: string | null } | undefined;
        if (m.type !== "skip" && r) post({ type: "edited", path: m.path, diff: r.diff ?? [], formula: r.formula ?? null, restored: m.type === "restore" });
      } catch (e) {
        void vscode.window.showErrorMessage(`Couldn't ${m.type === "skip" ? "skip" : m.type === "restore" ? "restore" : "swap in"} ${m.path.split("/").pop()}: ${(e as Error).message}`);
      }
      break;
    case "restartWith":
      await s?.customRequest("decider.restartWith", { params: m.params });
      break;
    case "whatIf": {
      if (!shown) return;
      await compare(
        { label: "current params", file: shown.file, pipeline: shown.pipeline },
        { label: m.label, file: shown.file, pipeline: shown.pipeline, params: m.params, overrides: m.overrides, row: m.row, forces: m.forces },
      );
      break;
    }
    case "setControls":
      controls = m.controls;
      await s?.customRequest("decider.setControls", controls);
      break;
    case "compareForces":
      if (shown) await compare({ ...shown, ...m.a }, { ...shown, ...m.b }, true);
      break;
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
    case "layout":
      // Only the two-group layout the panel set up; a user's own arrangement is left alone.
      if (vscode.window.tabGroups.all.length === 2 && GraphPanel.current?.column === vscode.ViewColumn.Two)
        await vscode.commands.executeCommand("vscode.setEditorLayout", { orientation: 0, groups: [{ size: m.wide ? 0.3 : 0.5 }, { size: m.wide ? 0.7 : 0.5 }] });
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
/** "Run to" breakpoints: gone once the run pauses there or the session ends, like run to cursor. */
let runToBreakpoints: vscode.FunctionBreakpoint[] = [];

function clearRunTo(pausedAt?: string) {
  const done = runToBreakpoints.filter((b) => pausedAt === undefined || b.functionName === pausedAt);
  if (!done.length) return;
  vscode.debug.removeBreakpoints(done);
  runToBreakpoints = runToBreakpoints.filter((b) => !done.includes(b));
}

async function runTo(nodePath: string) {
  const own = vscode.debug.breakpoints.some((b) => b instanceof vscode.FunctionBreakpoint && b.functionName === nodePath);
  if (!own) {
    const bp = new vscode.FunctionBreakpoint(nodePath);
    runToBreakpoints.push(bp);
    vscode.debug.addBreakpoints([bp]);
  }
  const s = deciderSession();
  if (s) await s.customRequest("continue", { threadId: 1 });
  else if (shown)
    await vscode.debug.startDebugging(undefined, { type: "decider", request: "launch", name: `decider: ${shown.pipeline}`, program: shown.file, pipeline: shown.pipeline, stopOnEntry: false, internalConsoleOptions: "neverOpen" });
}

async function compare(a: Side, b: Side, forced = false) {
  post({ type: "tab", tab: "compare" });
  post({ type: "compare", comparison: null, busy: `Running ${a.label} and ${b.label}…` });
  try {
    const comparison = await runComparison(a, b, pythonCommand(), path.dirname(b.file));
    if (a.params || b.params) comparison.paramsDocs = { a: a.params ?? {}, b: b.params ?? {} };
    comparison.forced = forced || !!(a.forces?.length || b.forces?.length);
    const rows = [...(a.forces ?? []), ...(b.forces ?? [])].map((f) => f.row).filter((r): r is number => r !== null && r !== undefined);
    if (rows.length) comparison.forcedRows = [...new Set(rows)];
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
  await compare({ label: pick.label, file: path.join(tree, path.relative(root, file)), pipeline }, { label: "your uncommitted changes", file, pipeline });
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
    const flash = vscode.window.createTextEditorDecorationType({ isWholeLine: true, backgroundColor: new vscode.ThemeColor("editor.findMatchHighlightBackground") });
    editor.setDecorations(flash, [new vscode.Range(pos, pos)]);
    setTimeout(() => flash.dispose(), 2500);
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
