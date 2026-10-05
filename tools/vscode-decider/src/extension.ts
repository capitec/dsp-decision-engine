import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";
import type { DescribeResult } from "@decider/ui";
import { currentBreakpoints, debugFlow, setBreakpointNodes, toggleBreakpoint, userCode, watchDebugger } from "./debugging";
import { copyDatasetName, DatasetsProvider, debugWithDataset, previewDataset, revealDataset } from "./datasets";
import { FlowCodeLens, FlowsProvider, isFlowPath } from "./flows";
import { GraphPanel, type FromPanel } from "./graphPanel";
import { pickNodeType, promptNode } from "./insert";
import { irOutline, yamlOutline, type Item } from "./outline";
import { ask } from "./python";
import { StructureProvider, ToolboxProvider, type NodeType } from "./structure";
import { deleteNode, insertNode, type Where } from "./yamlEdit";

/** The flow in the Structure view and the graph. */
interface Shown {
  file: string;
  pipeline?: string;
  describe?: DescribeResult;
  outline?: Item;
  error?: string;
}

export function activate(ctx: vscode.ExtensionContext) {
  let shown: Shown | undefined;
  let paused: string | undefined;
  let generation = 0;
  const components = new Map<string, Promise<string[]>>();

  const flows = new FlowsProvider();
  const datasets = new DatasetsProvider();
  const structure = new StructureProvider((kind, target) => void insertAt(kind, target));
  const tree = vscode.window.createTreeView("decider.structure", { treeDataProvider: structure, dragAndDropController: structure, showCollapseAll: true });
  const toolbox = new ToolboxProvider();
  const watcher = vscode.workspace.createFileSystemWatcher("**/*.{py,yaml,yml}", false, true, false);
  const debug = watchDebugger({
    ir: () => shown?.describe?.ir,
    show: (file, pipeline) => void (shown?.file !== file || shown.pipeline !== pipeline ? show(vscode.Uri.file(file), pipeline) : undefined),
    paused: (at) => {
      paused = at;
      structure.setPaused(at);
      GraphPanel.post({ type: "status", current: at ? { path: at, when: "before" } : null, finished: false, finishedPaths: [], visits: {}, record: null });
      const item = at && structure.find(at);
      if (!item) return;
      GraphPanel.post({ type: "select", path: at });
      if (tree.visible) tree.reveal(item, { select: false, focus: false });
    },
    breakpoints: pushBreakpoints,
  });

  const read = (file: string) =>
    vscode.workspace.textDocuments.find((d) => d.uri.fsPath === file)?.getText() ?? (fs.existsSync(file) ? fs.readFileSync(file, "utf8") : undefined);
  const ir = () => shown?.describe?.ir;
  const projectDir = (file: string) => (path.basename(path.dirname(file)) === "flows" ? path.dirname(path.dirname(file)) : path.dirname(file));

  function render() {
    if (shown && isFlowPath(shown.file)) {
      try {
        shown.outline = yamlOutline(shown.file, read);
      } catch (e) {
        shown.error = (e as Error).message;
      }
    } else if (shown?.describe) shown.outline = irOutline(shown.describe.ir, shown.describe.pipeline);
    structure.set(shown?.outline, ir());
    tree.message = shown?.error ? `⚠ ${shown.error}` : undefined;
  }

  async function describe() {
    const s = shown!;
    const g = ++generation;
    try {
      const d = await ask<DescribeResult>(path.dirname(s.file), "describe", { file: s.file, pipeline: s.pipeline });
      if (g !== generation) return;
      s.describe = d;
      s.error = undefined;
      GraphPanel.post({ type: "describe", describe: d });
      pushBreakpoints(currentBreakpoints(d.ir));
      render();
      debug.relocate();
      return;
    } catch (e) {
      if (g !== generation) return;
      const msg = (e as Error).message.split("\n")[0];
      s.error = /No module named 'decider'|ENOENT/.test(msg) ? `${msg}: set decider.python to an interpreter with decider installed` : msg;
    }
    render();
  }

  async function show(uri?: vscode.Uri, pipeline?: string, graph = true) {
    const file = uri?.fsPath ?? vscode.window.activeTextEditor?.document.uri.fsPath;
    if (!file) return;
    if (shown?.file !== file || shown.pipeline !== pipeline) shown = { file, pipeline };
    datasets.follow(file, pipeline);
    vscode.commands.executeCommand("setContext", "decider.flowShown", true);
    vscode.commands.executeCommand("setContext", "decider.yamlFlow", isFlowPath(file));
    if (graph) GraphPanel.show(ctx, onPanel);
    render();
    await describe();
  }

  function pushBreakpoints(paths: Set<string>) {
    structure.setBreakpoints(paths);
    GraphPanel.post({ type: "controls", controls: { forces: [], watches: [...paths].map((p) => ({ path: p })) } });
  }

  function onPanel(m: FromPanel) {
    if (m.type === "reveal") revealNode(m.path);
    else if (m.type === "setControls" && ir()) {
      const plain = m.controls.watches.filter((w) => w.path && w.name === undefined && w.condition === undefined && w.iteration === undefined);
      setBreakpointNodes(ir()!, new Set(plain.map((w) => w.path!)));
    } else if (m.type === "selected" && m.path && tree.visible) {
      const item = structure.find(m.path);
      if (item) tree.reveal(item, { select: true, focus: false });
    }
  }

  async function open(file: string, line: number | null) {
    const pos = new vscode.Position(Math.max((line ?? 1) - 1, 0), 0);
    await vscode.window.showTextDocument(vscode.Uri.file(file), { selection: new vscode.Range(pos, pos), preserveFocus: true, preview: true, viewColumn: vscode.ViewColumn.One });
  }

  function revealNode(nodePath: string) {
    const item = structure.find(nodePath);
    const call = item && structure.call(item);
    if (call && userCode(call)) return open(call.python!.file, call.python!.bodyLine ?? call.python!.line);
    if (item?.file) return open(item.file, item.line);
  }

  function names(): Promise<string[]> {
    const file = shown!.file;
    let found = components.get(file);
    if (!found) {
      found = ask<string[]>(path.dirname(file), "components", { file });
      components.set(file, found);
      found.catch(() => components.delete(file));
    }
    return found;
  }

  async function edit(file: string, change: (text: string) => string) {
    const doc = await vscode.workspace.openTextDocument(file);
    const e = new vscode.WorkspaceEdit();
    e.replace(doc.uri, new vscode.Range(0, 0, doc.lineCount, 0), change(doc.getText()));
    await vscode.workspace.applyEdit(e);
    await doc.save();
  }

  async function insert(item: Item, where: Where, kind?: NodeType["kind"]) {
    const target = where === "inside" ? item.into : item.yamlPath && item.file ? { file: item.file, yamlPath: item.yamlPath } : undefined;
    kind ??= await pickNodeType();
    if (!target || !kind || !shown) return;
    const node = await promptNode(kind, shown.file, names);
    if (node) await edit(target.file, (text) => insertNode(text, target.yamlPath, where, node, target.file.endsWith(".json")));
  }

  /** A drop from the Toolbox, or its + button: into a group, after anything else, at the end with no target. */
  async function insertAt(kind: NodeType["kind"], target: Item | undefined) {
    if (!shown?.outline) return;
    if (!target) return insert(shown.outline, "inside", kind);
    if (target.kind === "branch") {
      const arms = target.children.filter((c) => c.kind === "arm");
      const arm = await vscode.window.showQuickPick(arms.map((a) => ({ label: a.name, description: a.detail, arm: a })), { placeHolder: `Add to which arm of ${target.name}?` });
      return arm && insert(arm.arm, "inside", kind);
    }
    if (target.into) return insert(target, "inside", kind);
    if (target.yamlPath?.length && target.kind !== "condition") return insert(target, "after", kind);
    vscode.window.showInformationMessage("decider: drop onto a step, a flow, a loop or a branch arm");
  }

  ctx.subscriptions.push(
    tree,
    watcher,
    vscode.window.createTreeView("decider.flows", { treeDataProvider: flows }),
    vscode.window.createTreeView("decider.toolbox", { treeDataProvider: toolbox, dragAndDropController: toolbox }),
    datasets,
    (datasets.view = vscode.window.createTreeView("decider.datasets", { treeDataProvider: datasets })),
    vscode.commands.registerCommand("decider.refreshDatasets", () => datasets.refresh()),
    vscode.commands.registerCommand("decider.previewDataset", previewDataset),
    vscode.commands.registerCommand("decider.debugWithDataset", debugWithDataset),
    vscode.commands.registerCommand("decider.copyDatasetName", copyDatasetName),
    vscode.commands.registerCommand("decider.revealDataset", revealDataset),
    vscode.languages.registerCodeLensProvider([{ language: "python" }, { language: "yaml" }], new FlowCodeLens()),
    watcher.onDidCreate(() => flows.refresh()),
    watcher.onDidDelete(() => flows.refresh()),
    vscode.commands.registerCommand("decider.refreshFlows", () => flows.refresh()),
    vscode.commands.registerCommand("decider.showFlow", (uri?: vscode.Uri, pipeline?: string) =>
      show(uri ?? (shown && vscode.Uri.file(shown.file)), uri ? pipeline : shown?.pipeline),
    ),
    vscode.commands.registerCommand("decider.refreshStructure", () => shown && show(vscode.Uri.file(shown.file), shown.pipeline, false)),
    vscode.commands.registerCommand("decider.debugFlow", (arg?: vscode.Uri | { file: string; pipeline?: string }, pipeline?: string) => {
      const ref = arg instanceof vscode.Uri ? { file: arg.fsPath, pipeline } : arg ?? (shown ?? (vscode.window.activeTextEditor && { file: vscode.window.activeTextEditor.document.uri.fsPath }));
      return ref && debugFlow(ref.file, ref.pipeline);
    }),
    vscode.commands.registerCommand("decider.select", (item: Item) => {
      if (item.file) void open(item.file, item.line);
      GraphPanel.post({ type: "select", path: item.path });
    }),
    vscode.commands.registerCommand("decider.openCode", (item: Item) => revealNode(item.path)),
    vscode.commands.registerCommand("decider.toggleBreakpoint", (item: Item) => ir() && toggleBreakpoint(ir()!, item.path)),
    vscode.commands.registerCommand("decider.insertBefore", (item: Item) => insert(item, "before")),
    vscode.commands.registerCommand("decider.insertAfter", (item: Item) => insert(item, "after")),
    vscode.commands.registerCommand("decider.insertInside", (item: Item) => insert(item, "inside")),
    vscode.commands.registerCommand("decider.insertFromToolbox", (t: { kind: NodeType["kind"] }) => insertAt(t.kind, tree.selection[0])),
    vscode.commands.registerCommand("decider.deleteNode", async (item: Item) => {
      const ok = await vscode.window.showWarningMessage(`Delete ${item.kind} "${item.name}" from ${path.basename(item.file!)}?`, { modal: true }, "Delete");
      if (ok && item.file && item.yamlPath) await edit(item.file, (text) => deleteNode(text, item.yamlPath!, item.file!.endsWith(".json")));
    }),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (!shown || !doc.uri.fsPath.startsWith(projectDir(shown.file))) return;
      if (doc.languageId === "python") components.clear();
      if (doc.languageId === "python" || isFlowPath(doc.uri.fsPath)) void describe();
    }),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (shown && isFlowPath(shown.file) && isFlowPath(e.document.uri.fsPath) && e.document.uri.fsPath.startsWith(projectDir(shown.file))) render();
    }),
    ...debug.disposables,
  );
  // What the smoke test checks.
  return { state: () => ({ file: shown?.file, described: !!shown?.describe, error: shown?.error, paused }), datasets: () => datasets.getChildren() };
}

export function deactivate() {}
