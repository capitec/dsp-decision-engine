import * as path from "node:path";
import * as vscode from "vscode";
import { datasetLabel, datasetPage, isUnder, projectRoots, type Columns, type Dataset } from "./datasetFormat";
import { debugFlow } from "./debugging";
import { discover, isDir } from "./flows";
import { projectRoot } from "./projects";
import { ask } from "./python";

type Project = Dataset & { cwd: string };
type Node = Project | { project: string } | { column: [string, string] } | { message: string };
const EXCLUDE = "**/{node_modules,.venv,venv,.git,__pycache__,site-packages,dist,build}/**";
const cwd = () => vscode.workspace.workspaceFolders?.[0].uri.fsPath ?? process.cwd();

// The flow the Structure view shows: debugging a dataset uses it without asking.
let shown: { file: string; pipeline?: string } | undefined;

/** The Datasets view: what the project's datasets folder offers, with each one's columns underneath. */
export class DatasetsProvider implements vscode.TreeDataProvider<Node>, vscode.Disposable {
  private changed = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changed.event;
  private watchers = new Map<string, vscode.FileSystemWatcher>();
  private followed?: string;
  view?: vscode.TreeView<Node>;

  /** Show the datasets of the project owning `file`; all projects when it has none. */
  follow(file: string | undefined, pipeline?: string) {
    shown = file ? { file, pipeline } : undefined;
    const root = file && projectRoot(file, isDir);
    if (root === this.followed) return;
    this.followed = root;
    this.refresh();
  }

  refresh() {
    this.dispose();
    this.changed.fire();
  }

  dispose() {
    for (const w of this.watchers.values()) w.dispose();
    this.watchers.clear();
  }

  private watch(root: string) {
    if (this.watchers.has(root)) return;
    const w = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(vscode.Uri.file(root), "**"));
    for (const on of [w.onDidCreate, w.onDidChange, w.onDidDelete]) on(() => this.refresh());
    this.watchers.set(root, w);
  }

  private async datasets(cwd: string): Promise<Node[]> {
    try {
      const r = await ask<{ root: string; datasets: Dataset[] }>(cwd, "datasets", {});
      this.watch(r.root);
      return r.datasets.map((d) => ({ ...d, cwd }));
    } catch (e) {
      return [{ message: (e as Error).message.split("\n")[0] }];
    }
  }

  getTreeItem(n: Node): vscode.TreeItem {
    if ("message" in n) return new vscode.TreeItem(n.message);
    if ("project" in n) {
      const item = new vscode.TreeItem(vscode.workspace.asRelativePath(n.project), vscode.TreeItemCollapsibleState.Expanded);
      item.iconPath = new vscode.ThemeIcon("folder");
      return item;
    }
    if ("column" in n) {
      const item = new vscode.TreeItem(n.column[0]);
      item.description = n.column[1];
      return item;
    }
    const { description, contextValue } = datasetLabel(n);
    const item = new vscode.TreeItem(n.name, n.error ? vscode.TreeItemCollapsibleState.None : vscode.TreeItemCollapsibleState.Collapsed);
    item.description = description;
    item.contextValue = contextValue;
    item.iconPath = new vscode.ThemeIcon(n.error ? "error" : "database");
    item.tooltip = n.error ?? n.path ?? n.name;
    if (!n.error) item.command = { command: "decikit.previewDataset", title: "Preview", arguments: [n] };
    return item;
  }

  async getChildren(n?: Node): Promise<Node[]> {
    try {
      if (!n) {
        if (this.view) this.view.description = this.followed && path.basename(this.followed);
        if (this.followed) return this.datasets(this.followed);
        const files = await vscode.workspace.findFiles("**/configs/**", EXCLUDE, 2000);
        const roots = projectRoots(files.map((u) => u.fsPath));
        if (roots.length > 1) return roots.map((project) => ({ project }));
        return this.datasets(roots[0] ?? cwd());
      }
      if ("project" in n) return this.datasets(n.project);
      if (!("name" in n)) return [];
      const s = await ask<{ columns: Columns; row_count: number | null }>(n.cwd, "dataset_schema", { name: n.name });
      return [...(s.row_count == null ? [] : [{ message: `${s.row_count} rows` }]), ...s.columns.map((column) => ({ column }))];
    } catch (e) {
      return [{ message: (e as Error).message.split("\n")[0] }];
    }
  }
}

const previews = new Map<string, vscode.WebviewPanel>();

/** Show a dataset's first rows in a webview, where each can be sorted, filtered and sent to the debugger. */
export async function previewDataset(d: Project): Promise<void> {
  const key = `${d.cwd}\0${d.name}`;
  const open = previews.get(key);
  if (open) return open.reveal();
  const panel = vscode.window.createWebviewPanel("decikit.datasetPreview", `${d.name} (preview)`, vscode.ViewColumn.Active, { enableScripts: true });
  previews.set(key, panel);
  panel.onDidDispose(() => previews.delete(key));
  try {
    const [h, s] = await Promise.all([
      ask<{ columns: Columns; rows: Record<string, unknown>[] }>(d.cwd, "dataset_head", { name: d.name, n: 1000 }),
      ask<{ row_count: number | null }>(d.cwd, "dataset_schema", { name: d.name }),
    ]);
    panel.webview.html = datasetPage(Math.random().toString(36).slice(2));
    panel.webview.onDidReceiveMessage((m: { type: string; row?: number }) => {
      if (m.type === "ready") panel.webview.postMessage({ type: "dataset", name: d.name, columns: h.columns, rows: h.rows, rowCount: s.row_count });
      else if (m.type === "debugRecord") void debugWithDataset(d, m.row);
    });
  } catch (e) {
    panel.webview.html = `<p>${(e as Error).message.replace(/[&<]/g, (c) => `&#${c.charCodeAt(0)};`)}</p>`;
  }
}

/** Debug the shown flow on the dataset, or pick one of its project's flows when none is shown. */
export async function debugWithDataset(d: Project, row?: number): Promise<void> {
  // A dataset name resolves from the flow's project, so a flow shown from another project can't use it.
  if (shown && isUnder(d.cwd, shown.file)) return debugFlow(shown.file, shown.pipeline, d.name, row);
  const flows = (await discover()).filter((f) => isUnder(d.cwd, f.file));
  const pick = await vscode.window.showQuickPick(
    flows.map((f) => ({ label: f.label, description: vscode.workspace.asRelativePath(f.file), f })),
    { placeHolder: `Flow to debug on ${d.name}` },
  );
  if (pick) await debugFlow(pick.f.file, pick.f.pipeline, d.name, row);
}

export const revealDataset = (d: Dataset) =>
  d.path ? vscode.commands.executeCommand("revealInExplorer", vscode.Uri.file(d.path)) : undefined;
export const copyDatasetName = (d: Dataset) => vscode.env.clipboard.writeText(d.name);
