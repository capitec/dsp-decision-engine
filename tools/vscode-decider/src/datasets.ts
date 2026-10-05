import * as path from "node:path";
import * as vscode from "vscode";
import { datasetLabel, escapeHtml, htmlTable, isUnder, projectRoots, type Columns, type Dataset } from "./datasetFormat";
import { debugFlow } from "./debugging";
import { discover, isDir } from "./flows";
import { projectRoot } from "./projects";
import { ask } from "./python";

type Project = Dataset & { cwd: string };
type Node = Project | { project: string } | { column: [string, string] } | { message: string };
const EXCLUDE = "**/{node_modules,.venv,venv,.git,__pycache__,site-packages,dist,build}/**";
const cwd = () => vscode.workspace.workspaceFolders?.[0].uri.fsPath ?? process.cwd();

/** The Datasets view: what the project's datasets folder offers, with each one's columns underneath. */
export class DatasetsProvider implements vscode.TreeDataProvider<Node>, vscode.Disposable {
  private changed = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changed.event;
  private watchers = new Map<string, vscode.FileSystemWatcher>();
  private followed?: string;
  view?: vscode.TreeView<Node>;

  /** Show the datasets of the project owning `file`; all projects when it has none. */
  follow(file: string | undefined) {
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
    if (!n.error) item.command = { command: "decider.previewDataset", title: "Preview", arguments: [n] };
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

/** Show a dataset's first rows in a webview. */
export async function previewDataset(d: Project): Promise<void> {
  const panel = vscode.window.createWebviewPanel("decider.datasetPreview", `${d.name} (preview)`, vscode.ViewColumn.Active, {});
  const style = `body{color:var(--vscode-editor-foreground);background:var(--vscode-editor-background);font-family:var(--vscode-editor-font-family);font-size:var(--vscode-editor-font-size)}
table{border-collapse:collapse}th,td{border:1px solid var(--vscode-panel-border);padding:2px 8px;text-align:left}
th{background:var(--vscode-editorWidget-background)}small{color:var(--vscode-descriptionForeground);font-weight:normal}`;
  try {
    const h = await ask<{ columns: Columns; rows: Record<string, unknown>[] }>(d.cwd, "dataset_head", { name: d.name, n: 100 });
    panel.webview.html = `<style>${style}</style>${htmlTable(h.columns, h.rows)}`;
  } catch (e) {
    panel.webview.html = `<style>${style}</style><p>${escapeHtml((e as Error).message)}</p>`;
  }
}

/** Pick a workspace flow and debug it on the dataset. */
export async function debugWithDataset(d: Project): Promise<void> {
  const flows = (await discover()).filter((f) => isUnder(d.cwd, f.file));
  const pick = await vscode.window.showQuickPick(
    flows.map((f) => ({ label: f.label, description: vscode.workspace.asRelativePath(f.file), f })),
    { placeHolder: `Flow to debug on ${d.name}` },
  );
  if (pick) await debugFlow(pick.f.file, pick.f.pipeline, d.name);
}

export const revealDataset = (d: Dataset) =>
  d.path ? vscode.commands.executeCommand("revealInExplorer", vscode.Uri.file(d.path)) : undefined;
export const copyDatasetName = (d: Dataset) => vscode.env.clipboard.writeText(d.name);
