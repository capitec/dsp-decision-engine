import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";
import { isFlowFile } from "./outline";
import { groupFlows, projectRoot, type Group } from "./projects";

/** A pipeline the workspace defines: a flow file, or a pipeline in a `.py`. */
export interface FlowRef {
  file: string;
  pipeline?: string;
  label: string;
  line: number;
}

const EXCLUDE = "**/{node_modules,.venv,venv,.git,__pycache__,site-packages,dist,build,configs}/**";
const DECIDER_IMPORT = /^\s*(from|import)\s+decider\b/m;
// What `find_pipelines` offers: `def build`, a zero-arg `*_pipeline`/`*_flow` factory, or a module-level flow/dag/branch/loop.
const PIPELINE = /^(?:def\s+(build|\w*_pipeline|\w*_flow|pipeline|flow)\s*\(|(\w+)\s*(?::[^=\n]+)?=\s*(?:flow|dag|branch|loop)\()/gm;

export function isFlowPath(file: string): boolean {
  return /\.(ya?ml|json)$/i.test(file);
}

/** The pipelines a Python source defines, by the line each starts on (0-based). */
export function pythonPipelines(text: string): { name: string; line: number }[] {
  if (!DECIDER_IMPORT.test(text)) return [];
  return [...text.matchAll(PIPELINE)].map((m) => ({ name: m[1] ?? m[2], line: text.slice(0, m.index).split("\n").length - 1 }));
}

export async function discover(): Promise<FlowRef[]> {
  const [yamls, pys] = await Promise.all([
    vscode.workspace.findFiles("**/*.{yaml,yml}", EXCLUDE, 500),
    vscode.workspace.findFiles("**/*.py", EXCLUDE, 3000),
  ]);
  const text = async (u: vscode.Uri) => Buffer.from(await vscode.workspace.fs.readFile(u)).toString("utf8");
  const out: FlowRef[] = [];
  await Promise.all([
    ...yamls.map(async (u) => {
      const t = await text(u);
      if (isFlowFile(t)) out.push({ file: u.fsPath, label: /^flow:\s*(\S+)/m.exec(t)![1], line: 0 });
    }),
    ...pys.map(async (u) => {
      for (const p of pythonPipelines(await text(u))) out.push({ file: u.fsPath, pipeline: p.name, label: p.name, line: p.line });
    }),
  ]);
  return out.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line);
}

export const isDir = (p: string) => {
  try {
    return fs.statSync(p).isDirectory();
  } catch {
    return false;
  }
};

type SubGroup = { sub: FlowRef[]; root: string };
type Node = Group | SubGroup | FlowRef;
const isGroup = (n: Node): n is Group => "main" in n;
const isSubGroup = (n: Node): n is SubGroup => "sub" in n && !("main" in n);

/** The Flows view: the workspace's pipelines by project; picking one shows it. */
export class FlowsProvider implements vscode.TreeDataProvider<Node> {
  private changed = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changed.event;
  private flows?: Promise<Group[]>;

  refresh() {
    this.flows = undefined;
    this.changed.fire();
  }

  getTreeItem(n: Node): vscode.TreeItem {
    if (isGroup(n)) {
      const item = new vscode.TreeItem(n.root ? path.basename(n.root) : "Outside a project", vscode.TreeItemCollapsibleState.Expanded);
      const rel = n.root && vscode.workspace.asRelativePath(n.root);
      item.description = rel && rel !== item.label ? rel : undefined;
      item.iconPath = new vscode.ThemeIcon("folder");
      return item;
    }
    if (isSubGroup(n)) return new vscode.TreeItem(`Subflows (${n.sub.length})`, vscode.TreeItemCollapsibleState.Collapsed);
    const yaml = isFlowPath(n.file);
    const item = new vscode.TreeItem(n.label);
    item.contextValue = "flow";
    const root = projectRoot(n.file, isDir);
    item.description = root ? path.relative(root, n.file) : vscode.workspace.asRelativePath(n.file);
    item.iconPath = new vscode.ThemeIcon(yaml ? "symbol-namespace" : "symbol-method", new vscode.ThemeColor(yaml ? "charts.blue" : "charts.green"));
    item.tooltip = `${n.file}${n.pipeline ? ` · ${n.pipeline}` : ""}`;
    item.command = { command: "decider.showFlow", title: "Visualise flow", arguments: [vscode.Uri.file(n.file), n.pipeline] };
    return item;
  }

  async getChildren(n?: Node): Promise<Node[]> {
    if (!n) return (this.flows ??= discover().then((f) => groupFlows(f, (file) => projectRoot(file, isDir))));
    if (isGroup(n)) return n.sub.length && n.root ? [...n.main, { sub: n.sub, root: n.root }] : [...n.main, ...n.sub];
    if (!isSubGroup(n)) return [];
    return n.sub;
  }
}

/** Visualise and Debug above each pipeline: the first line of a flow file, or the line a Python one starts on. */
export class FlowCodeLens implements vscode.CodeLensProvider {
  provideCodeLenses(doc: vscode.TextDocument): vscode.CodeLens[] {
    const text = doc.getText();
    const found = isFlowPath(doc.fileName) ? (isFlowFile(text) ? [{ name: undefined, line: 0 }] : []) : pythonPipelines(text);
    return found.flatMap(({ name, line }) => {
      const range = new vscode.Range(line, 0, line, 0);
      const args = [doc.uri, name];
      return [
        new vscode.CodeLens(range, { title: "$(type-hierarchy) Visualise flow", command: "decider.showFlow", arguments: args }),
        new vscode.CodeLens(range, { title: "$(debug-alt) Debug flow", command: "decider.debugFlow", arguments: args }),
      ];
    });
  }
}
