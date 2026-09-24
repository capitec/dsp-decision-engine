import * as vscode from "vscode";
import { kindLabel, lastSegment, type Checkpoint, type DescribeResult, type IRNodeJson } from "@decider/ui";

/** The structural view: the IR tree, decorated with the running session's position. */
export class StructureProvider implements vscode.TreeDataProvider<IRNodeJson> {
  private changed = new vscode.EventEmitter<IRNodeJson | undefined>();
  readonly onDidChangeTreeData = this.changed.event;
  private describe?: DescribeResult;
  private current: Checkpoint | null = null;
  private finished = new Set<string>();
  private parents = new Map<string, IRNodeJson>();

  setDescribe(d: DescribeResult | undefined) {
    this.describe = d;
    this.parents.clear();
    this.current = null;
    this.finished.clear();
    this.changed.fire(undefined);
  }

  setStatus(current: Checkpoint | null, finishedPaths: string[]) {
    this.current = current;
    this.finished = new Set(finishedPaths);
    this.changed.fire(undefined);
  }

  getTreeItem(node: IRNodeJson): vscode.TreeItem {
    const item = new vscode.TreeItem(
      node.path === "" ? this.describe?.pipeline ?? "pipeline" : lastSegment(node.path),
      node.kind === "call" ? vscode.TreeItemCollapsibleState.None : vscode.TreeItemCollapsibleState.Expanded,
    );
    item.id = node.path || "<root>";
    // Only the writes fit beside the name; the tooltip has the full signature.
    item.description = node.kind === "call" ? `→ ${names(node.outputs)}` : node.kind;
    item.tooltip =
      `${node.path || this.describe?.pipeline}\n${node.source}` +
      (node.kind === "call" ? `\nreads ${names(node.inputs)}\nwrites ${names(node.outputs)}\nparams ${JSON.stringify(node.params)}` : "");
    item.iconPath = new vscode.ThemeIcon(
      this.current?.path === node.path ? "debug-stackframe" : this.finished.has(node.path) ? "pass" : ICONS[kindLabel(node)],
    );
    item.command = { command: "decider.selectStep", title: "Select in the flow", arguments: [node.path, node.file, node.line] };
    return item;
  }

  getChildren(node?: IRNodeJson): IRNodeJson[] {
    if (!this.describe) return [];
    if (!node) return [this.describe.ir];
    if (node.kind === "call") return [];
    for (const c of node.children) this.parents.set(c.path, node);
    return node.children;
  }

  getParent(node: IRNodeJson): IRNodeJson | undefined {
    return this.parents.get(node.path);
  }
}

const ICONS: Record<string, string> = {
  scalar: "symbol-function", row: "list-tree", frame: "table", sequence: "list-ordered", branch: "git-branch", loop: "sync",
};

function names(xs: string[] | null): string {
  return xs === null ? "?" : xs.join(", ");
}
