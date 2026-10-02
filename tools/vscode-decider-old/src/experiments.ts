import * as vscode from "vscode";
import type { ExperimentDef } from "@decider/ui";

export interface ExperimentItem {
  uri: vscode.Uri;
  definition: ExperimentDef;
  hasResult: boolean;
}

/** The experiments view: every `experiment.json` in the workspace, with its scenario and metric counts. */
export class ExperimentsProvider implements vscode.TreeDataProvider<ExperimentItem> {
  private changed = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changed.event;
  private items: ExperimentItem[] = [];
  private watcher = vscode.workspace.createFileSystemWatcher("**/experiment.json");

  constructor() {
    this.watcher.onDidCreate(() => this.refresh());
    this.watcher.onDidChange(() => this.refresh());
    this.watcher.onDidDelete(() => this.refresh());
    void this.refresh();
  }

  dispose() {
    this.watcher.dispose();
  }

  async refresh(): Promise<void> {
    const files = await vscode.workspace.findFiles("**/experiment.json", "**/{node_modules,.venv,.git}/**", 200);
    const items = await Promise.all(
      files.map(async (uri): Promise<ExperimentItem | undefined> => {
        try {
          const definition = JSON.parse((await vscode.workspace.fs.readFile(uri)).toString()) as ExperimentDef;
          return { uri, definition, hasResult: await exists(resultUri(uri)) };
        } catch {
          return undefined; // mid-edit, not valid JSON yet
        }
      }),
    );
    this.items = items.filter((i): i is ExperimentItem => i !== undefined).sort((a, b) => a.uri.fsPath.localeCompare(b.uri.fsPath));
    this.changed.fire();
  }

  getTreeItem(item: ExperimentItem): vscode.TreeItem {
    const treeItem = new vscode.TreeItem(item.definition.name || vscode.workspace.asRelativePath(item.uri), vscode.TreeItemCollapsibleState.None);
    treeItem.description = `${item.definition.scenarios?.length ?? 0} scenario(s) · ${item.definition.metrics?.length ?? 0} metric(s)`;
    treeItem.tooltip = item.definition.description || vscode.workspace.asRelativePath(item.uri);
    treeItem.contextValue = item.hasResult ? "experiment.hasResult" : "experiment";
    treeItem.iconPath = new vscode.ThemeIcon("beaker");
    treeItem.command = { command: "decider.openExperiment", title: "Open editor", arguments: [item] };
    return treeItem;
  }

  getChildren(item?: ExperimentItem): ExperimentItem[] {
    return item ? [] : this.items;
  }
}

export function resultUri(experimentJson: vscode.Uri): vscode.Uri {
  return vscode.Uri.joinPath(experimentJson, "..", "result.json");
}

export async function exists(uri: vscode.Uri): Promise<boolean> {
  try {
    await vscode.workspace.fs.stat(uri);
    return true;
  } catch {
    return false;
  }
}
