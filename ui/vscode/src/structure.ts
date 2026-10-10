import * as vscode from "vscode";
import { walk, type CallNodeJson, type IRNodeJson } from "@decikit/ui";
import { userCode } from "./debugging";
import { walkItems, type Item } from "./outline";

export const TOOLBOX_MIME = "application/vnd.code.tree.decikit.toolbox";

export interface NodeType {
  kind: "step" | "branch" | "loop" | "flow" | "subflow";
  label: string;
  icon: vscode.ThemeIcon;
  detail: string;
}

const icon = (id: string, color?: string) => new vscode.ThemeIcon(id, color ? new vscode.ThemeColor(color) : undefined);

export const NODE_TYPES: NodeType[] = [
  { kind: "step", label: "Step", icon: icon("symbol-function", "charts.green"), detail: "A Python function, or a scorecard, table or tree document" },
  { kind: "branch", label: "Branch", icon: icon("git-compare", "charts.red"), detail: "Rows where a condition holds run `then`; the rest run `else` or pass through" },
  { kind: "loop", label: "Loop", icon: icon("sync", "charts.orange"), detail: "Repeat steps per row while a condition holds" },
  { kind: "flow", label: "Flow", icon: icon("symbol-namespace", "charts.blue"), detail: "A named group of steps, written inline" },
  { kind: "subflow", label: "Subflow", icon: icon("file-symlink-file", "charts.purple"), detail: "A shared flows/<name>.yaml, by reference" },
];

const ICONS: Record<string, vscode.ThemeIcon> = {
  step: NODE_TYPES[0].icon,
  scalar: NODE_TYPES[0].icon,
  row: icon("list-tree", "charts.green"),
  frame: icon("table", "charts.green"),
  branch: NODE_TYPES[1].icon,
  loop: NODE_TYPES[2].icon,
  flow: NODE_TYPES[3].icon,
  subflow: NODE_TYPES[4].icon,
  condition: icon("symbol-boolean", "charts.yellow"),
  then: icon("check"),
  else: icon("close"),
};
const PAUSED = icon("debug-stackframe", "debugIcon.pauseForeground");
const BREAKPOINT = icon("debug-breakpoint", "debugIcon.breakpointForeground");

/** The Structure view: the shown flow's nodes, marked with breakpoints and where the debugger is paused. */
export class StructureProvider implements vscode.TreeDataProvider<Item>, vscode.TreeDragAndDropController<Item> {
  readonly dropMimeTypes = [TOOLBOX_MIME];
  readonly dragMimeTypes: string[] = [];
  private changed = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this.changed.event;
  private root?: Item;
  private parents = new Map<string, Item>();
  private ir = new Map<string, IRNodeJson>();
  private paused?: string;
  private marked = new Set<string>();

  constructor(private onDrop: (kind: NodeType["kind"], target: Item | undefined) => void) {}

  set(root: Item | undefined, ir: IRNodeJson | undefined) {
    this.root = root;
    this.parents.clear();
    if (root) walkItems(root, (i, parent) => parent && this.parents.set(i.id, parent));
    this.ir.clear();
    if (ir) walk(ir, (n) => this.ir.set(n.path, n));
    this.changed.fire();
  }

  setPaused(path: string | undefined) {
    this.paused = path;
    this.changed.fire();
  }

  setBreakpoints(paths: Set<string>) {
    this.marked = paths;
    this.changed.fire();
  }

  /** The first item for a node path. */
  find(path: string): Item | undefined {
    let found: Item | undefined;
    if (this.root) walkItems(this.root, (i) => void (i.path === path && i.kind !== "arm" && (found ??= i)));
    return found;
  }

  /** The described step behind an item, if it is one. */
  call(item: Item): CallNodeJson | undefined {
    const n = item.kind === "arm" ? undefined : this.ir.get(item.path);
    return n?.kind === "call" ? n : undefined;
  }

  getTreeItem(item: Item): vscode.TreeItem {
    const call = this.call(item);
    const t = new vscode.TreeItem(item.name, item.children.length ? vscode.TreeItemCollapsibleState.Expanded : vscode.TreeItemCollapsibleState.None);
    t.id = item.id;
    t.description = item.detail ?? (call ? `→ ${call.outputs?.join(", ") ?? "?"}` : undefined);
    t.tooltip = new vscode.MarkdownString(
      [`**${item.path || item.name}**`, call?.doc, call && `reads \`${call.inputs?.join(", ") ?? "?"}\` · writes \`${call.outputs?.join(", ") ?? "?"}\``]
        .filter(Boolean).join("\n\n"),
    );
    const kind = item.kind === "arm" ? item.name : item.kind === "flow" && item.into && item.into.file !== item.file ? "subflow" : call?.callKind ?? item.kind;
    t.iconPath = item.kind !== "arm" && this.paused === item.path ? PAUSED : call && this.marked.has(item.path) ? BREAKPOINT : ICONS[kind];
    t.contextValue = [
      call && userCode(call) && "code",
      item.yamlPath?.length && item.kind !== "condition" && "edit",
      item.into && "into",
      item.kind === "branch" && item.yamlPath && "branch",
    ].filter(Boolean).join(" ");
    t.command = { command: "decikit.select", title: "Show", arguments: [item] };
    return t;
  }

  getChildren(item?: Item): Item[] {
    if (!item) return this.root ? [this.root] : [];
    return item.children;
  }

  getParent(item: Item): Item | undefined {
    return this.parents.get(item.id);
  }

  handleDrop(target: Item | undefined, data: vscode.DataTransfer) {
    const kind = data.get(TOOLBOX_MIME)?.value;
    if (kind) this.onDrop(kind, target);
  }
}

/** The Toolbox view: node types to drag onto the Structure, or click to insert. */
export class ToolboxProvider implements vscode.TreeDataProvider<NodeType>, vscode.TreeDragAndDropController<NodeType> {
  readonly dragMimeTypes = [TOOLBOX_MIME];
  readonly dropMimeTypes: string[] = [];

  getTreeItem(n: NodeType): vscode.TreeItem {
    const t = new vscode.TreeItem(n.label);
    t.iconPath = n.icon;
    t.description = n.detail;
    t.tooltip = `${n.detail}\n\nDrag onto the Structure, or press + to insert after the selected node.`;
    return t;
  }

  getChildren(): NodeType[] {
    return NODE_TYPES;
  }

  handleDrag(source: readonly NodeType[], data: vscode.DataTransfer) {
    data.set(TOOLBOX_MIME, new vscode.DataTransferItem(source[0].kind));
  }
}
