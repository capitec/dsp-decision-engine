import * as vscode from "vscode";
import type { DescribeResult, FromUI, ToUI } from "@decider/ui";

/** The flow panel: one React webview, fed the described flow and the selection from the host. */
export class GraphPanel {
  static current: GraphPanel | undefined;
  /** Messages sent while the panel is being opened. */
  private static early: ToUI[] = [];
  private static opening: Promise<void> | undefined;
  private panel: vscode.WebviewPanel;
  private ready = false;
  private queue: ToUI[] = [];

  /** Send to the panel, or hold the message until it has opened. */
  static post(m: ToUI) {
    if (GraphPanel.current) GraphPanel.current.post(m);
    else GraphPanel.early.push(m);
  }

  static async show(ctx: vscode.ExtensionContext, describe: DescribeResult, onMessage: (m: FromUI) => void): Promise<void> {
    await GraphPanel.ensureOpen(ctx, onMessage);
    GraphPanel.current!.post({ type: "describe", describe });
  }

  /** Open the panel (or reveal it) without describing a flow — for a view, like the experiment editor, that isn't one. */
  static async ensureOpen(ctx: vscode.ExtensionContext, onMessage: (m: FromUI) => void): Promise<void> {
    if (!GraphPanel.current) {
      GraphPanel.opening ??= (async () => {
        let column = vscode.ViewColumn.Beside;
        if (vscode.window.tabGroups.all.length === 1) {
          await vscode.commands.executeCommand("vscode.setEditorLayout", { orientation: 1, groups: [{ size: 0.3 }, { size: 0.7 }] });
          column = vscode.ViewColumn.Two;
        }
        GraphPanel.current = new GraphPanel(ctx, onMessage, column);
        for (const m of GraphPanel.early.splice(0)) GraphPanel.current.post(m);
      })().finally(() => (GraphPanel.opening = undefined));
      await GraphPanel.opening;
      return;
    }
    GraphPanel.current.panel.reveal(undefined, true);
  }

  private constructor(ctx: vscode.ExtensionContext, onMessage: (m: FromUI) => void, column: vscode.ViewColumn) {
    const dist = vscode.Uri.joinPath(ctx.extensionUri, "dist", "webview");
    this.panel = vscode.window.createWebviewPanel("decider.graph", "decider: flow", { viewColumn: column, preserveFocus: true }, {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [dist],
    });
    const w = this.panel.webview;
    const nonce = Math.random().toString(36).slice(2);
    w.html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${w.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<link rel="stylesheet" href="${w.asWebviewUri(vscode.Uri.joinPath(dist, "index.css"))}">
</head><body><div id="root"></div>
<script nonce="${nonce}" src="${w.asWebviewUri(vscode.Uri.joinPath(dist, "index.js"))}"></script></body></html>`;
    w.onDidReceiveMessage((m: FromUI) => {
      if (m.type === "ready") {
        this.ready = true;
        for (const q of this.queue.splice(0)) w.postMessage(q);
      } else onMessage(m);
    });
    this.panel.onDidDispose(() => {
      GraphPanel.current = undefined;
    });
  }

  post(m: ToUI) {
    if (this.ready) this.panel.webview.postMessage(m);
    else this.queue.push(m);
  }
}
