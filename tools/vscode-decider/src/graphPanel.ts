import * as vscode from "vscode";
import type { DescribeResult, FromWebview, ToWebview } from "./protocol";

/** The graph view: one React webview panel, fed the IR and the session position. */
export class GraphPanel {
  static current: GraphPanel | undefined;
  /** Messages sent while the panel is being opened. */
  private static early: ToWebview[] = [];
  private static opening: Promise<void> | undefined;
  private panel: vscode.WebviewPanel;
  private ready = false;
  private queue: ToWebview[] = [];

  /** Send to the panel, or hold the message until it has opened. */
  static post(m: ToWebview) {
    if (GraphPanel.current) GraphPanel.current.post(m);
    else GraphPanel.early.push(m);
  }

  static async show(ctx: vscode.ExtensionContext, describe: DescribeResult, onMessage: (m: FromWebview) => void): Promise<void> {
    if (!GraphPanel.current) {
      GraphPanel.opening ??= (async () => {
        let column = vscode.ViewColumn.Beside;
        if (vscode.window.tabGroups.all.length === 1) {
          // Beside the code the flow gets 60% of the width: at half or less, its text and tables don't fit.
          // Lay out the groups first and then open the panel in the second, so the order is certain.
          await vscode.commands.executeCommand("vscode.setEditorLayout", { orientation: 0, groups: [{ size: 0.4 }, { size: 0.6 }] });
          column = vscode.ViewColumn.Two;
        }
        GraphPanel.current = new GraphPanel(ctx, onMessage, column);
        // The flow first, then anything sent meanwhile: a describe resets the run state it would carry.
        GraphPanel.current.post({ type: "describe", describe });
        for (const m of GraphPanel.early.splice(0)) GraphPanel.current.post(m);
      })().finally(() => (GraphPanel.opening = undefined));
      await GraphPanel.opening;
      return;
    }
    GraphPanel.current.panel.reveal(undefined, true);
    GraphPanel.current.post({ type: "describe", describe });
  }

  private constructor(ctx: vscode.ExtensionContext, onMessage: (m: FromWebview) => void, column: vscode.ViewColumn) {
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
    w.onDidReceiveMessage((m: FromWebview) => {
      if (m.type === "ready") {
        this.ready = true;
        for (const q of this.queue.splice(0)) w.postMessage(q);
      } else onMessage(m);
    });
    this.panel.onDidDispose(() => (GraphPanel.current = undefined));
  }

  post(m: ToWebview) {
    if (this.ready) this.panel.webview.postMessage(m);
    else this.queue.push(m);
  }
}
