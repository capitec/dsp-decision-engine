import * as vscode from "vscode";
import type { DescribeResult, FromWebview, ToWebview } from "./protocol";

/** The graph view: one React webview panel, fed the IR and the session position. */
export class GraphPanel {
  static current: GraphPanel | undefined;
  private panel: vscode.WebviewPanel;
  private ready = false;
  private queue: ToWebview[] = [];

  static show(ctx: vscode.ExtensionContext, describe: DescribeResult, onMessage: (m: FromWebview) => void): GraphPanel {
    GraphPanel.current ??= new GraphPanel(ctx, onMessage);
    GraphPanel.current.panel.reveal(vscode.ViewColumn.Beside, true);
    GraphPanel.current.post({ type: "describe", describe });
    return GraphPanel.current;
  }

  private constructor(ctx: vscode.ExtensionContext, onMessage: (m: FromWebview) => void) {
    const dist = vscode.Uri.joinPath(ctx.extensionUri, "dist", "webview");
    this.panel = vscode.window.createWebviewPanel("decider.graph", "decider: flow", vscode.ViewColumn.Beside, {
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
