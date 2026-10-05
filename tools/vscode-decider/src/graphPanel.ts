import * as vscode from "vscode";
import type { FromUI, ToUI } from "@decider/ui";

export type FromPanel = FromUI | { type: "selected"; path: string | undefined };

/** The flow panel: decider-ui's graph in a webview. Messages wait until it is ready. */
export class GraphPanel {
  private static current: GraphPanel | undefined;
  private panel: vscode.WebviewPanel;
  private ready = false;
  private queue: ToUI[] = [];

  /** Open (or reveal) the panel. */
  static show(ctx: vscode.ExtensionContext, onMessage: (m: FromPanel) => void): GraphPanel {
    if (GraphPanel.current) GraphPanel.current.panel.reveal(undefined, true);
    else GraphPanel.current = new GraphPanel(ctx, onMessage);
    return GraphPanel.current;
  }

  /** Send to the panel if it is open. */
  static post(m: ToUI) {
    GraphPanel.current?.post(m);
  }

  private constructor(ctx: vscode.ExtensionContext, onMessage: (m: FromPanel) => void) {
    const dist = vscode.Uri.joinPath(ctx.extensionUri, "dist", "webview");
    this.panel = vscode.window.createWebviewPanel("decider.graph", "decider: flow", { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true }, {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: [dist],
    });
    const w = this.panel.webview;
    const nonce = Math.random().toString(36).slice(2);
    w.html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${w.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}' ${w.cspSource};">
<link rel="stylesheet" href="${w.asWebviewUri(vscode.Uri.joinPath(dist, "index.css"))}">
</head><body><div id="root"></div>
<script type="module" nonce="${nonce}" src="${w.asWebviewUri(vscode.Uri.joinPath(dist, "index.js"))}"></script></body></html>`;
    w.onDidReceiveMessage((m: FromPanel) => {
      if (m.type === "ready") {
        this.ready = true;
        for (const q of this.queue.splice(0)) w.postMessage(q);
      } else onMessage(m);
    });
    this.panel.onDidDispose(() => (GraphPanel.current = undefined));
  }

  post(m: ToUI) {
    if (this.ready) this.panel.webview.postMessage(m);
    else this.queue.push(m);
  }
}
