import type { FromWebview } from "../src/protocol";

// The webview's one tie to VS Code: messages to the extension. Messages back arrive as window "message" events.
declare function acquireVsCodeApi(): { postMessage(m: FromWebview): void };
const vscode = acquireVsCodeApi();

export const send = (m: FromWebview) => vscode.postMessage(m);
