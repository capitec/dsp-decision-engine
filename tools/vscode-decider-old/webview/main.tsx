import { App, type EditorMessage, type FromUI, type ToUI } from "@decider/ui";
import "@decider/ui/style.css";
import { createRoot } from "react-dom/client";
import "./theme.css";

declare function acquireVsCodeApi(): { postMessage(m: FromUI): void };
const vscode = acquireVsCodeApi();

const listen = (on: (m: ToUI) => void) => {
  const onMessage = (e: MessageEvent<ToUI>) => on(e.data);
  window.addEventListener("message", onMessage);
  return () => window.removeEventListener("message", onMessage);
};

// Flow exploration only needs the host to open source; the debugger surfaces come later.
const can = new Set<EditorMessage>(["reveal"]);

createRoot(document.getElementById("root")!).render(<App send={(m) => vscode.postMessage(m)} listen={listen} can={can} />);
