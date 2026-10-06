import { App, type EditorMessage, type FromUI, type ToUI } from "@decider/ui";
import "@decider/ui/style.css";
import { createRoot } from "react-dom/client";
import "./theme.css";

type ToHost = FromUI | { type: "selected"; path: string | undefined };
declare function acquireVsCodeApi(): { postMessage(m: ToHost): void };
const vscode = acquireVsCodeApi();

const listen = (on: (m: ToUI) => void) => {
  const onMessage = (e: MessageEvent<ToUI>) => on(e.data);
  window.addEventListener("message", onMessage);
  return () => window.removeEventListener("message", onMessage);
};

// Stepping belongs to the Python debugger; the panel only opens source.
const can = new Set<EditorMessage>(["reveal"]);
const onSelectionChange = (path: string | undefined) => vscode.postMessage({ type: "selected", path });

createRoot(document.getElementById("root")!).render(
  <App send={(m) => vscode.postMessage(m)} listen={listen} can={can} showStructure={false} showBreakpointForm={false} onSelectionChange={onSelectionChange} />,
);
