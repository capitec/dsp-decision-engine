import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";

/** The interpreter command: the `decider.python` setting, else ms-python's active one, else python3. */
export function pythonCommand(): string[] {
  const configured = vscode.workspace.getConfiguration("decider").get<string[]>("python") ?? [];
  if (configured.length) return configured;
  const py = vscode.extensions.getExtension("ms-python.python");
  const active = py?.isActive ? py.exports?.environments?.getActiveEnvironmentPath?.() : undefined;
  if (active?.path) return [active.path];
  return ["python3"];
}

/** Where the ms-python.debugpy extension keeps its bundled debugpy, so the bridge can import it. */
export function debugpyLibs(): string | undefined {
  const ext = vscode.extensions.getExtension("ms-python.debugpy");
  if (!ext) return undefined;
  const libs = path.join(ext.extensionPath, "bundled", "libs");
  return fs.existsSync(path.join(libs, "debugpy")) ? libs : undefined;
}
