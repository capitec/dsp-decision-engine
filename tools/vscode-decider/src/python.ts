import { spawn, spawnSync } from "node:child_process";
import * as readline from "node:readline";
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

/** The interpreter file the `decider.python` command runs (so `uv run python` works for debugpy), or `undefined` when unset. */
export function configuredExecutable(cwd: string): string | undefined {
  const configured = vscode.workspace.getConfiguration("decider").get<string[]>("python") ?? [];
  if (!configured.length) return undefined;
  const [exe, ...pre] = configured;
  const r = spawnSync(exe, [...pre, "-c", "import sys; print(sys.executable)"], { cwd, encoding: "utf8" });
  if (r.status !== 0) throw new Error(`decider.python ${configured.join(" ")} failed: ${r.stderr || r.error?.message}`);
  return r.stdout.trim();
}

/**
 * Ask a short-lived `decider._bridge` one question. Replies come back on
 * fd 3, so a `print()` in a step can't corrupt them.
 */
export function ask<T>(cwd: string, cmd: string, args: Record<string, unknown>): Promise<T> {
  const [exe, ...pre] = pythonCommand();
  const proc = spawn(exe, [...pre, "-m", "decider._bridge", "--fd", "3"], {
    cwd,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    stdio: ["pipe", "ignore", "pipe", "pipe"],
  });
  let stderr = "";
  proc.stderr!.on("data", (d) => (stderr += d));
  return new Promise<T>((resolve, reject) => {
    readline.createInterface({ input: proc.stdio[3] as NodeJS.ReadableStream }).once("line", (line) => {
      const m = JSON.parse(line);
      m.ok ? resolve(m.result) : reject(new Error(m.error));
    });
    proc.on("error", reject);
    proc.on("close", (code) => reject(new Error(stderr.trim().split("\n").at(-1) || `${exe} exited with code ${code}`)));
    proc.stdin!.end(JSON.stringify({ id: 1, cmd, ...args }) + "\n");
  });
}
