import { createHash, randomBytes } from "node:crypto";
import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import * as readline from "node:readline";

/**
 * The per-window listener an MCP process reaches for editor-bound actions:
 * highlight, reveal and selection. One socket per workspace, named by a hash of
 * the workspace's real path (matching `decider/mcp/editor.py::ws_hash`), plus a
 * per-window secret token in an owner-only directory. A wrong token is refused;
 * the token blocks other same-user processes, the 0600 files block other users.
 */
export type EditorAction =
  | { kind: "highlight"; flow: string; nodes: string[] }
  | { kind: "reveal"; path: string; line: number | null }
  | { kind: "selection" };

export function workspaceHash(workspace: string): string {
  return createHash("sha256").update(fs.realpathSync.native(workspace)).digest("hex").slice(0, 16);
}

export function editorDir(): string {
  const dir = path.join(os.homedir(), ".decider", "editor");
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  fs.chmodSync(dir, 0o700);
  return dir;
}

export class EditorBridge {
  private server: net.Server;
  private socketPath: string;
  private tokenPath: string;
  readonly token: string;

  constructor(workspace: string, private dispatch: (action: EditorAction) => Promise<unknown>, dir?: string) {
    const root = dir ?? editorDir();
    const hash = workspaceHash(workspace);
    this.socketPath = path.join(root, `${hash}.sock`);
    this.tokenPath = path.join(root, `${hash}.token`);
    this.token = randomBytes(32).toString("hex");
    fs.writeFileSync(this.tokenPath, this.token, { mode: 0o600 });
    try { fs.unlinkSync(this.socketPath); } catch { /* no stale socket */ }
    this.server = net.createServer((conn) => this.handle(conn));
    this.server.on("error", (e) => console.error(`decider: editor bridge ${this.socketPath}: ${e.message}`));
    this.server.listen(this.socketPath);
  }

  private handle(conn: net.Socket) {
    const lines = readline.createInterface({ input: conn, crlfDelay: Infinity });
    lines.once("line", (line) => {
      lines.close();
      let msg: { token?: string; action?: EditorAction };
      try { msg = JSON.parse(line); } catch { return conn.end(JSON.stringify({ ok: false, error: "bad request" }) + "\n"); }
      if (msg.token !== this.token) return conn.end(JSON.stringify({ ok: false, error: "unauthorized" }) + "\n");
      if (!msg.action) return conn.end(JSON.stringify({ ok: false, error: "missing action" }) + "\n");
      void Promise.resolve(this.dispatch(msg.action)).then(
        (result) => conn.end(JSON.stringify({ ok: true, result }) + "\n"),
        (e) => conn.end(JSON.stringify({ ok: false, error: String(e) }) + "\n"),
      );
    });
  }

  dispose() {
    this.server.close();
    for (const p of [this.socketPath, this.tokenPath]) { try { fs.unlinkSync(p); } catch { /* already gone */ } }
  }
}
