import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import type { Action } from "./agentActions";

const MAX_REQUEST = 1 << 20;
const HANDLER_TIMEOUT = 60_000;

/** The default home of every window's socket and token: the directory decider's MCP server looks in. */
export const editorDir = () => path.join(os.homedir(), ".decider", "editor");

/** The file name the MCP server derives from a folder: a hash of its real path. */
export const folderHash = (folder: string) => crypto.createHash("sha256").update(fs.realpathSync(folder)).digest("hex").slice(0, 16);

/**
 * Serve one folder's actions to the MCP server on `<dir>/<hash>.sock`, guarded by a fresh
 * token in `<hash>.token`. A socket left by a crashed window is replaced; one a live window
 * answers on is left alone. The returned function stops serving and removes the files.
 */
export async function listen(folder: string, handle: (a: Action) => Promise<unknown>, dir = editorDir()): Promise<() => void> {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  fs.chmodSync(dir, 0o700);
  const real = fs.realpathSync(folder);
  const name = folderHash(real);
  const sock = path.join(dir, `${name}.sock`);
  const files = [sock, path.join(dir, `${name}.token`), path.join(dir, `${name}.folder`)];
  if (await alive(sock)) return () => {};
  fs.rmSync(sock, { force: true });

  const token = crypto.randomBytes(32).toString("hex");
  const server = net.createServer((conn) => serve(conn, token, handle));
  // ponytail: untested on Windows. `net` listens on a named pipe there, named as the Python client opens it;
  // the empty .sock file is only there so the client's existence check passes.
  const address = process.platform === "win32" ? "\\\\.\\pipe\\" + sock.replace(/\//g, "\\") : sock;
  await new Promise<void>((resolve, reject) => server.once("error", reject).listen(address, resolve));
  if (process.platform === "win32") fs.writeFileSync(sock, "");
  else fs.chmodSync(sock, 0o600);
  fs.writeFileSync(files[1], token, { mode: 0o600 });
  fs.writeFileSync(files[2], real, { mode: 0o600 });
  return () => {
    server.close();
    for (const f of files) fs.rmSync(f, { force: true });
  };
}

function alive(sock: string): Promise<boolean> {
  if (!fs.existsSync(sock)) return Promise.resolve(false);
  if (process.platform === "win32") return Promise.resolve(true);
  return new Promise((resolve) => {
    const c = net.connect(sock);
    c.once("connect", () => (c.destroy(), resolve(true)));
    c.once("error", () => resolve(false));
  });
}

function serve(conn: net.Socket, token: string, handle: (a: Action) => Promise<unknown>) {
  let buf = "";
  const reply = (r: object) => conn.end(JSON.stringify(r) + "\n");
  conn.on("error", () => {});
  conn.on("data", async (chunk) => {
    buf += chunk;
    if (buf.length > MAX_REQUEST) return reply({ ok: false, error: "request too large" });
    const end = buf.indexOf("\n");
    if (end < 0) return;
    conn.removeAllListeners("data");
    try {
      const req = JSON.parse(buf.slice(0, end));
      const given = Buffer.from(String(req.token ?? ""));
      const want = Buffer.from(token);
      if (given.length !== want.length || !crypto.timingSafeEqual(given, want)) return reply({ ok: false, error: "unauthorized" });
      let timer: NodeJS.Timeout;
      const timeout = new Promise<never>((_, no) => (timer = setTimeout(() => no(new Error("the editor did not answer in time")), HANDLER_TIMEOUT)));
      reply({ ok: true, result: await Promise.race([handle(req.action), timeout]).finally(() => clearTimeout(timer)) });
    } catch (e) {
      reply({ ok: false, error: (e as Error).message });
    }
  });
}
