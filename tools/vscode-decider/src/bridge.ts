import { spawn, type ChildProcess } from "node:child_process";
import * as net from "node:net";
import * as path from "node:path";
import * as readline from "node:readline";

export interface BridgeOptions {
  /** Interpreter command, e.g. ["uv", "run", "python"] or ["/usr/bin/python3"]. */
  python: string[];
  cwd?: string;
  /** Extra PYTHONPATH entries (the bundled debugpy, for `debugpyPort`). */
  pythonPath?: string[];
  debugpyPort?: number;
  onOutput?: (text: string, category: "stdout" | "stderr") => void;
}

export const BRIDGE_PY = path.resolve(__dirname, "..", "python", "bridge.py");

/**
 * One Python bridge process. Requests go down stdin as JSON lines; replies come
 * back on fd 3, so user `print()`s in steps go to stdout and cannot corrupt the
 * protocol.
 */
export class Bridge {
  private proc: ChildProcess;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  readonly exited: Promise<number | null>;

  constructor(opts: BridgeOptions) {
    const [exe, ...pre] = opts.python;
    const args = [...pre, BRIDGE_PY, "--fd", "3"];
    if (opts.debugpyPort) args.push("--debugpy", String(opts.debugpyPort));
    const env: NodeJS.ProcessEnv = { ...process.env, PYTHONUNBUFFERED: "1" };
    if (opts.pythonPath?.length) {
      env.PYTHONPATH = [...opts.pythonPath, process.env.PYTHONPATH ?? ""].filter(Boolean).join(path.delimiter);
    }
    this.proc = spawn(exe, args, { cwd: opts.cwd, env, stdio: ["pipe", "pipe", "pipe", "pipe"] });
    const replies = readline.createInterface({ input: this.proc.stdio[3] as NodeJS.ReadableStream });
    replies.on("line", (line) => this.onReply(line));
    for (const category of ["stdout", "stderr"] as const) {
      const stream = category === "stdout" ? this.proc.stdout! : this.proc.stderr!;
      readline.createInterface({ input: stream }).on("line", (l) => opts.onOutput?.(l + "\n", category));
    }
    this.exited = new Promise((resolve) => {
      this.proc.on("exit", (code) => {
        for (const p of this.pending.values()) p.reject(new Error(`bridge exited with code ${code}`));
        this.pending.clear();
        resolve(code);
      });
      this.proc.on("error", (e) => {
        for (const p of this.pending.values()) p.reject(e);
        this.pending.clear();
        resolve(null);
      });
    });
  }

  private onReply(line: string) {
    let msg: { id: number; ok: boolean; result?: unknown; error?: string };
    try {
      msg = JSON.parse(line);
    } catch {
      return;
    }
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    msg.ok ? p.resolve(msg.result) : p.reject(new Error(msg.error));
  }

  request<T = unknown>(cmd: string, args: Record<string, unknown> = {}): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.proc.stdin!.write(JSON.stringify({ id, cmd, ...args }) + "\n");
    });
  }

  async dispose(): Promise<void> {
    if (this.proc.exitCode !== null) return;
    this.request("exit").catch(() => undefined);
    const timer = setTimeout(() => this.proc.kill(), 1000);
    await this.exited;
    clearTimeout(timer);
  }
}

/** Spawn a bridge, run `fn`, and shut it down. */
export async function withBridge<T>(opts: BridgeOptions, fn: (b: Bridge) => Promise<T>): Promise<T> {
  const b = new Bridge(opts);
  try {
    return await fn(b);
  } finally {
    await b.dispose();
  }
}

/** A free local TCP port, for debugpy to listen on. */
export function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const port = (srv.address() as net.AddressInfo).port;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}
