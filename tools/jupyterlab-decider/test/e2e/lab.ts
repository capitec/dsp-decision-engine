// Drives a real JupyterLab with Playwright's Chromium: start the server, open the page, take screenshots.
import { spawn, type ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as net from "node:net";
import * as path from "node:path";
import { chromium, type Browser, type Page } from "playwright-core";

const ROOT = path.resolve(__dirname, "..", "..");
export const SHOTS = path.join(ROOT, "test", "e2e", "shots");
export const EXAMPLES = path.resolve(ROOT, "..", "vscode-decider", "examples");

export interface Lab {
  page: Page;
  /** The folder JupyterLab serves: a fresh copy of the example flows. */
  dir: string;
  shot(name: string): Promise<string>;
  /** Run a JupyterLab command, as the menus and palette do. */
  command(id: string, args?: Record<string, unknown>): Promise<void>;
  close(): Promise<void>;
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, "127.0.0.1", () => {
      const port = (srv.address() as net.AddressInfo).port;
      srv.close(() => resolve(port));
    });
    srv.on("error", reject);
  });
}

export async function launch(): Promise<Lab> {
  const dir = path.join(ROOT, "test", "e2e", "work");
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(path.join(dir, "settings", "@jupyterlab", "apputils-extension"), { recursive: true });
  fs.copyFileSync(path.join(EXAMPLES, "loan.py"), path.join(dir, "loan.py"));
  // No news prompt covering the page in screenshots.
  fs.writeFileSync(path.join(dir, "settings", "@jupyterlab", "apputils-extension", "notification.jupyterlab-settings"), JSON.stringify({ fetchNews: "false" }));
  fs.mkdirSync(SHOTS, { recursive: true });
  const port = await freePort();
  const token = Math.random().toString(36).slice(2);
  const server: ChildProcess = spawn(
    "uv",
    [
      "run", "--project", ROOT, "jupyter", "lab", "--no-browser", `--port=${port}`, `--IdentityProvider.token=${token}`,
      `--ServerApp.root_dir=${dir}`, `--LabApp.user_settings_dir=${path.join(dir, "settings")}`,
      `--LabApp.workspaces_dir=${path.join(dir, "workspaces")}`, "--expose-app-in-browser",
    ],
    { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
  );
  const log: string[] = [];
  server.stdout!.on("data", (d) => log.push(String(d)));
  server.stderr!.on("data", (d) => log.push(String(d)));
  const url = `http://127.0.0.1:${port}/lab?token=${token}`;
  for (let i = 0; ; i++) {
    if (i > 120) throw new Error(`JupyterLab didn't start:\n${log.join("")}`);
    if (await fetch(`http://127.0.0.1:${port}/api/status?token=${token}`).then((r) => r.ok, () => false)) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  const browser: Browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.on("pageerror", (e) => console.error(`page error: ${e.stack ?? e.message}`));
  await page.goto(url);
  await page.locator(".jp-DirListing-item", { hasText: "loan.py" }).waitFor({ timeout: 60_000 });
  return {
    page,
    dir,
    shot: async (name) => {
      const file = path.join(SHOTS, `${name}.png`);
      await page.screenshot({ path: file });
      return file;
    },
    command: (id, args = {}) =>
      page.evaluate(async ([id, args]) => {
        await (window as any).jupyterapp.commands.execute(id, args);
      }, [id, args] as const),
    close: async () => {
      await browser.close();
      // uv passes SIGTERM on, and the server shuts its kernels down before it exits.
      const exited = new Promise((r) => server.once("exit", r));
      server.kill("SIGTERM");
      await Promise.race([exited, new Promise((r) => setTimeout(r, 10_000))]);
    },
  };
}
