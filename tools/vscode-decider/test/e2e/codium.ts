// Drives a real VSCodium window with Playwright: launch, click, read webviews, take screenshots.
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { _electron, type ElectronApplication, type FrameLocator, type Page } from "playwright-core";

export const ROOT = path.resolve(__dirname, "..", "..");
export const SHOTS = path.join(ROOT, "test", "e2e", "shots");

export interface Codium {
  app: ElectronApplication;
  page: Page;
  /** The decider graph panel's content frame. */
  webview(): FrameLocator;
  /** A small JPEG of the whole window, for a person (or a model) to look at. */
  shot(name: string): Promise<string>;
  command(title: string): Promise<void>;
  close(): Promise<void>;
}

export async function launch(folder = path.join(ROOT, "examples"), size = { width: 1400, height: 860 }): Promise<Codium> {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), "decider-e2e-"));
  // A quiet profile: no git prompt, no empty secondary side bar, no welcome page.
  fs.mkdirSync(path.join(home, "user", "User"), { recursive: true });
  fs.writeFileSync(
    path.join(home, "user", "User", "settings.json"),
    JSON.stringify({
      "git.openRepositoryInParentFolders": "never",
      "workbench.secondarySideBar.defaultVisibility": "hidden",
      "workbench.startupEditor": "none",
      "workbench.tips.enabled": false,
      "chat.disableAIFeatures": true,
    }),
  );
  const env = Object.fromEntries(Object.entries(process.env).filter(([, v]) => v !== undefined)) as Record<string, string>;
  delete env.ELECTRON_RUN_AS_NODE;
  delete env.VSCODE_IPC_HOOK_CLI;
  const app = await _electron.launch({
    executablePath: process.env.VSCODE_EXE ?? "/usr/share/codium/codium",
    args: [
      folder,
      `--extensionDevelopmentPath=${ROOT}`,
      `--user-data-dir=${path.join(home, "user")}`,
      `--extensions-dir=${path.join(home, "extensions")}`,
      "--disable-workspace-trust",
      "--skip-welcome",
      "--skip-release-notes",
      "--disable-telemetry",
      "--disable-updates",
      "--new-window",
      // No window on the desktop; set E2E_HEADED=1 to watch.
      ...(process.env.E2E_HEADED ? [] : ["--ozone-platform=headless"]),
    ],
    env,
    timeout: 180_000,
  });
  const page = await app.firstWindow();
  await app.evaluate(({ BrowserWindow }, s) => BrowserWindow.getAllWindows()[0]?.setSize(s.width, s.height), size);
  await page.waitForSelector(".monaco-workbench", { timeout: 180_000 });
  fs.mkdirSync(SHOTS, { recursive: true });
  return {
    app,
    page,
    webview: () => page.frameLocator("iframe.webview.ready").frameLocator("#active-frame"),
    async shot(name) {
      const file = path.join(SHOTS, `${name}.jpg`);
      await page.screenshot({ path: file, type: "jpeg", quality: 55, scale: "css" });
      return file;
    },
    async command(title) {
      await page.keyboard.press("Control+Shift+P");
      await page.keyboard.type(title);
      await page.waitForTimeout(300);
      await page.keyboard.press("Enter");
    },
    async close() {
      await app.close();
      fs.rmSync(home, { recursive: true, force: true });
    },
  };
}
