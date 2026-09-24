// Story helpers: each story opens its own VSCodium, and every screenshot is captioned in a manifest
// that a reviewer scores from the screenshots alone.
import * as fs from "node:fs";
import * as path from "node:path";
import { launch, type Codium } from "./codium";

/** Where screenshots go; each suite sets its own folder. */
export const out = { dir: "" };
export const FILE = "bank/pipeline.py";

export function writeManifest() {
  fs.writeFileSync(path.join(out.dir, "manifest.json"), JSON.stringify(manifest, null, 1));
}

export const manifest: { story: string; goal: string; shots: { file: string; caption: string }[] }[] = [];

export async function story(name: string, goal: string, body: (c: Codium, shot: (caption: string) => Promise<void>) => Promise<void>, folder?: string) {
  const entry = { story: name, goal, shots: [] as { file: string; caption: string }[] };
  manifest.push(entry);
  const c = await launch(folder);
  let n = 0;
  const shot = async (caption: string) => {
    const file = path.join(out.dir, `${name}-${String(++n).padStart(2, "0")}.jpg`);
    await c.page.waitForTimeout(500);
    await c.page.screenshot({ path: file, type: "jpeg", quality: 60, scale: "css" });
    entry.shots.push({ file, caption });
  };
  try {
    await body(c, shot);
  } catch (e) {
    await shot(`FAILED here: ${(e as Error).message.split("\n")[0]}`);
    throw e;
  } finally {
    await c.close();
  }
}

export const keys = async (c: Codium, combo: string, text?: string) => {
  await c.page.keyboard.press(combo);
  if (text !== undefined) {
    await c.page.keyboard.type(text, { delay: 20 });
    await c.page.locator(".quick-input-list .monaco-list-row").first().waitFor({ timeout: 15_000 }).catch(() => undefined);
    await c.page.waitForTimeout(300);
    await c.page.keyboard.press("Enter");
  }
  await c.page.waitForTimeout(400);
};
export const lens = (c: Codium, title: string) => c.page.locator(".codelens-decoration a", { hasText: title }).first();
export const wv = (c: Codium) => c.webview();
export const tab = (c: Codium, name: string) => wv(c).locator("header nav button", { hasText: name }).click();

/** Type into the graph's find box and take the first hit. */
export async function find(c: Codium, query: string) {
  const box = wv(c).locator('input[aria-label="Find a step"]');
  await box.fill("");
  await box.fill(query);
  await wv(c).locator(".find-hits li").first().waitFor({ timeout: 10_000 });
  await box.press("Enter");
}

/** Run to the selected step and wait for the pause. */
export async function runTo(c: Codium) {
  await wv(c).locator("aside button", { hasText: /^Run to/ }).first().click({ timeout: 10_000 });
  await wv(c).locator(".pause-banner:not(.pending)").waitFor({ timeout: 180_000 });
}

/** Open the flow file and draw the flow. */
export async function visualise(c: Codium) {
  await keys(c, "Control+P", FILE);
  await lens(c, "Visualise flow").waitFor({ timeout: 90_000 });
  await lens(c, "Visualise flow").click();
  await wv(c).locator("svg .node").first().waitFor({ timeout: 90_000 });
}
