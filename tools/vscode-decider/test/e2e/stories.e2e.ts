// User stories for a usability review: each runs in a fresh VSCodium and saves captioned screenshots.
// `pnpm test:stories` writes test/e2e/shots/stories/manifest.json for a reviewer to read.
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, describe, it } from "vitest";
import { launch, ROOT, SHOTS, type Codium } from "./codium";

const OUT = path.join(SHOTS, "stories");
const manifest: { story: string; goal: string; shots: { file: string; caption: string }[] }[] = [];
const LINE = { pipeline: 114, capByIncome: 36, treeLowScore: 101 };

async function story(name: string, goal: string, body: (c: Codium, shot: (caption: string) => Promise<void>) => Promise<void>, folder?: string) {
  const entry = { story: name, goal, shots: [] as { file: string; caption: string }[] };
  manifest.push(entry);
  const c = await launch(folder);
  let n = 0;
  const shot = async (caption: string) => {
    const file = path.join(OUT, `${name}-${String(++n).padStart(2, "0")}.jpg`);
    await c.page.waitForTimeout(400); // let the webview paint
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

const open = async (c: Codium, file = "loan.py") => {
  await c.page.locator(".explorer-folders-view .monaco-list-row", { hasText: file }).click();
  await c.page.locator(".editor-instance .monaco-editor .view-lines").first().waitFor();
  await c.page.locator(".editor-instance .monaco-editor .view-lines").first().click();
};
const toggleBreakpoint = async (c: Codium) => {
  const before = await c.page.locator(".codicon-debug-breakpoint").count();
  await c.page.keyboard.press("F9");
  await c.page.waitForFunction((n) => document.querySelectorAll(".codicon-debug-breakpoint").length > n, before);
};
// Quick Open with a line number works whatever has focus, unlike Ctrl+G.
const goTo = async (c: Codium, line: number, file = "loan.py") => {
  // Keys pressed while the flow panel has focus stay in its webview; focus the editor like a person would.
  await c.page.locator(".tabs-container .tab", { hasText: file }).first().click().catch(() => undefined);
  await c.page.keyboard.press("Control+P");
  await c.page.keyboard.type(`${file}:${line}`);
  await c.page.waitForTimeout(300);
  await c.page.keyboard.press("Enter");
  await c.page.waitForTimeout(300);
};
const lens = (c: Codium, title: string) => c.page.locator(".codelens-decoration a", { hasText: title }).first();
const tab = (c: Codium, name: string) => c.webview().locator("header nav button", { hasText: name }).click();

async function runToBreakpoint(c: Codium, line: number) {
  await open(c);
  await goTo(c, line);
  await toggleBreakpoint(c);
  await goTo(c, LINE.pipeline);
  await lens(c, "Run flow").click({ timeout: 60_000 });
  await c.page.locator(".quick-input-widget .monaco-list-row", { hasText: "SAMPLE" }).click();
  await c.webview().locator(".pause-banner", { hasText: "before the start" }).waitFor({ timeout: 60_000 });
  await c.page.keyboard.press("F5");
}

describe("user stories", () => {
  afterAll(() => {
    fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 1));
  });

  it("understand a flow I have not seen before", async () => {
    fs.rmSync(OUT, { recursive: true, force: true });
    fs.mkdirSync(OUT, { recursive: true });
    await story("1-understand", "A new team member opens a pipeline file and wants to understand its structure and what each step reads and writes, without running anything.", async (c, shot) => {
      await open(c);
      await goTo(c, LINE.pipeline);
      await lens(c, "Visualise flow").waitFor({ timeout: 60_000 });
      await shot("The pipeline file is open; the decider actions sit above `pipeline = ...`.");
      await lens(c, "Visualise flow").click();
      const wv = c.webview();
      await wv.locator("svg .node", { hasText: "cap_by_income" }).waitFor({ timeout: 30_000 });
      await shot("After clicking 'Visualise flow': the flow panel opens beside the code.");
      await wv.locator("svg .node", { hasText: "cap_by_income" }).click();
      await shot("After clicking the cap_by_income step in the graph.");
      await c.page.locator(".activitybar .action-item a[aria-label*='decider']").first().click().catch(() => undefined);
      await c.page.mouse.move(700, 450); // off the activity bar, so its hover label is gone
      await shot("The decider view in the activity bar: the structure as a tree.");
    });
  }, 180_000);

  it("find out why one application got its term", async () => {
    await story("2-debug-record", "A credit analyst runs the flow on sample applications, pauses at the income cap, focuses on the application with client_id 1 and wants to see what its term_cap was computed from, then see which path the risk tree took for it.", async (c, shot) => {
      await open(c);
      await goTo(c, LINE.pipeline);
      await lens(c, "Visualise flow").click({ timeout: 60_000 });
      const wv = c.webview();
      await wv.locator("svg .node", { hasText: "cap_by_income" }).click({ timeout: 30_000 });
      await shot("After 'Visualise flow' and clicking the cap_by_income step.");
      await wv.locator("aside button", { hasText: "Run to cap_by_income" }).click();
      await wv.locator(".pause-banner", { hasText: "cap_by_income" }).waitFor({ timeout: 60_000 });
      await shot("After clicking 'Run to cap_by_income': the flow runs on the sample and pauses before cap_by_income.");
      await wv.locator("select[aria-label=record]").selectOption({ label: "client_id 1" });
      await wv.locator(".chip", { hasText: "term_cap" }).first().click();
      await wv.locator(".how-title", { hasText: "term_cap = " }).waitFor();
      await shot("After focusing client_id 1 in the header and clicking the term_cap chip.");
      await tab(c, "State");
      await shot("The State tab while focused on client_id 1.");
      await tab(c, "Graph");
      await wv.locator("svg .node", { hasText: "risk_tree" }).click();
      await wv.locator("aside button", { hasText: "Run to risk_tree" }).click();
      await wv.locator(".pause-banner", { hasText: "before risk_tree" }).waitFor({ timeout: 30_000 });
      await wv.locator("aside button", { hasText: "Run through risk_tree" }).click();
      await wv.locator(".pause-banner", { hasText: "after risk_tree" }).waitFor({ timeout: 30_000 });
      await wv.locator("aside h4", { hasText: "Path for client_id 1" }).waitFor({ timeout: 10_000 });
      await shot("After clicking risk_tree, 'Run to risk_tree', then 'Run through risk_tree': the path client_id 1 took.");
    });
  }, 180_000);

  it("check what a param change would do", async () => {
    await story("3-what-if", "A model owner wants to know what lowering the income cap from 48 to 24 months would change, before editing any code.", async (c, shot) => {
      await open(c);
      await goTo(c, LINE.pipeline);
      await lens(c, "What-if").click({ timeout: 60_000 });
      const wv = c.webview();
      await wv.locator('input[aria-label="term/cap_by_income cap"]').waitFor({ timeout: 30_000 });
      await shot("After clicking 'What-if' above the pipeline.");
      await wv.locator('input[aria-label="term/cap_by_income cap"]').fill("24");
      await shot("After typing 24 into term/cap_by_income · cap.");
      await wv.locator("button", { hasText: "Run and compare" }).click();
      await wv.locator(".compare .comparing").waitFor({ timeout: 90_000 });
      await shot("The comparison after clicking 'Run and compare'.");
      await tab(c, "Graph");
      await shot("The graph after the comparison.");
    });
  }, 180_000);

  it("sweep params and inputs from a paused point", async () => {
    await story("4-scenarios", "A risk analyst pauses the run just before the income cap and wants to try caps of 24, 36 and 48 together with requested amounts of 50000 and 150000, then compare every combination with the original run.", async (c, shot) => {
      await runToBreakpoint(c, LINE.capByIncome);
      const wv = c.webview();
      await wv.locator(".pause-banner", { hasText: "cap_by_income" }).waitFor({ timeout: 30_000 });
      await tab(c, "Scenarios");
      await shot("The Scenarios tab while paused before cap_by_income.");
      await wv.locator('input[aria-label="knob"]').first().fill("cap_by_income (cap) in term");
      await wv.locator('input[aria-label="knob values"]').first().fill("24, 36, 48");
      await wv.locator("button", { hasText: "+ add another" }).click();
      await wv.locator('input[aria-label="knob"]').nth(1).fill("field · requested_amount");
      await wv.locator('input[aria-label="knob values"]').nth(1).fill("50000, 150000");
      await shot("Two rows filled in: the cap parameter and the requested_amount input field.");
      await wv.locator("button", { hasText: /^Run 6 scenarios/ }).click();
      await wv.locator("table.sweep").waitFor({ timeout: 120_000 });
      await shot("The results table after running the 6 scenarios.");
      await wv.locator("table.sweep tr.clickable").nth(0).click();
      await wv.locator(".inline-compare .compare .comparing").waitFor();
      await shot("After clicking the first scenario row: its comparison opens under the table.");
    });
  }, 240_000);

  it("see what changed since the last commit", async () => {
    const repo = fs.mkdtempSync(path.join(os.tmpdir(), "decider-story-"));
    fs.copyFileSync(path.join(ROOT, "examples", "loan.py"), path.join(repo, "loan.py"));
    fs.mkdirSync(path.join(repo, ".vscode"));
    // The copy lives outside the project, so point uv at it for the decider package.
    fs.writeFileSync(path.join(repo, ".vscode", "settings.json"), JSON.stringify({ "decider.python": ["uv", "run", "--project", path.resolve(ROOT, "..", ".."), "python"] }));
    const g = (...a: string[]) => execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@t", ...a], { cwd: repo });
    g("init", "-q");
    g("add", ".");
    g("commit", "-qm", "loan pipeline");
    const src = fs.readFileSync(path.join(repo, "loan.py"), "utf8");
    fs.writeFileSync(path.join(repo, "loan.py"), src.replace("return offer * 0.8", "return offer * 0.9").replace("cap: float = param(48.0)", "cap: float = param(42.0)"));
    await story("5-git-compare", "A developer changed the loan shrink factor and a cap default, and wants to see, step by step, how the flow now behaves differently from the last commit.", async (c, shot) => {
      await open(c);
      await goTo(c, LINE.pipeline);
      await lens(c, "Compare with").click({ timeout: 60_000 });
      await c.page.locator(".quick-input-widget .monaco-list-row").first().waitFor({ timeout: 30_000 });
      await shot("After clicking 'Compare with…' above the pipeline.");
      await c.page.keyboard.press("Enter");
      const wv = c.webview();
      await wv.locator(".compare .compare-title", { hasText: "uncommitted" }).waitFor({ timeout: 120_000 });
      await shot("The comparison of HEAD against the working tree.");
      await tab(c, "Graph");
      await shot("The graph after the comparison.");
    }, repo);
  }, 240_000);
});
