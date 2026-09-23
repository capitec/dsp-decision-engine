// User stories on a realistic credit flow: about 1,000 steps across 40 files, 14 lookup tables,
// 3 decision trees and hundreds of params (examples/bank, from make_bank.py).
// `pnpm test:large` writes test/e2e/shots/large/manifest.json for a reviewer to read.
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterAll, describe, it } from "vitest";
import { launch, ROOT, SHOTS, type Codium } from "./codium";

const OUT = path.join(SHOTS, "large");
const FILE = "bank/pipeline.py";
const manifest: { story: string; goal: string; shots: { file: string; caption: string }[] }[] = [];

async function story(name: string, goal: string, body: (c: Codium, shot: (caption: string) => Promise<void>) => Promise<void>, folder?: string) {
  const entry = { story: name, goal, shots: [] as { file: string; caption: string }[] };
  manifest.push(entry);
  const c = await launch(folder);
  let n = 0;
  const shot = async (caption: string) => {
    const file = path.join(OUT, `${name}-${String(++n).padStart(2, "0")}.jpg`);
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

const keys = async (c: Codium, combo: string, text?: string) => {
  await c.page.keyboard.press(combo);
  if (text !== undefined) {
    await c.page.keyboard.type(text, { delay: 20 });
    await c.page.locator(".quick-input-list .monaco-list-row").first().waitFor({ timeout: 15_000 }).catch(() => undefined);
    await c.page.waitForTimeout(300);
    await c.page.keyboard.press("Enter");
  }
  await c.page.waitForTimeout(400);
};
const lens = (c: Codium, title: string) => c.page.locator(".codelens-decoration a", { hasText: title }).first();
const wv = (c: Codium) => c.webview();
const tab = (c: Codium, name: string) => wv(c).locator("header nav button", { hasText: name }).click();

/** Type into the graph's find box and take the first hit. */
async function find(c: Codium, query: string) {
  const box = wv(c).locator('input[aria-label="Find a step"]');
  await box.fill(query);
  await wv(c).locator(".find-hits li").first().waitFor({ timeout: 10_000 });
  await box.press("Enter");
}

/** Run to the selected step and wait for the pause. */
async function runTo(c: Codium) {
  await wv(c).locator("aside button", { hasText: /^Run to/ }).first().click({ timeout: 10_000 });
  await wv(c).locator(".pause-banner:not(.pending)").waitFor({ timeout: 180_000 });
}

/** Open the flow file and draw the flow. */
async function visualise(c: Codium) {
  await keys(c, "Control+P", FILE);
  await lens(c, "Visualise flow").waitFor({ timeout: 90_000 });
  await lens(c, "Visualise flow").click();
  await wv(c).locator("svg .node").first().waitFor({ timeout: 90_000 });
}

describe("large flow stories", () => {
  afterAll(() => {
    fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 1));
  });

  it("find one rule among a thousand steps", async () => {
    fs.rmSync(OUT, { recursive: true, force: true });
    fs.mkdirSync(OUT, { recursive: true });
    await story(
      "L1-find-a-rule",
      "A policy analyst was told that the personal loan rule 'loan too large for income in sector 4' declined an applicant. The flow has about 1,000 steps in 40 files. She wants to find that rule, see where it sits in the flow, and open its code.",
      async (c, shot) => {
        await visualise(c);
        await shot("After opening bank/pipeline.py and clicking 'Visualise flow' on the 1,000-step flow.");
        await wv(c).locator('input[aria-label="Find a step"]').fill("too large for income sector 4");
        await wv(c).locator(".find-hits li").first().waitFor({ timeout: 10_000 });
        await shot("Typed the words of the decline reason, 'too large for income sector 4', into the graph's find box.");
        await wv(c).locator('input[aria-label="Find a step"]').press("Enter");
        await shot("After pressing Enter on the first hit.");
        await wv(c).locator("aside button", { hasText: "Open source" }).first().click({ timeout: 10_000 });
        await c.page.waitForTimeout(1000);
        await shot("After 'Open source' in the step's details.");
      },
    );
  }, 300_000);

  it("adjust a rate table and see the effect", async () => {
    await story(
      "L2-adjust-a-rate-table",
      "A pricing manager wants to lower the personal loan base rate for terms of 49 to 84 months from 25.5% to 24.5% in the pl_base_rates lookup table, and see which of the 40 applications' offers change, before touching any code.",
      async (c, shot) => {
        await keys(c, "Control+P", FILE);
        await lens(c, "What-if").click({ timeout: 90_000 });
        await wv(c).locator(".params").waitFor({ timeout: 60_000 });
        await shot("After clicking 'What-if' above the pipeline: the params of the whole flow.");
        await wv(c).locator('input[aria-label="Filter params"]').fill("pl_base_rates");
        await shot("Typed pl_base_rates into the params filter.");
        await wv(c).locator('input[aria-label="pl_base_rates row 3 pl_base_rate"]').fill("24.5%");
        await shot("Typed 24.5% over the 49 to 84 month rate (row 3), which was 25.5%.");
        await wv(c).locator("button", { hasText: "Run and compare" }).click({ timeout: 10_000 });
        await wv(c).locator(".compare .verdict").waitFor({ timeout: 120_000 });
        await shot("The comparison after 'Run and compare'.");
      },
    );
  }, 400_000);

  it("find out why one applicant got their rate", async () => {
    await story(
      "L3-why-this-rate",
      "A credit analyst is asked why client 20400, a personal loan applicant, was quoted the rate she was: which row of the base rate table applied, what risk loading and discounts were added, and whether the regulatory cap kicked in.",
      async (c, shot) => {
        await visualise(c);
        await find(c, "pl_monthly_rate");
        await shot("Found pl_monthly_rate, the step after the cap and the floor, with the find box.");
        await runTo(c);
        await wv(c).locator("select[aria-label=record]").selectOption({ label: "client_id 20400" });
        await wv(c).locator(".chip", { hasText: "pl_rate" }).first().click({ timeout: 10_000 });
        await wv(c).locator(".how .formula").first().waitFor({ timeout: 20_000 });
        await shot("Ran to pl_monthly_rate, focused client_id 20400 and clicked its pl_rate input: how the rate was computed.");
        await wv(c).locator("ul.explain li", { hasText: "pl_base_rate" }).locator("button.twisty").first().click({ timeout: 10_000 }).catch(() => undefined);
        await shot("Opened pl_base_rate in the breakdown: which table row it came from.");
        await find(c, "pl_base_rates");
        await wv(c).locator(".table-match").waitFor({ timeout: 20_000 }).catch(() => undefined);
        await shot("Selected the pl_base_rates lookup table step: which row matched for client_id 20400.");
      },
    );
  }, 400_000);

  it("find and tune one parameter among hundreds", async () => {
    await story(
      "L4-tune-a-shared-param",
      "The repo rate moved from 7.75% to 7.5%. A model owner wants to find the shared repo_rate parameter among the flow's hundreds of parameters, see which steps use it, change it, and see which offers change.",
      async (c, shot) => {
        await keys(c, "Control+P", FILE);
        await lens(c, "What-if").click({ timeout: 90_000 });
        await wv(c).locator(".params").waitFor({ timeout: 60_000 });
        await shot("After clicking 'What-if': every parameter of the flow.");
        await wv(c).locator('input[aria-label="Filter params"]').fill("repo_rate");
        await shot("Filtered the params to repo_rate: which steps read it.");
        await wv(c).locator('input[aria-label="shared repo_rate"]').first().fill("7.5%");
        await wv(c).locator("button", { hasText: "Run and compare" }).click({ timeout: 10_000 });
        await wv(c).locator(".compare .verdict").waitFor({ timeout: 120_000 });
        await shot("The comparison after changing repo_rate from 7.75% to 7.5%.");
      },
    );
  }, 400_000);

  it("sweep the repo rate and personal loan cap over 40 applications", async () => {
    await story(
      "L5-sweep-many-records",
      "A risk analyst wants to see how the repo rate (7%, 7.75%, 8.5%) and the personal loan product cap (R250,000 and R350,000) together change approvals and offers across all 40 applications.",
      async (c, shot) => {
        await visualise(c);
        await tab(c, "Scenarios");
        await shot("The Scenarios tab on the large flow.");
        await wv(c).locator('input[aria-label="knob"]').first().fill("repo_rate · shared");
        await wv(c).locator('input[aria-label="knob values"]').first().fill("0.07, 0.0775, 0.085");
        await wv(c).locator("button", { hasText: "+ add another" }).click();
        await wv(c).locator('select[aria-label="knob kind"]').nth(1).selectOption("param");
        await wv(c).locator('input[aria-label="knob"]').nth(1).fill("cap · pl_product_cap (personal_loan/limits)");
        await wv(c).locator('input[aria-label="knob values"]').nth(1).fill("250000, 350000");
        await shot("After typing repo_rate and the personal loan product cap into the two knob pickers, with values.");
        await wv(c).locator("button", { hasText: /^Run 6 scenarios/ }).click({ timeout: 10_000 });
        await wv(c).locator("table.sweep").waitFor({ timeout: 180_000 });
        await shot("The results of the six scenarios across 40 applications.");
      },
    );
  }, 500_000);

  it("skip a step and swap in edited code mid-run", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "decider-bank-"));
    fs.cpSync(path.join(ROOT, "examples", "bank"), path.join(dir, "bank"), { recursive: true });
    fs.mkdirSync(path.join(dir, ".vscode"));
    fs.writeFileSync(path.join(dir, ".vscode", "settings.json"), JSON.stringify({ "decider.python": ["uv", "run", "--project", path.resolve(ROOT, "..", ".."), "python"] }));
    const pricing = path.join(dir, "bank", "products", "personal_loan", "pricing.py");
    await story(
      "L7-skip-and-swap-mid-run",
      "A pricing developer is debugging the personal loan rate. Mid-run, without restarting the 1,000-step flow, she wants to see the offers without the rate floor step, then try a tighter cap by editing pl_regulated_rate's code and running the edited version from there.",
      async (c, shot) => {
        await visualise(c);
        await find(c, "pl_rate_floor");
        await runTo(c);
        await shot("Paused before pl_rate_floor after 'Run to pl_rate_floor'.");
        await wv(c).locator("aside button", { hasText: /^Skip / }).click();
        await wv(c).locator(".pause-banner:not(.pending)").waitFor({ timeout: 60_000 });
        await find(c, "pl_rate_floor");
        await shot("After 'Skip pl_rate_floor': the run re-ran from there without it.");
        await wv(c).locator(".pause-banner button", { hasText: "Compare with the flow as started" }).click();
        await wv(c).locator(".compare .verdict").waitFor({ timeout: 120_000 });
        await shot("Clicked 'Compare with the flow as started' in the pause banner.");
        await tab(c, "Graph");
        fs.writeFileSync(pricing, fs.readFileSync(pricing, "utf8").replace("return min(pl_raw_rate, repo_rate * cap_multiple + cap_margin)", "return min(pl_raw_rate, repo_rate * cap_multiple + cap_margin - 0.01)"));
        await find(c, "pl_regulated_rate");
        await wv(c).locator("aside button", { hasText: "Use edited code" }).click();
        await wv(c).locator(".pause-banner:not(.pending)").waitFor({ timeout: 60_000 });
        await shot("Edited pl_regulated_rate's cap in pricing.py (1% tighter), saved, then clicked 'Use edited code'.");
        await wv(c).locator("aside button", { hasText: /^Run through/ }).click({ timeout: 10_000 }).catch(() => undefined);
        await wv(c).locator(".pause-banner:not(.pending)").waitFor({ timeout: 60_000 }).catch(() => undefined);
        await wv(c).locator("select[aria-label=record]").selectOption({ label: "client_id 20400" }).catch(() => undefined);
        await shot("After 'Run through pl_regulated_rate' with client_id 20400 focused.");
        await wv(c).locator(".pause-banner button", { hasText: "Compare with the flow as started" }).click();
        await wv(c).locator(".compare .verdict").waitFor({ timeout: 120_000 });
        await shot("Compared again: the flow as started against the skipped floor and the tighter cap.");
      },
      dir,
    );
  }, 500_000);

  it("see what a two-file commit changed", async () => {
    const repo = fs.mkdtempSync(path.join(os.tmpdir(), "decider-bank-"));
    fs.cpSync(path.join(ROOT, "examples", "bank"), path.join(repo, "bank"), { recursive: true });
    fs.mkdirSync(path.join(repo, ".vscode"));
    fs.writeFileSync(path.join(repo, ".vscode", "settings.json"), JSON.stringify({ "decider.python": ["uv", "run", "--project", path.resolve(ROOT, "..", ".."), "python"] }));
    const g = (...a: string[]) => execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@t", ...a], { cwd: repo });
    g("init", "-q", "-b", "main");
    g("add", ".");
    g("commit", "-qm", "Origination flow");
    const params = path.join(repo, "bank", "params.json");
    const doc = JSON.parse(fs.readFileSync(params, "utf8"));
    doc.shared.pl_base_rates[2].pl_base_rate = 0.245;
    fs.writeFileSync(params, JSON.stringify(doc, null, 1));
    const policy = path.join(repo, "bank", "products", "personal_loan", "policy.py");
    fs.writeFileSync(policy, fs.readFileSync(policy, "utf8").replace(/def pl_max_enquiries\(enquiries_6m: float, limit: float = param\([\d.]+\)\)/, "def pl_max_enquiries(enquiries_6m: float, limit: float = param(10.0))"));
    await story(
      "L6-two-file-commit",
      "A developer changed a row in the personal loan rate table (params.json) and relaxed the enquiries rule (policy.py), in two different files. Before opening a pull request she wants to see, step by step, what behaves differently from the last commit across the 40 applications.",
      async (c, shot) => {
        await keys(c, "Control+P", FILE);
        await lens(c, "Compare with").click({ timeout: 90_000 });
        await c.page.locator(".quick-input-widget .monaco-list-row").first().waitFor({ timeout: 30_000 });
        await shot("After clicking 'Compare with…'.");
        await c.page.keyboard.press("Enter");
        await wv(c).locator(".compare .verdict").waitFor({ timeout: 180_000 }).catch(() => undefined);
        await shot("The comparison of the last commit against the working tree.");
        await tab(c, "Graph");
        await shot("The graph after the comparison.");
      },
      repo,
    );
  }, 500_000);
});
