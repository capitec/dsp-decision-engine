// End to end in a real VSCodium: what a person clicks, and screenshots of what they see.
// Run with `pnpm test:e2e`; screenshots land in test/e2e/shots/.
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { launch, type Codium } from "./codium";

const PIPELINE_LINE = 109;
const CAP_PUBLIC_LINE = 47;

describe("decider in VSCodium", () => {
  let c: Codium;

  beforeAll(async () => {
    c = await launch();
  }, 90_000);

  afterAll(async () => {
    await c?.close();
  });

  const goToLine = async (line: number) => {
    await c.page.keyboard.press("Control+G");
    await c.page.keyboard.type(String(line));
    await c.page.keyboard.press("Enter");
  };

  const lens = (title: string) => c.page.locator(".codelens-decoration a", { hasText: title }).first();

  it("opens the pipeline and shows the lenses", async () => {
    await c.page.locator(".explorer-folders-view .monaco-list-row", { hasText: "loan.py" }).click();
    await goToLine(PIPELINE_LINE);
    await lens("Visualise flow").waitFor({ timeout: 60_000 });
    for (const t of ["Run flow", "What-if", "Compare with"]) await lens(t).waitFor({ timeout: 10_000 });
    await c.shot("01-lenses");
  }, 90_000);

  it("visualises the flow as a graph", async () => {
    await lens("Visualise flow").click();
    const wv = c.webview();
    await wv.locator("svg .node", { hasText: "cap_by_income" }).waitFor({ timeout: 30_000 });
    expect(await wv.locator("svg .node").count()).toBeGreaterThan(10);
    await c.shot("02-graph");
  }, 60_000);

  it("runs the flow to a breakpoint and shows the state for one record", async () => {
    await c.page.locator(".explorer-folders-view .monaco-list-row", { hasText: "loan.py" }).click();
    await goToLine(CAP_PUBLIC_LINE);
    await c.page.keyboard.press("F9");
    await goToLine(PIPELINE_LINE);
    await lens("Run flow").click();
    await c.page.locator(".quick-input-widget .quick-input-list .monaco-list-row", { hasText: "SAMPLE" }).click();
    const wv = c.webview();
    await wv.locator(".badge", { hasText: "before <root>" }).waitFor({ timeout: 60_000 });
    await c.page.keyboard.press("F5");
    await wv.locator(".badge", { hasText: "term/by_sector/cap_public" }).waitFor({ timeout: 30_000 });
    await wv.locator("select[aria-label=record]").selectOption("1");
    await wv.locator(".chip", { hasText: "term_cap" }).first().click();
    await wv.locator("aside h4", { hasText: "Lineage of term_cap, record 1" }).waitFor();
    await c.shot("03-paused-record");
    await wv.locator("header nav button", { hasText: "State" }).click();
    await wv.locator("td", { hasText: "requested_amount" }).waitFor();
    await c.shot("04-state");
  }, 120_000);

  it("runs a what-if on a param and shows where the runs diverge", async () => {
    const wv = c.webview();
    await wv.locator("header nav button", { hasText: "Params" }).click();
    await wv.locator('input[aria-label="term/cap_by_income cap"]').fill("24");
    await c.shot("05-params");
    await wv.locator("button", { hasText: "Compare what-if with defaults" }).click();
    await wv.locator(".compare a", { hasText: "term/cap_by_income" }).first().waitFor({ timeout: 90_000 });
    await c.shot("06-compare");
    await wv.locator("header nav button", { hasText: "Graph" }).click();
    expect(await wv.locator("svg .node.diff-changed").count()).toBeGreaterThan(0);
    await c.shot("07-graph-diff");
  }, 150_000);

  it("compares the flow with the last commit, step by step", async () => {
    await c.command("decider: Compare flow with a git revision");
    const pick = c.page.locator(".quick-input-widget");
    await pick.locator(".monaco-list-row").first().waitFor({ timeout: 30_000 });
    await c.page.keyboard.type("HEAD");
    await c.page.waitForTimeout(300);
    await c.shot("08-pick-revision");
    await c.page.keyboard.press("Escape");
    // The branch this worktree is on: its tip holds the same example, so nothing should differ.
    await c.command("decider: Compare flow with a git revision");
    await pick.locator(".monaco-list-row").first().waitFor({ timeout: 30_000 });
    await c.page.keyboard.type(process.env.E2E_REF ?? "worktree-debug-tools");
    await c.page.waitForTimeout(300);
    await c.page.keyboard.press("Enter");
    const wv = c.webview();
    await wv.locator(".compare .summary", { hasText: "working tree" }).waitFor({ timeout: 90_000 });
    await c.shot("09-compare-revision");
    expect(await wv.locator(".compare").innerText()).toContain("Every step produces the same values.");
  }, 150_000);
});
