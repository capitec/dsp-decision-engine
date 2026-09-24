// Forcing branches and loops, and value and iteration breakpoints, as a person drives them in VSCodium.
// Run with `pnpm test:controls`; screenshots land in test/e2e/shots/.
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { launch, type Codium } from "./codium";

const PIPELINE_LINE = 114;

describe("branch and loop controls", () => {
  let c: Codium;

  beforeAll(async () => {
    c = await launch();
  }, 90_000);

  afterAll(async () => {
    await c?.close();
  });

  const lens = (title: string) => c.page.locator(".codelens-decoration a", { hasText: title }).first();
  const openLoan = async () => {
    await c.page.locator(".explorer-folders-view .monaco-list-row", { hasText: "loan.py" }).click();
    await c.page.keyboard.press("Control+G");
    await c.page.keyboard.type(String(PIPELINE_LINE));
    await c.page.keyboard.press("Enter");
  };

  it("compares a branch's two arms from the step details", async () => {
    await openLoan();
    await lens("Visualise flow").waitFor({ timeout: 60_000 });
    await lens("Visualise flow").click();
    const wv = c.webview();
    await wv.locator("svg .node", { hasText: "cap_public" }).first().click();
    await wv.locator(".group-controls h4", { hasText: "Branch by_sector" }).waitFor({ timeout: 30_000 });
    await c.shot("c01-branch-controls");
    await wv.locator(".group-controls button", { hasText: "Compare" }).click();
    await wv.locator(".compare .compare-title", { hasText: "down cap_public" }).waitFor({ timeout: 90_000 });
    await c.shot("c02-compare-arms");
  }, 180_000);

  it("compares a loop run 5 and 10 times", async () => {
    const wv = c.webview();
    await wv.locator("header nav button", { hasText: "Graph" }).click();
    await wv.locator("svg .node", { hasText: "shrink" }).first().click();
    await wv.locator(".group-controls h4", { hasText: "Loop shrink_offer" }).waitFor();
    await wv.locator(".group-controls button", { hasText: "Compare" }).click();
    await wv.locator(".compare .compare-title", { hasText: "10×" }).waitFor({ timeout: 90_000 });
    expect(await wv.locator(".compare").innerText()).toContain("offer");
    await c.shot("c03-compare-iterations");
  }, 180_000);

  it("pauses before an iteration and where a value first crosses a line", async () => {
    const wv = c.webview();
    await wv.locator("header nav button", { hasText: "Graph" }).click();
    await wv.locator("svg .node", { hasText: "shrink" }).first().click();
    await wv.locator('input[aria-label="pause shrink_offer at iteration"]').fill("3");
    await wv.locator(".group-controls button", { hasText: "Add breakpoint" }).click();
    await wv.locator(".watch-form summary").click();
    await wv.locator('select[aria-label="break when name"]').selectOption("offer");
    await wv.locator('select[aria-label="break when op"]').selectOption("<");
    await wv.locator('input[aria-label="break when value"]').fill("70000");
    await wv.locator(".watch-form button", { hasText: "Add breakpoint" }).click();
    await wv.locator(".controls-bar .chip", { hasText: "offer < 70,000" }).waitFor();
    await c.shot("c04-breakpoints-set");

    await c.page.locator(".tab", { hasText: "loan.py" }).first().click();
    await c.command("decider: Run flow with inputs");
    await c.page.locator(".quick-input-widget .quick-input-list .monaco-list-row", { hasText: "SAMPLE" }).click();
    await wv.locator(".pause-banner", { hasText: "before the start" }).waitFor({ timeout: 60_000 });
    await c.page.locator(".tab", { hasText: "loan.py" }).first().click();
    await c.page.keyboard.press("F5");
    await wv.locator(".banner-note.hit", { hasText: "iteration 3 of shrink_offer" }).waitFor({ timeout: 30_000 });
    await c.shot("c05-iteration-hit");
    await c.page.locator(".tab", { hasText: "loan.py" }).first().click();
    await c.page.keyboard.press("F5");
    await wv.locator(".banner-note.hit", { hasText: "offer < 70000" }).waitFor({ timeout: 30_000 });
    expect(await wv.locator(".banner-note.hit").innerText()).toContain("client_id 1");
    await c.shot("c06-value-hit");
  }, 180_000);

  it("shows how a value changed and goes back to the iteration that made it", async () => {
    const wv = c.webview();
    await wv.locator("select[aria-label=record]").first().selectOption("0");
    await wv.locator("select[aria-label=explain]").selectOption("offer");
    const timeline = wv.locator(".timeline");
    await timeline.locator(".why", { hasText: "because shrink set it in iteration 4" }).waitFor({ timeout: 30_000 });
    expect(await timeline.locator("ol.history li").count()).toBe(6); // the start, the offer step, four shrinks so far
    await c.shot("c07-value-history");
    await timeline.locator("li", { hasText: "iteration 2" }).locator("button", { hasText: "go back here" }).click();
    await wv.locator(".pause-banner", { hasText: "after shrink" }).waitFor({ timeout: 30_000 });
    await timeline.locator(".why", { hasText: "in iteration 2" }).waitFor({ timeout: 30_000 });
    await c.shot("c08-went-back");
  }, 120_000);

  it("forces the branch mid-run and re-runs it", async () => {
    const wv = c.webview();
    await wv.locator("svg .node", { hasText: "cap_private" }).first().click();
    await wv.locator('select[aria-label="force by_sector"]').selectOption("1");
    await wv.locator(".controls-bar .chip", { hasText: "by_sector down cap_public" }).waitFor();
    await wv.locator(".group-controls button", { hasText: "Re-run by_sector forced" }).click();
    await wv.locator(".pause-banner", { hasText: "by_sector" }).waitFor({ timeout: 30_000 });
    await c.shot("c09-forced-rerun");
  }, 120_000);
});
