// User stories for steering a run and tracing a value: forcing a branch or a loop, comparing arms and
// iteration counts, value and iteration breakpoints, and a value's history. Mostly on the 1,000-step bank flow.
// `pnpm test:features` writes test/e2e/shots/features/manifest.json for a reviewer to read.
import * as fs from "node:fs";
import * as path from "node:path";
import { afterAll, describe, it } from "vitest";
import { SHOTS, type Codium } from "./codium";
import { find, keys, lens, out, runTo, story, visualise, writeManifest, wv } from "./harness";

out.dir = path.join(SHOTS, "features");

/** Pick the option of `select` whose text passes `test`. */
async function pick(c: Codium, select: string, test: (text: string) => boolean) {
  const options = await wv(c).locator(`${select} option`).allTextContents();
  const i = options.findIndex(test);
  if (i < 0) throw new Error(`no option in ${select} among ${options.join(" | ")}`);
  await wv(c).locator(select).selectOption({ index: i });
}

const focus = (c: Codium, row: string) => wv(c).locator("select[aria-label=record]").first().selectOption(row);

/** Start a debug run of the open flow on its SAMPLE, paused at the start. */
async function debug(c: Codium, file: string) {
  await c.page.locator(".tab", { hasText: file }).first().click();
  await c.command("decider: Run flow with inputs");
  await c.page.locator(".quick-input-widget .quick-input-list .monaco-list-row", { hasText: "SAMPLE" }).click();
  await wv(c).locator(".pause-banner:not(.pending)").waitFor({ timeout: 180_000 });
}

async function resume(c: Codium, file: string) {
  await c.page.locator(".tab", { hasText: file }).first().click();
  await c.page.keyboard.press("F5");
}

describe("steering and tracing stories", () => {
  afterAll(writeManifest);

  it("compare two products for one applicant", async () => {
    fs.rmSync(out.dir, { recursive: true, force: true });
    fs.mkdirSync(out.dir, { recursive: true });
    await story(
      "F1-compare-products",
      "Applicant client_id 20400 asked for a personal loan. See what they would be offered as a credit card instead, side by side.",
      async (c, shot) => {
        await visualise(c);
        await find(c, "product_arm");
        await runTo(c);
        await focus(c, "0");
        await wv(c).locator(".group-controls h4", { hasText: "product" }).waitFor({ timeout: 30_000 });
        await wv(c).locator(".group-controls h4").scrollIntoViewIfNeeded();
        await shot("Ran to product_arm (the product branch's condition) and focused client_id 20400: the branch's controls in the details.");
        await wv(c).locator('select[aria-label="product compare for"]').selectOption("one");
        await wv(c).locator('.group-controls select[aria-label$="what-if b"]').selectOption("1");
        await shot("Under 'Compare two ways', picked 'For client_id 20400: down personal_loan vs down credit_card'.");
        await wv(c).locator(".group-controls button", { hasText: "Compare" }).click();
        await wv(c).locator(".compare .compare-title").waitFor({ timeout: 240_000 });
        await shot("The comparison after clicking Compare.");
      },
    );
  }, 1_200_000);

  it("send one applicant down another product mid-run", async () => {
    await story(
      "F2-force-a-product",
      "In a debug run paused at the end of the flow, send client_id 20400 down the credit card arm instead, re-run from the branch, and see their final offer.",
      async (c, shot) => {
        await visualise(c);
        await find(c, "total_cost_of_credit");
        await runTo(c);
        await focus(c, "0");
        await find(c, "product_arm");
        await wv(c).locator('select[aria-label="product force for"]').selectOption("one");
        await pick(c, 'select[aria-label="force product"]', (t) => t.includes("credit_card"));
        await wv(c).locator(".controls-bar .chip").first().waitFor();
        await shot("Paused at total_cost_of_credit (the last step) with client_id 20400 focused, selected product_arm and chose 'Send client_id 20400 down credit_card'.");
        await wv(c).locator(".group-controls button", { hasText: "Re-run" }).click();
        await wv(c).locator(".pause-banner", { hasText: "product_arm" }).waitFor({ timeout: 120_000 });
        await shot("Clicked 'Re-run product forced'.");
        await find(c, "total_cost_of_credit");
        await wv(c).locator("aside button", { hasText: /^Run to/ }).first().click();
        await wv(c).locator(".pause-banner", { hasText: "total_cost_of_credit" }).waitFor({ timeout: 180_000 });
        await wv(c).locator("select[aria-label=explain]").selectOption("offer_rate");
        await wv(c).locator(".timeline").waitFor({ timeout: 30_000 });
        await shot("Clicked 'Run to total_cost_of_credit' and picked offer_rate under 'explain a value…'.");
      },
    );
  }, 1_200_000);

  it("pause where a rate crosses a line inside personal loans", async () => {
    await story(
      "F3-value-breakpoint",
      "Pause the run the first time any applicant's personal-loan rate goes above 25%, looking only inside the personal loan steps.",
      async (c, shot) => {
        await visualise(c);
        await find(c, "pl_regulated_rate");
        await wv(c).locator(".watch-form summary").click();
        await wv(c).locator('select[aria-label="break when name"]').selectOption("pl_rate");
        await wv(c).locator('select[aria-label="break when op"]').selectOption(">");
        await wv(c).locator('input[aria-label="break when value"]').fill("25%");
        await pick(c, 'select[aria-label="break when scope"]', (t) => t.endsWith("personal_loan"));
        await wv(c).locator(".watch-form").scrollIntoViewIfNeeded();
        await shot("Selected pl_regulated_rate, opened 'Break when a value…' and filled pl_rate > 25% only in personal_loan.");
        await wv(c).locator(".watch-form button", { hasText: "Add breakpoint" }).click();
        await wv(c).locator(".controls-bar .chip").first().waitFor();
        await shot("Clicked 'Add breakpoint'.");
        await debug(c, "pipeline.py");
        await resume(c, "pipeline.py");
        await wv(c).locator(".banner-note.hit").waitFor({ timeout: 180_000 });
        await shot("Started a debug run and pressed F5 (continue).");
      },
    );
  }, 1_200_000);

  it("find what set an applicant's rate and go back there", async () => {
    await story(
      "F4-value-history",
      "Find out which step gave client_id 20400 their personal-loan rate, see how the rate changed along the way, and go back to the moment it was set.",
      async (c, shot) => {
        await visualise(c);
        await find(c, "pl_monthly_rate");
        await runTo(c);
        await focus(c, "0");
        await wv(c).locator("select[aria-label=explain]").selectOption("pl_rate");
        await wv(c).locator(".timeline").waitFor({ timeout: 30_000 });
        await wv(c).locator(".timeline").scrollIntoViewIfNeeded();
        await shot("Ran to pl_monthly_rate, focused client_id 20400 and picked pl_rate under 'explain a value…'.");
        await wv(c).locator(".timeline .why button").click();
        await wv(c).locator(".pause-banner", { hasText: "Went back" }).waitFor({ timeout: 120_000 });
        await wv(c).locator(".timeline").scrollIntoViewIfNeeded();
        await shot("Clicked the '⤺ Go back…' link in the answer.");
      },
    );
  }, 1_200_000);

  it("try a loop with 5 and 10 iterations and pause in iteration 3", async () => {
    await story(
      "F5-loop",
      "In loan.py's offer-shrinking loop, compare running it 5 times with 10 times, then pause the debug run at iteration 3 and see how the offer got there.",
      async (c, shot) => {
        await keys(c, "Control+P", "loan.py");
        await keys(c, "Control+G");
        await c.page.keyboard.type("114");
        await c.page.keyboard.press("Enter");
        await lens(c, "Visualise flow").waitFor({ timeout: 90_000 });
        await lens(c, "Visualise flow").click();
        await wv(c).locator("svg .node", { hasText: "shrink" }).last().click();
        await wv(c).locator(".group-controls h4", { hasText: "Loop" }).waitFor({ timeout: 30_000 });
        await wv(c).locator(".group-controls").scrollIntoViewIfNeeded();
        await shot("Opened loan.py, clicked 'Visualise flow' and selected the shrink step inside the loop.");
        await wv(c).locator(".group-controls button", { hasText: "Compare" }).click();
        await wv(c).locator(".compare .compare-title").waitFor({ timeout: 120_000 });
        await shot("Clicked Compare beside 'What if: 5 vs 10 iterations'.");
        await wv(c).locator("header nav button", { hasText: "Graph" }).click();
        await wv(c).locator("svg .node", { hasText: "shrink" }).last().click();
        await wv(c).locator('input[aria-label="pause shrink_offer at iteration"]').fill("3");
        await wv(c).locator(".group-controls button", { hasText: "Add breakpoint" }).click();
        await debug(c, "loan.py");
        await resume(c, "loan.py");
        await wv(c).locator(".banner-note.hit").waitFor({ timeout: 60_000 });
        await focus(c, "0");
        await wv(c).locator("select[aria-label=explain]").selectOption("offer");
        await wv(c).locator(".timeline").waitFor({ timeout: 30_000 });
        await wv(c).locator(".timeline").scrollIntoViewIfNeeded();
        await shot("Added 'pause before iteration 3', started a debug run, pressed F5, focused client_id 1 and picked offer under 'explain a value…'.");
      },
    );
  }, 1_200_000);
});
