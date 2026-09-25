// The flow debugger in a real JupyterLab, as a person drives it. Run with `pnpm e2e`; screenshots land in test/e2e/shots/.
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { launch, type Lab } from "./lab";

describe("decider in JupyterLab", () => {
  let lab: Lab;

  beforeAll(async () => {
    lab = await launch();
  });

  afterEach(async (ctx) => {
    if (ctx.task.result?.state === "fail") await lab.shot(`failed-${ctx.task.name.replace(/\W+/g, "-")}`);
  });

  afterAll(async () => {
    await lab?.close();
  });

  const panel = () => lab.page.locator(".jp-MainAreaWidget:visible", { has: lab.page.locator(".jp-DeciderPanel") }).last();
  const ui = () => panel().locator(".decider");
  const button = (label: string) => panel().locator(".jp-Toolbar .jp-ToolbarButtonComponent", { hasText: label });
  const banner = (text: string) => ui().locator(".pause-banner", { hasText: text }).waitFor({ timeout: 60_000 });

  it("opens a flow file's graph from the file browser", async () => {
    await lab.page.locator(".jp-DirListing-item", { hasText: "loan.py" }).click({ button: "right" });
    await lab.page.locator(".lm-Menu-item", { hasText: "Debug decider flow" }).click();
    await ui().locator("svg .node", { hasText: "join_bureau" }).first().waitFor({ timeout: 90_000 });
    expect(await ui().locator("svg .node").count()).toBeGreaterThan(10);
    // The launcher would take half the width in every screenshot.
    await lab.page.locator(".lm-TabBar-tab", { hasText: "Launcher" }).locator(".lm-TabBar-tabCloseIcon").click();
    await lab.shot("01-graph");
    await ui().locator("svg .node", { hasText: "banding" }).first().click();
    await ui().locator("aside button", { hasText: "Open source" }).click();
    await lab.page.locator(".lm-TabBar-tab", { hasText: /^loan\.py/ }).waitFor({ timeout: 30_000 });
    await lab.page.locator(".jp-StatusBar-Widget", { hasText: "Ln 55, Col 1" }).first().waitFor({ timeout: 30_000 }); // banding's decorator
    await lab.shot("01b-open-source");
    await lab.page.locator(".lm-TabBar-tab", { hasText: /^loan\.py/ }).locator(".lm-TabBar-tabCloseIcon").click();
  });

  it("runs to a step, focuses a record and explains a value with its history", async () => {
    await button("Start").click();
    await banner("before the start");
    await ui().locator("svg .node", { hasText: "banding" }).first().click();
    await ui().locator("aside button", { hasText: "Run to banding" }).click();
    await banner("before banding");
    await ui().locator("select[aria-label=record]").first().selectOption("0");
    await ui().locator("select[aria-label=explain]").selectOption("ratio");
    await ui().locator(".timeline h4", { hasText: "How ratio changed" }).waitFor({ timeout: 30_000 });
    expect(await ui().locator(".timeline ol.history li").count()).toBeGreaterThan(0);
    await lab.shot("02-explain-value");
    await button("Step over").click();
    await banner("after banding");
    await ui().locator("header nav button", { hasText: "State" }).click();
    await ui().locator("td", { hasText: "band_score" }).first().waitFor();
    await lab.shot("03-state");
  });

  it("forces a branch arm, re-runs it, and compares the two arms", async () => {
    await ui().locator("header nav button", { hasText: "Graph" }).click();
    await ui().locator("svg .node", { hasText: "offer" }).first().click();
    await ui().locator("aside button", { hasText: "Run to offer" }).click();
    await banner("before offer");
    await ui().locator("svg .node", { hasText: "is_private" }).first().click();
    await ui().locator('select[aria-label="force by_sector"]').selectOption("1");
    await ui().locator(".controls-bar .chip", { hasText: "by_sector → cap_public" }).waitFor();
    await ui().locator(".group-controls button", { hasText: "Re-run from by_sector with this force" }).click();
    await ui().locator(".banner-note", { hasText: "Re-ran from is_private" }).waitFor({ timeout: 60_000 });
    await lab.shot("04-forced-rerun");
    await ui().locator(".group-controls button", { hasText: "Compare" }).click();
    await ui().locator(".compare .compare-title", { hasText: "cap_public (at by_sector)" }).waitFor({ timeout: 90_000 });
    await lab.shot("05-compare-arms");
  });

  it("runs a what-if on a param and shows where the runs diverge", async () => {
    await button("Stop").click();
    await ui().locator("header nav button", { hasText: "What-if" }).click();
    await ui().locator('input[aria-label="term/cap_by_income cap"]').fill("24");
    await ui().locator("button", { hasText: "Run and compare" }).click();
    await ui().locator(".compare a", { hasText: "term/cap_by_income" }).first().waitFor({ timeout: 90_000 });
    await lab.shot("06-what-if");
    await ui().locator("header nav button", { hasText: "Graph" }).click();
    expect(await ui().locator("svg .node.diff-changed").count()).toBeGreaterThan(0);
  });

  it("pauses before a loop iteration and where a value first crosses a line", async () => {
    await ui().locator("svg .node", { hasText: "shrink" }).first().click();
    await ui().locator('input[aria-label="pause shrink_offer at iteration"]').fill("3");
    await ui().locator(".group-controls button", { hasText: "Add breakpoint" }).click();
    await ui().locator(".watch-form summary").click();
    await ui().locator('select[aria-label="break when name"]').selectOption("offer");
    await ui().locator('select[aria-label="break when op"]').selectOption("<");
    await ui().locator('input[aria-label="break when value"]').fill("70000");
    await ui().locator(".watch-form button", { hasText: "Add breakpoint" }).click();
    await ui().locator(".controls-bar .chip", { hasText: "offer < R 70,000.00" }).waitFor();
    await button("Continue").click();
    await ui().locator(".banner-note.hit", { hasText: "iteration 3 of shrink_offer" }).waitFor({ timeout: 60_000 });
    await button("Continue").click();
    await ui().locator(".banner-note.hit", { hasText: "offer < R 70,000.00" }).waitFor({ timeout: 30_000 });
    expect(await ui().locator(".banner-note.hit").innerText()).toContain("client_id 1");
    await lab.shot("07-value-breakpoint");
  });

  it("sweeps scenarios from the start", async () => {
    await button("Stop").click();
    await ui().locator("header nav button", { hasText: "Scenarios" }).click();
    await ui().locator('input[aria-label="knob"]').first().fill("cap_by_income (cap) in term");
    await ui().locator('input[aria-label="knob values"]').first().fill("24, 36, 48");
    await ui().locator("button", { hasText: /^Run 3 scenarios/ }).click();
    await ui().locator("table.sweep").waitFor({ timeout: 120_000 });
    await lab.shot("08-scenarios");
  });

  it("debugs a pipeline defined in a notebook, on the notebook's DataFrame", async () => {
    await lab.command("notebook:create-new", { kernelName: "python3" });
    const cell = lab.page.locator(".jp-NotebookPanel:visible .jp-Cell .cm-content").first();
    await cell.waitFor({ timeout: 60_000 });
    await lab.page.locator(".jp-NotebookPanel:visible .jp-Notebook-ExecutionIndicator[data-status=idle]").waitFor({ timeout: 60_000 });
    const code = [
      "import polars as pl",
      "from decider import flow, param, step",
      "from decider_jupyter import debug",
      "",
      "def total(a: float, b: float) -> float:",
      "    return a + b",
      "",
      "@step(output='total')",
      "def scale(total: float, k: float = param(2.0)) -> float:",
      "    return total * k",
      "",
      "pricing = flow(total, scale)",
      "df = pl.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})",
      "debug(pricing, df)",
    ];
    // The editor indents after a colon itself, so set the cell's text rather than type it.
    await lab.page.evaluate((src) => (window as any).jupyterapp.shell.currentWidget.content.activeCell.model.sharedModel.setSource(src), code.join("\n"));
    await lab.command("notebook:run-cell");
    await ui().locator("svg .node", { hasText: "scale" }).first().waitFor({ timeout: 120_000 });
    await button("Start").click();
    await banner("before the start");
    await ui().locator("svg .node", { hasText: "scale" }).first().click();
    await ui().locator("aside button", { hasText: "Run to scale" }).click();
    await banner("before scale");
    await button("Step over").click();
    await banner("after scale");
    await ui().locator("header nav button", { hasText: "State" }).click();
    await ui().locator("td", { hasText: "total" }).first().waitFor();
    expect(await ui().innerText()).toContain("8");
    await lab.shot("09-notebook-flow");
    await lab.command("apputils:change-theme", { theme: "JupyterLab Dark" });
    await lab.page.locator("body[data-jp-theme-light=false]").waitFor({ state: "attached", timeout: 30_000 });
    await ui().locator("header nav button", { hasText: "Graph" }).click();
    await lab.shot("10-dark-theme");
  });
});
