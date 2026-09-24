import * as path from "node:path";
import { DebugClient } from "@vscode/debugadapter-testsupport";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SAMPLE_WITHOUT_SECTOR } from "./fixtures";
import { compareTraces, type TraceResult } from "../src/compare";

const ROOT = path.resolve(__dirname, "..");
const LOAN = path.join(ROOT, "examples", "loan.py");
const ADAPTER = path.join(ROOT, "dist", "adapterMain.js");

// Drives the built adapter over stdio, exactly as an editor would, against the real decider session.
describe("decider debug adapter", () => {
  let dc: DebugClient;

  beforeEach(async () => {
    dc = new DebugClient("node", ADAPTER, "decider", { env: { ...process.env, DECIDER_PYTHON: "uv run python" } });
    dc.defaultTimeout = 20000;
    await dc.start();
  });

  afterEach(async () => {
    await dc.stop();
  });

  const launch = (extra: Record<string, unknown> = {}) =>
    Promise.all([dc.configurationSequence(), dc.launch({ program: LOAN, stopOnEntry: true, ...extra }), dc.waitForEvent("stopped")]);

  const breakAt = async (...names: string[]) => {
    await dc.setFunctionBreakpointsRequest({ breakpoints: names.map((name) => ({ name })) });
    const [, stopped] = await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    return stopped;
  };

  const top = async () => (await dc.stackTraceRequest({ threadId: 1 })).body.stackFrames;

  const scopes = async () => {
    const frames = await top();
    return (await dc.scopesRequest({ frameId: frames[0].id })).body.scopes;
  };

  const vars = async (ref: number) => (await dc.variablesRequest({ variablesReference: ref })).body.variables;

  it("skipping a step mid-run can be compared with the flow as started", async () => {
    await launch();
    await breakAt("term/cap_by_income");
    await dc.customRequest("decider.skip", { path: "term/cap_by_income" });
    const r = (await dc.customRequest("decider.compareEdits", {})).body as { a: TraceResult; b: TraceResult };
    const c = compareTraces(r.a, r.b, "the flow as started", "cap_by_income skipped");
    // The trace alone sees a step that didn't run; the extension names it skipped from its list of edits.
    expect(c.steps.find((st) => st.path === "term/cap_by_income")?.status).toBe("not taken");
    expect(c.output.map((o) => o.name)).toContain("term_cap");
  });

  it("a value breakpoint pauses where a record first meets it, and says why", async () => {
    await launch({ controls: { forces: [], watches: [{ name: "offer", op: "<", value: 70000 }] } });
    const [, stopped] = await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    expect(stopped.body.reason).toBe("data breakpoint");
    expect(stopped.body.text).toBe("offer < 70000");
    expect((await top())[0].name).toContain("shrink");
  });

  it("forces set mid-run send the branch down the other arm", async () => {
    await launch({ params: { term: { cap_by_income: { cap: 100.0 } } } });
    await dc.customRequest("decider.setControls", { forces: [{ path: "term/by_sector", arm: 1 }], watches: [] });
    await breakAt("sizing");
    const { columns } = (await dc.customRequest("decider.state")).body as { columns: { name: string; preview: unknown[] }[] };
    expect(columns.find((c) => c.name === "term_cap")?.preview).toEqual([60, 36]); // record 1 is private, capped as public
  });

  it("stops on entry at the root and shows the pipeline as a stack", async () => {
    const [, , stopped] = await launch();
    expect(stopped.body.reason).toBe("entry");
    expect((await top()).map((f) => f.name)).toEqual(["<root>  [sequence]  before"]);
  });

  it("a function breakpoint on a path pauses there with inputs, outputs and state scopes", async () => {
    await launch();
    await breakAt("term/cap_by_income");
    const frames = await top();
    expect(frames.map((f) => f.name)).toEqual(["cap_by_income  [scalar]  before", "term  [sequence]", "<root>  [sequence]"]);
    expect(frames[0].source?.path).toBe(LOAN);
    const s = await scopes();
    expect(s.map((x) => x.name)).toEqual(["Inputs", "Outputs", "State"]);
    const inputs = await vars(s[0].variablesReference);
    expect(inputs.map((v) => [v.name, v.value])).toEqual([["term_cap", "60, 36"], ["min_net_salary", "4,000  (1 of 2 empty)"]]);
  });

  it("setVariable overrides a column and the run continues with it", async () => {
    await launch();
    await breakAt("term/cap_by_income");
    const s = await scopes();
    const set = await dc.setVariableRequest({ variablesReference: s[2].variablesReference, name: "term_cap", value: "12.0" });
    expect(set.body.value).toBe("12, 12");
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("terminated")]);
  });

  it("source breakpoints map a step's def, and a line in a tree's Python, to their nodes", async () => {
    await launch();
    const bps = await dc.setBreakpointsRequest({ source: { path: LOAN }, breakpoints: [{ line: 56 }, { line: 101 }] });
    expect(bps.body.breakpoints.map((b) => b.message)).toEqual(["banding", "risk_tree"]);
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    expect((await top())[0].name).toBe("banding  [scalar]  before");
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    expect((await top())[0].name).toBe("risk_tree  [row]  before");
  });

  it("next steps over a branch, stepIn enters it and stepOut leaves it", async () => {
    await launch();
    await breakAt("term/by_sector");
    await Promise.all([dc.stepInRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    expect((await top())[0].name).toBe("is_private  [scalar]  before");
    await Promise.all([dc.stepOutRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    expect((await top())[0].name).toBe("by_sector  [branch]  after");
    await Promise.all([dc.nextRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    expect((await top())[0].name).toBe("term  [sequence]  after");
  });

  it("a loop body breakpoint is hit once per iteration", async () => {
    await launch();
    await breakAt("sizing/shrink_offer/shrink");
    let hits = 1;
    for (;;) {
      const next = await Promise.race([
        Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]).then(() => "stopped"),
        dc.waitForEvent("terminated").then(() => "terminated"),
      ]);
      if (next === "terminated") break;
      hits++;
    }
    expect(hits).toBe(5);
  });

  it("a tree locator breakpoint stops after the row node with a Visited scope", async () => {
    await launch();
    await breakAt("risk_tree#bureau_score < 680");
    expect((await top())[0].name).toBe("risk_tree  [row]  after");
    const visited = (await scopes()).find((x) => x.name === "Visited")!;
    expect((await vars(visited.variablesReference)).map((v) => [v.name, v.value])).toEqual([
      ["#root", "2 rows"],
      ["#bureau_score < 680", "1 rows"],
      ["#ratio >= 3", "1 rows"],
      ["#bureau_score >= 680", "1 rows"],
    ]);
  });

  it("a record's value history goes back to the iteration that made a value", async () => {
    await launch();
    await breakAt("risk_tree");
    await dc.customRequest("decider.setRecord", { row: 0 });
    const { changes } = (await dc.customRequest("decider.changes", { name: "offer" })).body as { changes: { change: number; iteration: number | null; value: unknown }[] };
    expect(changes.map((c) => c.iteration)).toEqual([null, 1, 2, 3, 4, 5]);
    const [, stopped] = await Promise.all([dc.customRequest("decider.goTo", { change: changes[2].change }), dc.waitForEvent("stopped")]);
    expect(stopped.body.reason).toBe("rewind");
    expect((await top())[0].name).toBe("shrink  [scalar]  after");
  });

  it("focusing a record shows its values and its runtime lineage", async () => {
    await launch();
    await breakAt("term/by_sector/cap_public");
    await dc.customRequest("decider.setRecord", { row: 1 });
    expect((await dc.evaluateRequest({ expression: "term_cap" })).body.result).toBe("36");
    const lineage = await dc.customRequest("decider.lineage", { name: "term_cap" });
    expect(lineage.body.producer).toBe("term/cap_by_income");
    expect(lineage.body.value).toBe(36);
  });

  it("custom requests expose the IR and the state for the views", async () => {
    await launch();
    await breakAt("term/by_sector/cap_private");
    const describe = await dc.customRequest("decider.describe");
    expect(describe.body.pipeline).toBe("pipeline");
    const state = await dc.customRequest("decider.state");
    expect(state.body.columns.map((c: { name: string }) => c.name)).toContain("bureau_score");
    expect((await dc.evaluateRequest({ expression: "band" })).body.result).toBe("1, 1");
  });

  it("a step that raises stops with an exception", async () => {
    await launch({ data: SAMPLE_WITHOUT_SECTOR });
    const [, stopped] = await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    expect(stopped.body.reason).toBe("exception");
    expect(stopped.body.text).toContain("sector_code");
  });
});
