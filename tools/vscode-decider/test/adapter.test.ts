import * as path from "node:path";
import { DebugClient } from "@vscode/debugadapter-testsupport";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

const ROOT = path.resolve(__dirname, "..");
const LOAN = path.join(ROOT, "examples", "loan.py");
const ADAPTER = path.join(ROOT, "dist", "adapterMain.js");

// Drives the built adapter over stdio, exactly as an editor would.
describe("decider debug adapter", () => {
  let dc: DebugClient;

  beforeEach(async () => {
    dc = new DebugClient("node", ADAPTER, "decider", { env: { ...process.env, DECIDER_PYTHON: "python3" } });
    await dc.start();
  });

  afterEach(async () => {
    await dc.stop();
  });

  const launch = (extra: Record<string, unknown> = {}) =>
    Promise.all([dc.configurationSequence(), dc.launch({ program: LOAN, stopOnEntry: true, ...extra }), dc.waitForEvent("stopped")]);

  it("stops on entry at the root and shows the pipeline as a stack", async () => {
    const [, , stopped] = await launch();
    expect(stopped.body.reason).toBe("entry");
    const st = await dc.stackTraceRequest({ threadId: 1 });
    expect(st.body.stackFrames.map((f) => f.name)).toEqual(["<root>  [sequence]"]);
  });

  it("a function breakpoint on a path pauses there with inputs, outputs and state scopes", async () => {
    await launch();
    await dc.setFunctionBreakpointsRequest({ breakpoints: [{ name: "term/cap_by_income" }] });
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    const st = await dc.stackTraceRequest({ threadId: 1 });
    expect(st.body.stackFrames.map((f) => f.name)).toEqual(["cap_by_income  [call]", "term  [sequence]", "<root>  [sequence]"]);
    expect(st.body.stackFrames[0].source?.path).toBe(LOAN);

    const scopes = await dc.scopesRequest({ frameId: st.body.stackFrames[0].id });
    expect(scopes.body.scopes.map((s) => s.name)).toEqual(["Inputs", "Outputs", "State"]);
    const inputs = await dc.variablesRequest({ variablesReference: scopes.body.scopes[0].variablesReference });
    expect(inputs.body.variables.map((v) => v.name)).toEqual(["term_cap", "min_net_salary"]);
    expect(inputs.body.variables[0].value).toBe("[60, 36]");
    expect(inputs.body.variables[1].value).toBe("[4000, null] 1 null");
  });

  it("setVariable overrides a column and the run continues with it", async () => {
    await launch();
    await dc.setFunctionBreakpointsRequest({ breakpoints: [{ name: "term/cap_by_income" }] });
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    const st = await dc.stackTraceRequest({ threadId: 1 });
    const scopes = await dc.scopesRequest({ frameId: st.body.stackFrames[0].id });
    const set = await dc.setVariableRequest({ variablesReference: scopes.body.scopes[2].variablesReference, name: "term_cap", value: "12.0" });
    expect(set.body.value).toBe("[12, 12]");
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("terminated")]);
  });

  it("a source breakpoint on a step's def line maps to that node", async () => {
    await launch();
    const bps = await dc.setBreakpointsRequest({ source: { path: LOAN }, breakpoints: [{ line: 46 }] });
    expect(bps.body.breakpoints[0].verified).toBe(true);
    expect(bps.body.breakpoints[0].message).toBe("banding");
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    const st = await dc.stackTraceRequest({ threadId: 1 });
    expect(st.body.stackFrames[0].name).toBe("banding  [call]");
  });

  it("next steps over a group and stepIn enters it", async () => {
    await launch();
    await dc.setFunctionBreakpointsRequest({ breakpoints: [{ name: "term" }] });
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    await Promise.all([dc.stepInRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    let st = await dc.stackTraceRequest({ threadId: 1 });
    expect(st.body.stackFrames[0].name).toBe("term_cap  [call]");
    await Promise.all([dc.nextRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    await Promise.all([dc.nextRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    st = await dc.stackTraceRequest({ threadId: 1 });
    expect(st.body.stackFrames[0].name).toBe("by_sector  [branch]");
    await Promise.all([dc.nextRequest({ threadId: 1 }), dc.waitForEvent("terminated")]);
  });

  it("custom requests expose the IR, the state and lineage for the views", async () => {
    await launch();
    await dc.setFunctionBreakpointsRequest({ breakpoints: [{ name: "term/by_sector/cap_private" }] });
    await Promise.all([dc.continueRequest({ threadId: 1 }), dc.waitForEvent("stopped")]);
    const describe = await dc.customRequest("decider.describe");
    expect(describe.body.pipeline).toBe("pipeline");
    const state = await dc.customRequest("decider.state");
    expect(state.body.columns.map((c: { name: string }) => c.name)).toContain("band_score");
    const lineage = await dc.customRequest("decider.lineage", { name: "term_cap" });
    expect(lineage.body.producer).toBe("term/cap_by_income");
    const ev = await dc.evaluateRequest({ expression: "band" });
    expect(ev.body.result).toBe("[1, 1]");
  });
});
