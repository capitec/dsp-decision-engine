import { describe, expect, it } from "vitest";
import { makeHandler, type AgentState } from "../src/agentActions";

function fake(over: Partial<AgentState> = {}) {
  const calls: unknown[][] = [];
  const state: AgentState = {
    editor: () => undefined,
    shown: () => ({ file: "/w/flow.yaml", described: true }),
    structureSelection: () => [],
    graphSelection: () => undefined,
    paused: () => undefined,
    breakpoints: () => [],
    frame: async () => undefined,
    highlight: async (n) => (calls.push(["highlight", n]), n.startsWith("credit/")),
    reveal: async (...a) => void calls.push(["reveal", ...a]),
    showFlow: async (...a) => void calls.push(["showFlow", ...a]),
    setBreakpoints: (n) => (calls.push(["setBreakpoints", n]), n.filter((x) => x.endsWith("a"))),
    debugFlow: async (...a) => (calls.push(["debugFlow", ...a]), "credit/a"),
    debugControl: async (...a) => (calls.push(["debugControl", ...a]), undefined),
    debugEvaluate: async (e) => `=${e}`,
    ...over,
  };
  return { state, calls, handle: makeHandler(state, "/w") };
}

describe("selection", () => {
  it("reports file, selection, config location and debugger state", async () => {
    const text = '{"type":"tree","tree":{"nodes":[{"id":"n1"}]}}';
    const { handle } = fake({
      editor: () => ({ file: "/w/trees/t.json", text, cursor: text.indexOf("n1"), start: { line: 1, column: 5 }, end: { line: 1, column: 7 }, selected: "n1" }),
      structureSelection: () => ["credit/a"],
      graphSelection: () => "credit/b",
      paused: () => "credit/a",
      frame: async () => ({ file: "/w/steps.py", line: 9, name: "a" }),
      breakpoints: () => ["credit/a"],
    });
    expect(await handle({ kind: "selection" })).toMatchObject({
      file: "/w/trees/t.json",
      selection: { text: "n1" },
      config: { path: ["tree", "nodes", 0, "id"], nodeId: "n1" },
      structure_selection: ["credit/a"],
      graph_selection: "credit/b",
      debug: { paused: "credit/a", frame: { line: 9 }, breakpoints: ["credit/a"] },
    });
  });

  it("is empty-handed with no editor, not an error", async () => {
    expect(await fake().handle({ kind: "selection" })).toMatchObject({ file: null, selection: undefined, config: null });
  });

  it("skips config parsing for source files", async () => {
    const { handle } = fake({ editor: () => ({ file: "/w/a.py", text: "x: 1", cursor: 0, start: { line: 1, column: 1 }, end: { line: 1, column: 1 }, selected: "" }) });
    expect(await handle({ kind: "selection" })).toMatchObject({ config: null });
  });
});

describe("navigation", () => {
  it("highlights the first node the flow has and reports the rest as skipped", async () => {
    const { handle } = fake();
    expect(await handle({ kind: "highlight", nodes: ["x", "credit/a", "credit/b"] })).toEqual({ highlighted: "credit/a", skipped: ["x", "credit/b"] });
  });

  it("fails when no node is in the flow, or no flow is shown", async () => {
    await expect(fake().handle({ kind: "highlight", nodes: ["nope"] })).rejects.toThrow(/none of/);
    await expect(fake({ shown: () => undefined }).handle({ kind: "highlight", nodes: ["credit/a"] })).rejects.toThrow(/show_flow/);
  });

  it("shows the named flow first when it is not the one shown", async () => {
    const { handle, calls } = fake();
    await handle({ kind: "highlight", nodes: ["credit/a"], flow: "other/flow.yaml" });
    expect(calls[0]).toEqual(["showFlow", "/w/other/flow.yaml"]);
  });

  it("resolves a relative file against the window's folder", async () => {
    const { handle, calls } = fake();
    await handle({ kind: "reveal", path: "steps.py", line: 12 });
    await handle({ kind: "show_flow", file: "/abs/flow.yaml", pipeline: "main" });
    expect(calls).toEqual([["reveal", "/w/steps.py", 12], ["showFlow", "/abs/flow.yaml", "main"]]);
  });
});

describe("debugging", () => {
  it("lists nodes that could not take a breakpoint", async () => {
    expect(await fake().handle({ kind: "set_breakpoints", nodes: ["credit/a", "credit/b"] })).toEqual({ breakpoints: ["credit/a"], not_set: ["credit/b"] });
  });

  it("needs a described flow to set breakpoints", async () => {
    await expect(fake({ shown: () => undefined }).handle({ kind: "set_breakpoints", nodes: [] })).rejects.toThrow(/show_flow/);
  });

  it("starts a flow on a dataset and reports where it paused", async () => {
    const { handle, calls } = fake();
    expect(await handle({ kind: "debug_flow", file: "flow.yaml", dataset: "applicants" })).toEqual({ started: true, paused: "credit/a" });
    expect(calls).toEqual([["debugFlow", "/w/flow.yaml", undefined, "applicants"]]);
  });

  it("rejects an unknown control and passes a known one", async () => {
    const { handle, calls } = fake();
    await expect(handle({ kind: "debug_control", action: "rewind" })).rejects.toThrow(/one of/);
    expect(await handle({ kind: "debug_control", action: "step_over" })).toEqual({ paused: null });
    expect(calls).toEqual([["debugControl", "step_over"]]);
  });

  it("evaluates an expression", async () => {
    expect(await fake().handle({ kind: "debug_evaluate", expression: "x + 1" })).toEqual({ result: "=x + 1" });
  });

  it("requires string arguments", async () => {
    await expect(fake().handle({ kind: "debug_evaluate" })).rejects.toThrow(/'expression'/);
  });

  it("rejects an unknown action", async () => {
    await expect(fake().handle({ kind: "explode" })).rejects.toThrow(/unknown action/);
  });
});
