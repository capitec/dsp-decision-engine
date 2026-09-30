import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { experimentHeadline, isDeterministicComplete, sankeyLayout, statusLabel, type ExperimentResult, type Sankey } from "../src/model/experiment";

const ROOT = path.resolve(__dirname, "../../vscode-decider");
const LOAN = path.join(ROOT, "examples", "loan.py");

const completed = (status: ExperimentResult["status"] = "completed"): ExperimentResult => ({
  name: "drift",
  status,
  nondeterministic: status === "non_reproducible",
  scenarios: [
    { name: "baseline", status: "completed", divergences: [], params: {}, overrides: {} },
    { name: "cap 24", status: status === "partial" ? "failed" : "completed", error: status === "partial" ? "bad value" : undefined, divergences: [{ kind: "divergence", location: "term_cap[0]", expected: 48, actual: 24 }], params: {}, overrides: {} },
  ],
});

describe("experiment run states", () => {
  it("names non-reproducible and partial runs, never reading them as done", () => {
    expect(experimentHeadline(completed())).toContain("1 divergence");
    expect(experimentHeadline(completed("non_reproducible"))).toContain("non-reproducible");
    expect(experimentHeadline(completed("partial"))).toContain("partial");
    expect(isDeterministicComplete(completed())).toBe(true);
    expect(isDeterministicComplete(completed("non_reproducible"))).toBe(false);
    expect(isDeterministicComplete(completed("partial"))).toBe(false);
  });

  it("labels each status with its own word", () => {
    expect(statusLabel("non_reproducible")).toBe("non-reproducible");
    expect(statusLabel("timed_out")).toBe("timed out");
    expect(statusLabel("completed")).toBe("completed");
  });
});

describe("sankey layout", () => {
  const sankey: Sankey = {
    outcome: "term_cap",
    nodes: [
      { id: "in", label: "2 records", kind: "input" },
      { id: "b#0", label: "arm 0", kind: "arm" },
      { id: "b#1", label: "arm 1", kind: "arm" },
      { id: "out:48", label: "term_cap 48", kind: "outcome" },
      { id: "out:60", label: "term_cap 60", kind: "outcome" },
    ],
    links: [
      { source: "in", target: "b#0", value: 1 },
      { source: "in", target: "b#1", value: 1 },
      { source: "b#0", target: "out:48", value: 1 },
      { source: "b#1", target: "out:60", value: 1 },
    ],
  };

  it("lays every node and link out, conserving each link's value", () => {
    const layout = sankeyLayout(sankey);
    expect(layout.nodes.map((n) => n.id)).toEqual(sankey.nodes.map((n) => n.id));
    expect(layout.links.map((l) => l.value)).toEqual(sankey.links.map((l) => l.value));
    // Arms sit one column right of the input; outcomes another right.
    const x = (id: string) => layout.nodes.find((n) => n.id === id)!.x;
    expect(x("b#0")).toBeGreaterThan(x("in"));
    expect(x("out:48")).toBeGreaterThan(x("b#0"));
  });
});

describe("a real experiment run end to end", () => {
  it("summarises divergence and a branch Sankey against the loan flow", () => {
    const script = `import json, sys
from decider.debug_bridge.bridge import Bridge
b = Bridge()
r = b.handle({"cmd": "experiment", "file": ${JSON.stringify(LOAN)}, "def_": {"name": "drift", "flow": {"entry": "loan:pipeline"}, "scenarios": [{"name": "baseline"}, {"name": "cap 24", "params": {"term": {"cap_by_income": {"cap": 24.0}}}}], "comparison": {"outputs": ["term_cap", "offer"]}}})
print(json.dumps(r))`;
    const result = JSON.parse(execFileSync("uv", ["run", "python", "-c", script], { cwd: ROOT }).toString()) as ExperimentResult;

    expect(result.status).toBe("completed");
    expect(result.summary!.first_divergence).toEqual({ scenario: "cap 24", location: "term_cap[0]" });
    expect(result.sankey).not.toBeNull();
    expect(result.sankey!.nodes.some((n) => n.kind === "arm")).toBe(true);
    const layout = sankeyLayout(result.sankey!);
    expect(layout.links.reduce((t, l) => t + l.value, 0)).toBeGreaterThan(0);
  });
});
