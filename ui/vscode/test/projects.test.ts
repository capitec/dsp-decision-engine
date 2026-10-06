import { describe, expect, it } from "vitest";
import { groupFlows, projectRoot } from "../src/projects";

const ref = (file: string, pipeline?: string) => ({ file, pipeline, label: pipeline ?? "f", line: 0 });

describe("projects", () => {
  const dirs = new Set(["/w/credit/configs", "/w/bank/configs"]);
  it("finds the nearest ancestor holding configs", () => {
    expect(projectRoot("/w/credit/flows/screen.yaml", (p) => dirs.has(p))).toBe("/w/credit");
    expect(projectRoot("/w/loan.py", (p) => dirs.has(p))).toBeUndefined();
  });
  const rootOf = (f: string) => ["/w/credit", "/w/bank"].find((r) => f.startsWith(r + "/"));
  it("puts root flow files in main and flows/ files in subflows", () => {
    const [g] = groupFlows([ref("/w/credit/flow.yaml"), ref("/w/credit/flows/screen.yaml")], rootOf);
    expect(g.main.map((f) => f.file)).toEqual(["/w/credit/flow.yaml"]);
    expect(g.sub.map((f) => f.file)).toEqual(["/w/credit/flows/screen.yaml"]);
  });
  it("makes build the main flow of pipeline.py and the module-level flows subflows", () => {
    const f = [ref("/w/bank/pipeline.py", "build"), ref("/w/bank/pipeline.py", "risk"), ref("/w/bank/products/cc/policy.py", "cc_policy")];
    const [g] = groupFlows(f, rootOf);
    expect(g.main.map((x) => x.pipeline)).toEqual(["build"]);
    expect(g.sub.map((x) => x.pipeline)).toEqual(["risk", "cc_policy"]);
  });
  it("makes every pipeline main when pipeline.py has no build", () => {
    const [g] = groupFlows([ref("/w/bank/pipeline.py", "a"), ref("/w/bank/pipeline.py", "b")], rootOf);
    expect(g.main).toHaveLength(2);
  });
  it("lists projects by path, flows outside any project last", () => {
    const g = groupFlows([ref("/w/loan.py", "x"), ref("/w/credit/flow.yaml"), ref("/w/bank/pipeline.py", "build")], rootOf);
    expect(g.map((x) => x.root)).toEqual(["/w/bank", "/w/credit", undefined]);
  });
});
