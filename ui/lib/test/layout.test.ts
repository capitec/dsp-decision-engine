import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import type { CallNodeJson, DescribeResult, GroupNodeJson, IRNodeJson } from "../src/model/protocol";
import { callNodes } from "../src/model/protocol";
import { findSteps } from "../src/FindStep";
import { dataEdges, findNode, fold, layout, orderEdges, touchPoints } from "../src/layout";

// The example flows live with the VS Code extension; the Python bridge is decikit._tooling.bridge.
const ROOT = path.resolve(__dirname, "../../vscode");
const ir: DescribeResult["ir"] = JSON.parse(
  execFileSync("uv", ["run", "python", "-c", `from decikit._tooling.bridge.bridge import Bridge; import json; print(json.dumps(Bridge().describe(${JSON.stringify(path.join(ROOT, "examples", "loan.py"))})["ir"]))`]).toString(),
);

// Three-way fan-out `depth` levels deep: 3^depth steps, so a few thousand nodes in all.
function nested(depth: number, path: string): IRNodeJson {
  if (depth === 0) return { path, source: "", file: null, line: 1, kind: "call", callKind: "scalar", inputs: ["x"], outputs: ["y"], params: {}, python: null, code: "", formula: null } as unknown as CallNodeJson;
  const children = [0, 1, 2].map((i) => nested(depth - 1, `${path}/${i}`));
  return { path, source: "", file: null, line: null, kind: "sequence", children } as GroupNodeJson;
}

describe("graph edges", () => {
  it("data edges feed each read from its last write, never from a sibling arm", () => {
    const edges = dataEdges(ir).map((e) => `${e.from} -${e.column}-> ${e.to}`);
    expect(edges).toContain("term/cap_by_income -term_cap-> term/by_sector/cap_private");
    expect(edges).toContain("term/cap_by_income -term_cap-> term/by_sector/cap_public");
    expect(edges).not.toContain("term/by_sector/cap_private -term_cap-> term/by_sector/cap_public");
    expect(edges).toContain("affordability/ratio -ratio-> banding");
  });

  it("order edges follow written order and fan out of a branch condition", () => {
    const edges = orderEdges(ir).map((e) => `${e.from} -> ${e.to}${e.label ? ` (${e.label})` : ""}`);
    expect(edges).toContain("affordability/affordable -> banding");
    expect(edges).toContain("term/by_sector/is_private -> term/by_sector/cap_private (is_private = true)");
    expect(edges).toContain("term/by_sector/is_private -> term/by_sector/cap_public (is_private = false)");
    expect(edges).toContain("sizing/shrink_offer/too_big -> sizing/shrink_offer/shrink (while)");
    expect(edges).toContain("sizing/shrink_offer/shrink -> sizing/shrink_offer/too_big (repeat)");
    expect(edges).toContain("sizing/shrink_offer/too_big -> risk_tree");
  });

  it("a frame step's declared writes feed later reads", () => {
    const edges = dataEdges(ir).map((e) => `${e.from} -${e.column}-> ${e.to}`);
    expect(edges).toContain("join_bureau -bureau_score-> risk_tree");
  });

  it("lays out every call node inside its cluster", () => {
    const l = layout(ir);
    expect(l.nodes.map((n) => n.path)).toContain("term/by_sector/cap_public");
    const cluster = l.clusters.find((c) => c.path === "term/by_sector")!;
    const inside = l.nodes.filter((n) => n.path.startsWith("term/by_sector/"));
    for (const n of inside) {
      expect(n.x).toBeGreaterThanOrEqual(cluster.x);
      expect(n.y + n.height).toBeLessThanOrEqual(cluster.y + cluster.height + 0.01);
    }
    expect(l.width).toBeGreaterThan(0);
  });
});

describe("large flows", () => {
  it("a folded group is one box, wired in order like a step", () => {
    const folded = fold(ir, (p) => p !== "term");
    const l = layout(folded);
    expect(l.nodes.map((n) => n.path)).toContain("term");
    expect(l.nodes.map((n) => n.path)).not.toContain("term/cap_by_income");
    const box = l.nodes.find((n) => n.path === "term")!.node as { folded?: string[] };
    expect(box.folded).toContain("term/by_sector/cap_public");
    const edges = orderEdges(folded).map((e) => `${e.from} -> ${e.to}`);
    expect(edges).toContain("banding -> term");
    expect(edges).toContain("term -> sizing/offer");
  });

  it("lays out a large, deeply nested flow without crashing", () => {
    const big = nested(7, "root");
    const l = layout(big);
    expect(l.nodes.length).toBeGreaterThan(2000);
    expect(l.width).toBeGreaterThan(0);
    expect(l.height).toBeGreaterThan(0);
  });

  it("find matches a step's description word by word, name hits first", () => {
    const nodes = callNodes(ir).map((n) => (n.path === "term/cap_by_income" ? { ...n, doc: "Cap the term for low earners." } : n));
    expect(findSteps(nodes, "low earners term")[0].path).toBe("term/cap_by_income");
    expect(findSteps(nodes, "cap")[0].path.split("/").pop()).toMatch(/cap/);
    expect(findSteps(nodes, "nothing like this")).toEqual([]);
  });

  it("every read and write of a value lists as a touch point", () => {
    const paths = touchPoints(ir, "term_cap").map((n) => n.path);
    expect(paths).toContain("term/term_cap");
    expect(paths).toContain("term/cap_by_income");
    expect(paths).toContain("term/by_sector/cap_private");
    expect(paths).toContain("term/by_sector/cap_public");
    expect(paths).not.toContain("join_bureau");
  });

  it("findNode resolves a nested path back to its node", () => {
    expect(findNode(ir, "term/by_sector/cap_public")?.kind).toBe("call");
    expect(findNode(ir, "term")?.kind).toBe("sequence");
    expect(findNode(ir, "missing")).toBeUndefined();
  });
});
