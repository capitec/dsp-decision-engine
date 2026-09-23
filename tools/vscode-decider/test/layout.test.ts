import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import type { DescribeResult } from "../src/protocol";
import { dataEdges, layout, orderEdges } from "../webview/layout";

const ROOT = path.resolve(__dirname, "..");
const ir: DescribeResult["ir"] = JSON.parse(
  execFileSync("uv", ["run", "python", "-c", `import sys; sys.path.insert(0, ${JSON.stringify(path.join(ROOT, "python"))})
from bridge import Bridge; import json; print(json.dumps(Bridge().describe(${JSON.stringify(path.join(ROOT, "examples", "loan.py"))})["ir"]))`]).toString(),
);

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
    expect(edges).toContain("term/by_sector/is_private -> term/by_sector/cap_private (arm 0)");
    expect(edges).toContain("term/by_sector/is_private -> term/by_sector/cap_public (arm 1)");
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
