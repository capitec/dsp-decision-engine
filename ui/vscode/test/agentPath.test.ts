import { describe, expect, it } from "vitest";
import { configAt } from "../src/agentPath";

const TREE = JSON.stringify(
  { type: "tree", name: "grade", tree: { nodes: [{ id: "root", data: { type: "unary", condition: { op: ">=", feature: "score" } } }, { id: "leaf_a", data: { type: "leaf" } }], edges: [] } },
  null,
  2,
);
const at = (text: string, needle: string, delta = 0) => configAt(text, text.indexOf(needle) + delta);

describe("configAt", () => {
  it("gives the key path and node id under the cursor in a tree document", () => {
    expect(at(TREE, '"feature"', 3)).toMatchObject({ path: ["tree", "nodes", 0, "data", "condition", "feature"], nodeId: "root", type: "tree" });
    expect(at(TREE, "leaf_a")).toMatchObject({ path: ["tree", "nodes", 1, "id"], nodeId: "leaf_a" });
  });

  it("reports the value path when the cursor is inside a value", () => {
    expect(at(TREE, ">=", 1)?.path).toEqual(["tree", "nodes", 0, "data", "condition", "op"]);
  });

  it("has no node id outside the nodes list", () => {
    expect(at(TREE, '"name"', 2)).toMatchObject({ path: ["name"], nodeId: undefined });
  });

  it("names a flat rule by its meta name", () => {
    const flat = JSON.stringify({ type: "tree", tree: { type: "prioritized_flat_rule", rules: [{ meta: { name: "CF-1" }, rule: { op: "and" } }] } }, null, 1);
    expect(at(flat, '"op"', 2)).toMatchObject({ path: ["tree", "rules", 0, "rule", "op"], nodeId: "CF-1" });
  });

  it("walks YAML documents", () => {
    const yaml = "type: table\nrows:\n  - band: low\n  - band: high\n";
    expect(at(yaml, "high")).toMatchObject({ path: ["rows", 1, "band"], type: "table" });
  });

  it("returns undefined for text that is not a document", () => {
    expect(configAt("", 0)).toBeUndefined();
    expect(configAt("just a string", 3)).toBeUndefined();
  });
});
