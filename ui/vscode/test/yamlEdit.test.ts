import { describe, expect, it } from "vitest";
import { deleteNode, insertNode } from "../src/yamlEdit";

const FLOW = `# credit decisions
flow: credit
steps:
  - step: a   # first
  - branch: b
    when: ok
    modifies: [x]
    then:
      - step: c
`;

describe("insertNode", () => {
  it("inserts after a node and keeps comments", () => {
    const out = insertNode(FLOW, ["steps", 0], "after", { step: "new" });
    expect(out).toContain("# credit decisions");
    expect(out).toContain("- step: a # first\n  - step: new\n  - branch: b");
  });

  it("inserts before a node", () => {
    expect(insertNode(FLOW, ["steps", 1], "before", { step: "new" })).toContain("  - step: new\n  - branch: b");
  });

  it("appends inside the root flow", () => {
    expect(insertNode(FLOW, [], "inside", { step: "z" }).trimEnd().endsWith("- step: z")).toBe(true);
  });

  it("creates a missing else arm", () => {
    const out = insertNode(FLOW, ["steps", 1, "else"], "inside", { step: "d" });
    expect(out).toContain("    else:\n      - step: d");
  });

  it("writes short lists and expressions in flow style", () => {
    const out = insertNode(FLOW, ["steps", 1], "after", { loop: "l", while: { expr: "n > 0" }, carries: ["n"], max: 5, steps: [{ step: "s" }] });
    expect(out).toContain("while: { expr: n > 0 }");
    expect(out).toContain("carries: [ n ]");
  });

  it("writes JSON flow files as JSON", () => {
    const out = insertNode('{"flow": "f", "steps": [{"step": "a"}]}', ["steps", 0], "after", { step: "b" }, true);
    expect(JSON.parse(out)).toEqual({ flow: "f", steps: [{ step: "a" }, { step: "b" }] });
  });
});

describe("deleteNode", () => {
  it("removes the node at a key path", () => {
    expect(deleteNode(FLOW, ["steps", 1, "then", 0])).not.toContain("step: c");
  });
});
