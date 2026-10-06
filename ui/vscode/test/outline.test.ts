import * as fs from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { isFlowFile, walkItems, yamlOutline, type Item } from "../src/outline";

const EXAMPLE = path.join(__dirname, "..", "examples", "credit_flow");
const read = (f: string) => (fs.existsSync(f) ? fs.readFileSync(f, "utf8") : undefined);

function byId(root: Item): Map<string, Item> {
  const out = new Map<string, Item>();
  walkItems(root, (i) => out.set(i.id, i));
  return out;
}

describe("yamlOutline", () => {
  const root = yamlOutline(path.join(EXAMPLE, "flow.yaml"), read);
  const items = byId(root);

  it("names nodes by the paths the loader gives them", () => {
    expect([...items.keys()]).toEqual([
      "credit", "credit/screen", "credit/screen/eligible", "credit/card",
      "credit/affordable", "credit/affordable/eligible", "credit/affordable#then", "credit/affordable/full_limit",
      "credit/affordable#else", "credit/affordable/no_limit",
      "credit/payoff", "credit/payoff/payoff_while", "credit/payoff/pay",
    ]);
  });

  it("follows a flow reference into its flows/ file", () => {
    const eligible = items.get("credit/screen/eligible")!;
    expect(eligible.file).toBe(path.join(EXAMPLE, "flows", "screen.yaml"));
    expect([eligible.line, eligible.yamlPath]).toEqual([3, ["steps", 0]]);
    expect(items.get("credit/screen")!.into).toEqual({ file: eligible.file, yamlPath: [] });
  });

  it("records each node's line and key path in its own file", () => {
    const pay = items.get("credit/payoff/pay")!;
    expect([pay.line, pay.yamlPath]).toEqual([19, ["steps", 3, "steps", 0]]);
    expect(items.get("credit/payoff/payoff_while")!.detail).toBe("balance > 0");
  });

  it("shows a missing else arm as passing rows through", () => {
    const text = "flow: f\nsteps:\n  - branch: b\n    when: c\n    modifies: [x]\n    then: [{step: s}]\n";
    const only = byId(yamlOutline("/p/flow.yaml", (f) => (f === "/p/flow.yaml" ? text : undefined)));
    expect(only.get("f/b#else")!.detail).toBe("false rows pass through");
  });
});

describe("isFlowFile", () => {
  it("wants a top-level flow name and steps list", () => {
    expect(isFlowFile("flow: credit\nsteps:\n  - step: a\n")).toBe(true);
    expect(isFlowFile("name: ci\nsteps:\n  - run: x\n")).toBe(false);
  });
});
