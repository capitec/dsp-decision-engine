import { describe, expect, it } from "vitest";
import { breakpointNodes } from "../src/model/breakpoints";
import type { CallNodeJson, GroupNodeJson, IRNodeJson } from "../src/model/protocol";

function call(path: string, inputs: string[], outputs: string[]): CallNodeJson {
  return { path, source: "", file: null, line: null, kind: "call", callKind: "scalar", inputs, outputs, params: {}, python: null, code: "c" };
}

function group(kind: GroupNodeJson["kind"], path: string, children: IRNodeJson[], extra: Partial<GroupNodeJson> = {}): GroupNodeJson {
  return { path, source: "", file: null, line: null, kind, children, ...extra };
}

describe("breakpoint nodes on the graph", () => {
  // A loop whose condition writes `too_big`; a step that writes `offer`.
  const ir = group("sequence", "pipeline", [
    call("sizing/offer", ["requested_amount"], ["offer"]),
    group("loop", "sizing/shrink", [call("sizing/shrink/too_big", ["offer"], ["too_big"]), call("sizing/shrink/step", ["offer"], ["offer"])]),
  ]);

  it("a plain watch marks its step", () => {
    expect([...breakpointNodes(ir, [{ path: "sizing/offer" }], null)]).toEqual([["sizing/offer", "plain"]]);
  });

  it("a value watch marks every step that writes the value", () => {
    const marked = breakpointNodes(ir, [{ name: "offer", op: "<", value: 70000 }], null);
    expect(marked.get("sizing/offer")).toBe("value");
    expect(marked.get("sizing/shrink/step")).toBe("value");
    expect(marked.has("sizing/shrink/too_big")).toBe(false);
  });

  it("an iteration watch marks the loop's condition step", () => {
    const marked = breakpointNodes(ir, [{ path: "sizing/shrink", iteration: 3 }], null);
    expect(marked.get("sizing/shrink/too_big")).toBe("iteration");
  });

  it("a hit marks the path that paused the run", () => {
    const marked = breakpointNodes(ir, [], { watch: 0, text: "offer < 70000", path: "sizing/shrink/step" });
    expect(marked.get("sizing/shrink/step")).toBe("plain");
  });
});
