import { describe, expect, it } from "vitest";
import type { IRNodeJson } from "@decider/ui";
import { breakpointPaths, fromRepr, onlyAt, onlyAtPath, pausedAt } from "../src/locate";

const call = (path: string, line: number, endLine: number) => ({
  path, kind: "call", callKind: "scalar", source: "", file: null, line: null, inputs: [], outputs: [], params: {}, code: "",
  python: { file: "/p/steps.py", line, bodyLine: line + 1, endLine },
});
const IR = {
  path: "credit", kind: "sequence", source: "", file: null, line: null,
  children: [call("credit/screen/eligible", 4, 5), call("credit/affordable/eligible", 4, 5), call("credit/limit", 8, 12)],
} as IRNodeJson;

describe("breakpointPaths", () => {
  it("marks every node a shared function runs for", () => {
    expect([...breakpointPaths(IR, [{ file: "/p/steps.py", line: 5 }])]).toEqual(["credit/screen/eligible", "credit/affordable/eligible"]);
  });

  it("narrows to the node an onlyAt condition names", () => {
    const bp = { file: "/p/steps.py", line: 5, condition: onlyAt("credit/affordable/eligible") };
    expect([...breakpointPaths(IR, [bp])]).toEqual(["credit/affordable/eligible"]);
  });

  it("ignores lines outside every step", () => {
    expect(breakpointPaths(IR, [{ file: "/p/steps.py", line: 7 }, { file: "/p/other.py", line: 9 }]).size).toBe(0);
  });
});

describe("pausedAt", () => {
  it("takes the innermost frame inside a step, past helpers", () => {
    expect(pausedAt(IR, [{ file: "/p/helpers.py", line: 3 }, { file: "/p/steps.py", line: 11 }])).toBe("credit/limit");
  });
});

it("reads a condition and a repr back", () => {
  expect(onlyAtPath(onlyAt('a/"b"'))).toBe('a/"b"');
  expect(onlyAtPath("x > 1")).toBeUndefined();
  expect(fromRepr("'credit/limit'")).toBe("credit/limit");
  expect(fromRepr("None")).toBeUndefined();
});
