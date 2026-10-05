import { describe, expect, it } from "vitest";
import { filterSort } from "../src/model/datasetRows";

const rows = [{ a: 10, b: "x" }, { a: 9, b: "Xy" }, { a: 11, b: "z" }];

describe("filterSort", () => {
  it("filters case-insensitively and keeps the original row index", () => {
    expect(filterSort(rows, { b: "X" }, null).map((r) => r.i)).toEqual([0, 1]);
  });
  it("combines filters across columns", () => {
    expect(filterSort(rows, { a: "1", b: "x" }, null).map((r) => r.i)).toEqual([0]);
  });
  it("sorts numbers numerically, both ways", () => {
    expect(filterSort(rows, {}, { column: "a", dir: "asc" }).map((r) => r.i)).toEqual([1, 0, 2]);
    expect(filterSort(rows, {}, { column: "a", dir: "desc" }).map((r) => r.i)).toEqual([2, 0, 1]);
  });
});
