import { describe, expect, it } from "vitest";
import { datasetLabel, filterSort, isUnder, projectRoots } from "../src/datasetFormat";


describe("datasets", () => {
  it("describes a keyed local dataset", () => {
    expect(datasetLabel({ name: "a", type: "folder", key: ["client_id"], path: "/x" })).toEqual({
      description: "folder · key client_id",
      contextValue: "dataset dataset-local",
    });
  });
  it("marks a broken entry", () => {
    expect(datasetLabel({ name: "a", type: "?", key: null, path: null, error: "bad" }).contextValue).toBe("dataset-broken");
  });
  const rows = [{ a: 10, b: "x" }, { a: 9, b: "Xy" }, { a: 11, b: "z" }];
  it("filters per column, case-insensitively, keeping each row's index", () => {
    expect(filterSort(rows, { b: "X" }, null).map((r) => r.i)).toEqual([0, 1]);
    expect(filterSort(rows, { a: "1", b: "x" }, null).map((r) => r.i)).toEqual([0]);
  });
  it("sorts numbers numerically, both ways", () => {
    expect(filterSort(rows, {}, { column: "a", dir: "asc" }).map((r) => r.i)).toEqual([1, 0, 2]);
    expect(filterSort(rows, {}, { column: "a", dir: "desc" }).map((r) => r.i)).toEqual([2, 0, 1]);
  });
  it("finds project roots above configs, deduped", () => {
    expect(projectRoots(["/w/b/configs/x.yaml", "/w/a/configs/y/z.yaml", "/w/a/configs/q.yaml", "/w/other.py"])).toEqual(["/w/a", "/w/b"]);
  });
  it("tells whether a file is inside a project", () => {
    expect(isUnder("/w/a", "/w/a/flows/f.py")).toBe(true);
    expect(isUnder("/w/a", "/w/ab/f.py")).toBe(false);
  });
});
