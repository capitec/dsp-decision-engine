import { describe, expect, it } from "vitest";
import { datasetLabel, htmlTable, isUnder, projectRoots } from "../src/datasetFormat";


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
  it("escapes cell values and shows dtypes", () => {
    const html = htmlTable([["a", "int64"]], [{ a: "<b>" }]);
    expect(html).toContain("&#60;b&#62;");
    expect(html).toContain("int64");
  });
  it("finds project roots above configs, deduped", () => {
    expect(projectRoots(["/w/b/configs/x.yaml", "/w/a/configs/y/z.yaml", "/w/a/configs/q.yaml", "/w/other.py"])).toEqual(["/w/a", "/w/b"]);
  });
  it("tells whether a file is inside a project", () => {
    expect(isUnder("/w/a", "/w/a/flows/f.py")).toBe(true);
    expect(isUnder("/w/a", "/w/ab/f.py")).toBe(false);
  });
});
