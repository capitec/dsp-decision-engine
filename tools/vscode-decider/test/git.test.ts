import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { listRefs, materialise } from "../src/git";

describe("git revisions", () => {
  it("lists refs and extracts a revision's tree without touching the working tree", async () => {
    const repo = fs.mkdtempSync(path.join(os.tmpdir(), "decider-git-"));
    const g = (...args: string[]) => execFileSync("git", ["-c", "user.name=t", "-c", "user.email=t@t", ...args], { cwd: repo });
    g("init", "-q");
    fs.writeFileSync(path.join(repo, "flow.py"), "v = 1\n");
    g("add", ".");
    g("commit", "-qm", "first");
    g("tag", "v1");
    fs.writeFileSync(path.join(repo, "flow.py"), "v = 2\n");
    g("commit", "-qam", "second");
    const refs = await listRefs(repo);
    expect(refs[0].label).toMatch(/^HEAD \(/); // HEAD, its branch and its sha are one entry
    expect(refs.some((r) => r.label.startsWith("v1"))).toBe(true);
    expect(refs[0].description).toMatch(/^last commit: second/);
    expect(refs.length).toBe(2);
    const cache = path.join(repo, ".cache");
    const dir = await materialise(repo, "v1", cache);
    expect(fs.readFileSync(path.join(dir, "flow.py"), "utf8")).toBe("v = 1\n");
    expect(await materialise(repo, "v1", cache)).toBe(dir);
    expect(fs.readFileSync(path.join(repo, "flow.py"), "utf8")).toBe("v = 2\n");
  });
});
