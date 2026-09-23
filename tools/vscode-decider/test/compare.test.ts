import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { compareTraces, diffDoc, paramChangeLines, paramReaders, same, type TraceResult } from "../src/compare";
import { matchRow, substitute } from "../webview/Explain";
import { listRefs, materialise } from "../src/git";
import { scenarios, summariseSweep } from "../src/sweep";

const ROOT = path.resolve(__dirname, "..");
const LOAN = path.join(ROOT, "examples", "loan.py");

function trace(file: string, extra: Record<string, unknown> = {}): TraceResult {
  const script = `import sys, json; sys.path.insert(0, ${JSON.stringify(path.join(ROOT, "python"))})
from bridge import Bridge
print(json.dumps(Bridge().trace(${JSON.stringify(file)}, **json.loads(sys.argv[1])), default=str))`;
  return JSON.parse(execFileSync("uv", ["run", "python", "-c", script, JSON.stringify(extra)], { cwd: ROOT }).toString());
}

describe("comparing two runs", () => {
  const base = trace(LOAN);

  it("a param change shows where outputs first diverge, and on which rows", () => {
    const tuned = trace(LOAN, { params: { term: { cap_by_income: { cap: 24.0 } } } });
    const c = compareTraces(base, tuned, "defaults", "cap 24");
    expect(c.firstDivergence).toBe("term/cap_by_income");
    const cap = c.steps.find((s) => s.path === "term/cap_by_income")!;
    expect(cap.status).toBe("changed");
    expect(cap.outputs).toEqual([{ name: "term_cap", changedRows: [0, 1], samples: [{ row: 0, a: 48, b: 24 }, { row: 1, a: 36, b: 24 }] }]);
    expect(c.steps.find((s) => s.path === "banding")!.status).toBe("same");
    expect(c.output.map((o) => o.name)).toEqual(["term_cap"]);
  });

  it("an input override for one record only changes that record", () => {
    const whatIf = trace(LOAN, { overrides: { requested_amount: 10000.0 }, row: 0 });
    const c = compareTraces(base, whatIf, "sample", "record 0 borrows 10000");
    expect(c.firstDivergence).toBe("sizing/offer");
    expect(c.output.find((o) => o.name === "offer")!.changedRows).toEqual([0]);
  });

  it("a removed and an added step are slotted into execution order", () => {
    const a = structuredClone(base);
    const b = structuredClone(base);
    const seq = b.ir.kind === "call" ? null : b.ir;
    seq!.children = seq!.children.filter((c) => c.path !== "banding");
    (a.ir as typeof seq)!.children = (a.ir as typeof seq)!.children.filter((c) => c.path !== "risk_tree");
    const c = compareTraces(a, b, "a", "b");
    const at = (p: string) => c.steps.findIndex((s) => s.path === p);
    expect(c.steps[at("banding")].status).toBe("removed");
    expect(at("banding")).toBe(at("affordability/affordable") + 1);
    expect(c.steps[at("risk_tree")].status).toBe("added");
  });

  it("numbers compare with a relative tolerance", () => {
    expect(same(0.1 + 0.2, 0.3)).toBe(true);
    expect(same(1, 1.001)).toBe(false);
    expect(same(null, 0)).toBe(false);
  });
});

describe("scenario sweeps", () => {
  it("every combination of knobs becomes a scenario with nested params and overrides", () => {
    const sc = scenarios(
      [
        { kind: "param", key: "term/cap_by_income|cap", values: [24, 36] },
        { kind: "value", key: "requested_amount", values: [1000, 2000] },
      ],
      1,
    );
    expect(sc.map((s) => s.label)).toEqual([
      "term/cap_by_income · cap = 24, requested_amount = 1000",
      "term/cap_by_income · cap = 24, requested_amount = 2000",
      "term/cap_by_income · cap = 36, requested_amount = 1000",
      "term/cap_by_income · cap = 36, requested_amount = 2000",
    ]);
    expect(sc[3]).toMatchObject({ params: { term: { cap_by_income: { cap: 36 } } }, overrides: { requested_amount: 2000 }, row: 1 });
    expect(scenarios([], null)).toEqual([]);
  });

  it("summarises forks against the original: changed columns and a step diff each", () => {
    const script = `import sys, json; sys.path.insert(0, ${JSON.stringify(path.join(ROOT, "python"))})
from bridge import Bridge
b = Bridge(); b.start(${JSON.stringify(LOAN)}, breakpoints=["term/cap_by_income"]); b.handle({"cmd": "resume"})
print(json.dumps(b.sweep([{"label": "cap 6", "params": {"term": {"cap_by_income": {"cap": 6.0}}}}]), default=str))`;
    const r = JSON.parse(execFileSync("uv", ["run", "python", "-c", script], { cwd: ROOT }).toString());
    const s = summariseSweep(r);
    expect(s.at).toBe("before term/cap_by_income");
    expect(s.changedColumns).toEqual(["term_cap"]);
    expect(s.outputs[0]!.term_cap).toEqual([6, 6]);
    expect(s.comparisons[0].firstDivergence).toBe("term/cap_by_income");
  });
});

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

describe("param readers", () => {
  it("a shared param is read by the steps that declare it; a step param by its step", () => {
    const doc = { shared: { repo_rate: 0.075, pl_base_rates: [{ rate: 1 }] }, term: { cap_by_income: { cap: 42 } } };
    expect(paramReaders(doc, { repo_rate: ["a/x", "b/y"] })).toEqual([
      { param: "repo_rate", steps: ["a/x", "b/y"] },
      { param: "pl_base_rates", steps: [] },
      { param: "cap", steps: ["term/cap_by_income"] },
    ]);
  });
});

describe("explaining a value", () => {
  it("fills a formula with the record's values", () => {
    expect(substitute("min(pl_raw_rate, repo_rate * cap_multiple + cap_margin)", { pl_raw_rate: 0.252, repo_rate: 0.0775, cap_multiple: 1, cap_margin: 0.21 }))
      .toBe("min(0.252, 0.0775 * 1 + 0.21)");
  });

  it("finds a band's row, lower edge included", () => {
    const rows = [{ lo: 6, hi: 25 }, { lo: 25, hi: 49 }, { lo: 49, hi: 85 }];
    const expr = { type: "between", variable: "term", lower_bound_column: "lo", upper_bound_column: "hi" };
    expect(matchRow(expr, rows, 49)).toEqual([2, "49 ≤ term 49 < 85"]);
    expect(matchRow(expr, rows, 85)).toBeNull();
  });

  it("says what changed in a params document, row by row for a table", () => {
    const a = { shared: { repo_rate: 0.0775, t: [{ lo: 1, r: 0.2 }, { lo: 2, r: 0.3 }] } };
    const b = { shared: { repo_rate: 0.0775, t: [{ lo: 1, r: 0.2 }, { lo: 2, r: 0.25 }] } };
    expect(diffDoc(a, b)).toEqual({ shared: { t: b.shared.t } });
    expect(paramChangeLines(diffDoc(a, b), a)).toEqual(["t row 2: r 0.3 → 0.25"]);
    expect(paramChangeLines({ shared: { repo_rate: 0.075 } }, a)).toEqual(["repo_rate: 0.0775 → 0.075"]);
  });
});
