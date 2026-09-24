import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { compareTraces, diffDoc, paramChangeLines, paramReaders, same, type TraceResult } from "../src/model/compare";
import { clampNote, evaluate, matchRow, substitute, sumTerms, summaryRows } from "../src/Explain";
import { scenarios, summariseSweep } from "../src/model/sweep";

// The Python bridge and the example flows live with the VS Code extension.
const ROOT = path.resolve(__dirname, "../../vscode-decider");
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

  it("a step only the first run took is not taken, not removed", () => {
    const a = structuredClone(base);
    const b = structuredClone(base);
    delete b.steps["banding"];
    expect(compareTraces(a, b, "a", "b").steps.find((s) => s.path === "banding")?.status).toBe("not taken");
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
      .toBe("min(25.2%, 7.75% + 21%)");
    expect(substitute("pl_base_rate + pl_risk_loading - pl_loyalty_discount", { pl_base_rate: 0.255, pl_risk_loading: 0, pl_loyalty_discount: 0.001 })).toBe("25.5% - 0.1%");
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
    expect(paramChangeLines(diffDoc(a, b), a)).toEqual(["t row 2 (lo 2): r 0.3 → 0.25"]);
    expect(paramChangeLines({ shared: { repo_rate: 0.075 } }, a)).toEqual(["repo_rate: 7.75% → 7.5%"]);
  });
});

describe("caps and floors", () => {
  it("evaluates a limit's arithmetic and says how far off it was", () => {
    expect(evaluate("repo_rate * cap_multiple + cap_margin", { repo_rate: 0.0775, cap_multiple: 1, cap_margin: 0.21 })).toBeCloseTo(0.2875);
    expect(evaluate("(a + 1) * -2", { a: 2 })).toBe(-6);
    expect(evaluate("f(a)", { a: 1 })).toBeNull();
    const known = { pl_raw_rate: 0.252, repo_rate: 0.0775, cap_multiple: 1, cap_margin: 0.21 };
    const inputs = [{ name: "pl_raw_rate", value: 0.252 }];
    expect(clampNote("min(pl_raw_rate, repo_rate * cap_multiple + cap_margin)", inputs, 0.252, known)).toBe("cap 28.75%, not reached (3.55 pp below)");
    expect(clampNote("min(pl_raw_rate, repo_rate * cap_multiple + cap_margin)", [{ name: "pl_raw_rate", value: 0.3 }], 0.2875, known)).toBe("the cap applied: 30% → 28.75%");
  });
});

describe("sums as waterfalls", () => {
  it("reads a plus/minus chain of names as signed terms, and nothing else", () => {
    expect(sumTerms("pl_base_rate + pl_risk_loading - pl_loyalty_discount")).toEqual([["+", "pl_base_rate"], ["+", "pl_risk_loading"], ["-", "pl_loyalty_discount"]]);
    expect(sumTerms("min(a, b)")).toBeNull();
    expect(sumTerms("a * b")).toBeNull();
    expect(sumTerms("a")).toBeNull();
  });
});

describe("a rate's summary", () => {
  it("lists every part, zeros marked, then each limit and the final value", () => {
    const node = (path: string, formula: string, params: Record<string, unknown> = {}) => ({ kind: "call", path, source: "", file: null, line: null, callKind: "scalar", inputs: [], outputs: [], params, python: null, code: "", formula });
    const nodes = [
      node("p/raw", "base + loading - discount"),
      node("p/cap", "min(raw_rate, repo_rate + margin)", { repo_rate: 0.0775, margin: 0.21 }),
      node("p/floor", "max(rate, repo_rate + margin)", { repo_rate: 0.0775, margin: 0.03 }),
    ];
    const raw = { name: "raw_rate", producer: "p/raw", value: 0.254, inputs: [
      { name: "base", producer: "p/base", value: 0.255, inputs: [] },
      { name: "loading", producer: "p/loading", value: 0, inputs: [] },
      { name: "discount", producer: "p/discount", value: 0.001, inputs: [] },
    ] };
    const capped = { name: "rate", producer: "p/cap", value: 0.254, inputs: [raw] };
    const floored = { name: "rate", producer: "p/floor", value: 0.254, inputs: [capped] };
    const rows = summaryRows(floored, nodes as never, {})!;
    expect(rows.map((r) => [r.kind, r.label, r.value])).toEqual([
      ["part", "base", "0.255"],
      ["zero", "loading", "+ 0%"],
      ["part", "discount", "− 0.1%"],
      ["total", "= raw_rate", "25.4%"],
      ["limit", "cap (cap)", "28.75%"],
      ["limit", "floor (floor)", "10.75%"],
      ["total", "= rate", "25.4%"],
    ]);
  });
});
