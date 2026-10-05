import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import {
  experimentHeadline,
  flattenSummary,
  plotsForMetric,
  plotTypesFor,
  reportOf,
  scenarioDefSummary,
  statusLabel,
  summaryTable,
  typedFields,
  type ExperimentDef,
  type ExperimentResult,
  type ExperimentSchema,
} from "../src/model/experiment";

const DEF: ExperimentDef = {
  name: "drift",
  flow: { entry: "credit:pipeline" },
  scenarios: [{ name: "limit_0.3", params: { credit: { approved: { limit: 0.3 } } } }],
  metrics: [{ type: "divergence", column: "approved" }],
  plots: [],
};

const completed = (status: ExperimentResult["status"] = "completed"): ExperimentResult => ({
  name: "drift",
  status,
  nondeterministic: status === "non_reproducible",
  scenarios: [
    { name: "baseline", status: "completed", records: 2, params: {}, overrides: {} },
    { name: "cap_24", status: status === "partial" ? "failed" : "completed", error: status === "partial" ? "bad value" : undefined, records: 2, params: {}, overrides: {} },
  ],
  summaries: { divergence: { cap_24: { changed: 1, first: 0 } } },
  plots: {},
  definition: DEF,
});

describe("experiment run states", () => {
  it("names non-reproducible and partial runs, never reading them as done", () => {
    expect(experimentHeadline(completed())).toContain("2 scenarios");
    expect(experimentHeadline(completed("non_reproducible"))).toContain("non-reproducible");
    expect(experimentHeadline(completed("partial"))).toContain("partial");
    expect(experimentHeadline(completed("cancelled"))).toContain("cancelled");
  });

  it("labels each status with its own word", () => {
    expect(statusLabel("non_reproducible")).toBe("non-reproducible");
    expect(statusLabel("timed_out")).toBe("timed out");
    expect(statusLabel("completed")).toBe("completed");
  });
});

describe("the report to render", () => {
  it("falls back to an overview and a section per metric without a report of its own", () => {
    const r = completed();
    r.definition = { ...DEF, report: null };
    expect(reportOf(r).blocks).toEqual([{ type: "overview" }, { type: "metric_section", metric: "divergence" }]);
  });

  it("uses the definition's own report when it has one", () => {
    const report = { title: "x", blocks: [{ type: "text", markdown: "hi" }] };
    expect(reportOf({ ...completed(), definition: { ...DEF, report } }).blocks).toEqual(report.blocks);
  });
});

describe("summary tables", () => {
  it("flattens nested fields and truncates long arrays", () => {
    expect(flattenSummary({ changed: 1, bins: { counts: [1, 2, 3, 4, 5, 6, 7] } })).toEqual({
      changed: "1",
      "bins.counts": "[1, 2, 3, 4, 5, 6, …] (7)",
    });
  });

  it("unions columns across scenarios that give different fields", () => {
    const { columns, rows } = summaryTable({ a: { x: 1 }, b: { y: 2 } });
    expect(columns).toEqual(["scenario", "x", "y"]);
    expect(rows).toEqual([{ scenario: "a", x: "1" }, { scenario: "b", y: "2" }]);
  });
});

describe("plot and metric pairing", () => {
  const schema: ExperimentSchema = {
    metrics: { divergence: { properties: {}, summary: "Divergences" }, histogram: { properties: {}, summary: "Bins" } },
    plots: { changes: { properties: {}, requires: "Divergences" }, distribution: { properties: {}, requires: "Bins" } },
    blocks: {},
  };

  it("offers only the plot types whose summary matches the metric's", () => {
    expect(plotTypesFor(schema, "divergence")).toEqual(["changes"]);
    expect(plotTypesFor(schema, "histogram")).toEqual(["distribution"]);
  });

  it("finds the plots that draw a metric by its key", () => {
    const def: ExperimentDef = { ...DEF, plots: [{ type: "changes", metric: "divergence" }, { type: "changes", metric: "divergence", name: "d2" }] };
    expect(plotsForMetric(def, "divergence")).toEqual(["divergence", "d2"]);
  });
});

describe("read-only summary helpers", () => {
  it("describes a scenario definition's own changes", () => {
    expect(scenarioDefSummary({ name: "x" })).toBe("unchanged");
    expect(scenarioDefSummary({ name: "x", params: { a: 1 } })).toBe('params {"a":1}');
    expect(scenarioDefSummary({ name: "x", overrides: { a: 1 }, at: "step", row: 2 })).toBe('set {"a":1} after step on record 2');
    expect(scenarioDefSummary({ name: "x", revision: "HEAD^" })).toBe("revision HEAD^");
  });

  it("lists a typed object's fields beyond the ones already shown, skipping empties", () => {
    expect(typedFields({ type: "histogram", column: "payment", bins: 20, range: null }, new Set(["type"]))).toEqual([
      ["column", "payment"],
      ["bins", "20"],
    ]);
  });
});

describe("a real experiment run end to end", () => {
  it("runs the new contract against a real pipeline and summarises a divergence", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "decider-ui-experiment-"));
    fs.writeFileSync(
      path.join(dir, "credit.py"),
      "from decider import flow, param\n" +
        "def ratio(income, debt):\n" +
        "    return debt / income if income > 0 else 1.0\n" +
        "def approved(ratio, limit=param(0.4)):\n" +
        "    return ratio <= limit\n" +
        "pipeline = flow(ratio, approved, name='credit')\n",
    );
    const data = [
      { income: 1000.0, debt: 200.0 },
      { income: 1000.0, debt: 350.0 },
    ];
    const script = `import io, json
from decider._bridge.bridge import Bridge
b = Bridge(out=io.StringIO())  # progress lines go to a throwaway stream, not stdout's one JSON reply
r = b.handle({"id": 1, "cmd": "experiment", "file": ${JSON.stringify(path.join(dir, "credit.py"))}, "def_": ${JSON.stringify(DEF)}, "data": ${JSON.stringify(data)}})
print(json.dumps(r))`;
    const result = JSON.parse(execFileSync("uv", ["run", "python", "-c", script], { cwd: path.resolve(__dirname, "../../vscode-decider") }).toString()) as ExperimentResult;

    expect(result.status).toBe("completed");
    expect(result.summaries.divergence["limit_0.3"].changed).toBe(1);
    const { rows } = summaryTable(result.summaries.divergence);
    expect(rows.map((r) => r.scenario)).toEqual(["baseline", "limit_0.3"]);
  });
});
