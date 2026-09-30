import { callNodes, walk, type CallNodeJson, type ColumnSummary, type DescribeResult, type GroupNodeJson, type IRNodeJson, type RunStatus } from "../src/model/protocol";

// Representative describe payloads for the standalone render harness: a small flow with every
// node kind, and a large nested flow that starts folded.

function call(path: string, inputs: string[], outputs: string[], callKind: CallNodeJson["callKind"], extra: Partial<CallNodeJson> = {}): CallNodeJson {
  return {
    path,
    source: `def ${path.split("/").pop()}()`,
    file: "pipeline.py",
    line: 1,
    kind: "call",
    callKind,
    inputs,
    outputs,
    params: {},
    python: { file: "pipeline.py", line: 1, bodyLine: 1 },
    code: "abc123",
    formula: outputs[0] ? `${outputs[0]} = f(${inputs.join(", ")})` : null,
    ...extra,
  };
}

function seq(path: string, children: IRNodeJson[], extra: Partial<GroupNodeJson> = {}): GroupNodeJson {
  return { path, source: "", file: null, line: null, kind: "sequence", children, ...extra };
}

function branch(path: string, cond: IRNodeJson, arms: IRNodeJson[], extra: Partial<GroupNodeJson> = {}): GroupNodeJson {
  return { path, source: "", file: null, line: null, kind: "branch", modifies: arms[0].kind === "call" ? arms[0].outputs ?? [] : [], children: [cond, ...arms], ...extra };
}

function loop(path: string, cond: IRNodeJson, body: IRNodeJson, extra: Partial<GroupNodeJson> = {}): GroupNodeJson {
  return { path, source: "", file: null, line: null, kind: "loop", carries: body.kind === "call" ? body.outputs ?? [] : [], maxIterations: 20, children: [cond, body], ...extra };
}

function describeOf(pipeline: string, ir: IRNodeJson, pipelines: DescribeResult["pipelines"]): DescribeResult {
  const nodes = callNodes(ir);
  let total = 0;
  walk(ir, () => total++);
  return { pipelines, pipeline, size: { nodes: total, calls: nodes.length }, ir, params: {}, fields: {} };
}

/** The loan example's shape: a frame join, a nested group, a branch and a loop. */
export function smallFlow(): DescribeResult {
  const ir = seq("pipeline", [
    call("join_bureau", ["client_id"], ["bureau_score"], "frame", { doc: "join the bureau score onto each record" }),
    seq("affordability", [
      call("affordability/disposable_income", ["net_income", "expenses"], ["disposable_income"], "scalar"),
      call("affordability/ratio", ["disposable_income", "instalment"], ["ratio"], "scalar"),
      call("affordability/affordable", ["ratio"], ["affordable"], "scalar", { params: { min_ratio: 0.3 } }),
    ]),
    call("banding", ["ratio"], ["band", "band_score"], "scalar", { formula: "band, band_score = (1, 10) if ratio > 2 else (0, 0)" }),
    seq("term", [
      call("term/term_cap", ["requested_term"], ["term_cap"], "scalar", { params: { ceiling: 60 } }),
      call("term/cap_by_income", ["term_cap", "min_net_salary"], ["term_cap"], "scalar", { params: { cap: 48 } }),
      branch("term/by_sector", call("term/by_sector/is_private", ["sector_code"], ["is_private"], "scalar"), [
        call("term/by_sector/cap_private", ["term_cap"], ["term_cap"], "scalar", { params: { cap: 54 } }),
        call("term/by_sector/cap_public", ["term_cap"], ["term_cap"], "scalar", { params: { cap: 60 } }),
      ]),
    ]),
    seq("sizing", [
      call("sizing/offer", ["requested_amount"], ["offer"], "scalar"),
      loop("sizing/shrink_offer", call("sizing/shrink_offer/too_big", ["offer", "disposable_income"], ["too_big"], "scalar"), call("sizing/shrink_offer/shrink", ["offer"], ["offer"], "scalar")),
    ]),
    call("risk_tree", ["bureau_score", "ratio"], ["risk_band"], "row", { doc: "walk the tree to a risk band" }),
  ]);
  return describeOf("pipeline", ir, [{ name: "pipeline", line: 114, kind: "flow", status: "pipeline" }]);
}

/** A deep, wide flow (~2200 steps) that starts fully folded, so only the top groups draw. */
export function largeFlow(): DescribeResult {
  const depth = 6;
  let i = 0;
  const make = (d: number, path: string): IRNodeJson => {
    if (d === 0) return call(path, [`c${Math.max(0, i - 1)}`], [`c${i++}`, `c${i}`], "scalar");
    const kids: IRNodeJson[] = [];
    for (let a = 0; a < 3; a++) kids.push(make(d - 1, `${path}/${a}`));
    return seq(path, kids);
  };
  const ir = seq("root", [make(depth, "root/0"), make(depth, "root/1"), make(depth, "root/2")]);
  return describeOf("pipeline", ir, [{ name: "pipeline", line: 1, kind: "build", status: "pipeline" }]);
}

/** A paused run over `smallFlow`, with breakpoints and state, to audit the debug states. */
export function pausedRun(): { status: RunStatus; columns: ColumnSummary[] } {
  const col = (name: string, preview: unknown[], value: unknown, producer = "input", extra: Partial<ColumnSummary> = {}): ColumnSummary => ({
    name, dtype: "Float64", rows: 2, nulls: 0, preview, value, producer, versions: 1, ...extra,
  });
  const columns = [
    col("client_id", [1, 2], 1, "input", { dtype: "Int64" }),
    col("net_income", [9000, 7200], 9000),
    col("expenses", [4000, 3600], 4000),
    col("disposable_income", [5000, 3600], 5000, "affordability/disposable_income"),
    col("ratio", [0.25, 0.32], 0.25, "affordability/ratio"),
    col("term_cap", [60, 48], 60, "term/cap_by_income", { writtenBy: "term/cap_by_income", changedBy: "term/cap_by_income" }),
    col("offer", [150000, 120000], 150000, "sizing/offer", { writtenBy: "override@term/cap_by_income", changedBy: "override@term/cap_by_income" }),
  ];
  const status: RunStatus = {
    current: { path: "risk_tree", when: "after" },
    finished: false,
    finishedPaths: ["join_bureau", "affordability/disposable_income", "affordability/ratio", "affordability/affordable", "banding", "term/term_cap", "term/cap_by_income", "sizing/offer", "risk_tree"],
    visits: { risk_tree: { "bureau_score < 680": 1, "ratio >= 3": 1, "bureau_score >= 680": 1 } },
    record: null,
    hit: { watch: 0, text: "risk_tree#bureau_score < 680", path: "risk_tree" },
    trace: { available: false, reason: "trace capture is off; showing live state" },
  };
  return { status, columns };
}
