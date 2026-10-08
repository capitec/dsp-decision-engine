// Runs inside VS Code: `pnpm test:vscode`. Shows the YAML example, sets a breakpoint on one use of a
// shared step, debugs the flow under debugpy, and checks the pause is mapped to that node.
const assert = require("node:assert");
const crypto = require("node:crypto");
const fs = require("node:fs");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");
const vscode = require("vscode");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function until(what, fn, ms = 30000) {
  for (const end = Date.now() + ms; Date.now() < end; await sleep(200)) if (fn()) return;
  throw new Error(`timed out waiting for ${what}`);
}

/** Send one action over the window's agent socket, as `decider mcp` does. */
function agent(folder, action, token) {
  const name = crypto.createHash("sha256").update(fs.realpathSync(folder)).digest("hex").slice(0, 16);
  const dir = path.join(os.homedir(), ".decider", "editor");
  token ??= fs.readFileSync(path.join(dir, `${name}.token`), "utf8");
  return new Promise((resolve, reject) => {
    let buf = "";
    const c = net.connect(path.join(dir, `${name}.sock`), () => c.write(JSON.stringify({ token, action }) + "\n"));
    c.on("data", (d) => (buf += d));
    c.on("end", () => resolve(JSON.parse(buf)));
    c.on("error", reject);
  });
}

exports.run = async function () {
  const api = await vscode.extensions.getExtension("capitec.decider-vscode").activate();
  const flow = path.join(vscode.workspace.workspaceFolders[0].uri.fsPath, "credit_flow", "flow.yaml");

  await vscode.commands.executeCommand("decider.showFlow", vscode.Uri.file(flow));
  assert.ok(api.state().described, `describe failed: ${api.state().error}`);
  const named = (await api.datasets()).map((d) => d.name ?? d.message);
  assert.deepStrictEqual(named.sort(), ["applicants", "portfolio"], "the shown flow's project datasets");

  await vscode.commands.executeCommand("decider.toggleBreakpoint", { path: "credit/affordable/eligible" });
  const [bp] = vscode.debug.breakpoints;
  assert.match(bp.condition, /node_path\(\) == "credit\/affordable\/eligible"/);

  const debugging = vscode.commands.executeCommand("decider.debugFlow", vscode.Uri.file(flow));
  // The dataset picker opens once Python has listed the datasets, which takes longer on a cold start.
  await until("the dataset pick", () => {
    void vscode.commands.executeCommand("workbench.action.acceptSelectedQuickOpenItem");
    return vscode.debug.activeDebugSession;
  });
  await debugging;
  await until("the pause", () => api.state().paused);
  assert.strictEqual(api.state().paused, "credit/affordable/eligible");

  await vscode.commands.executeCommand("workbench.action.debug.continue");
  await until("the run to end", () => !vscode.debug.activeDebugSession);
  vscode.debug.removeBreakpoints(vscode.debug.breakpoints);

  // With a flow shown, debugging a dataset record starts at once on that flow: no flow picker.
  let launched;
  const sub = vscode.debug.onDidStartDebugSession((s) => (launched = s.configuration));
  await vscode.commands.executeCommand("decider.debugWithDataset", { name: "applicants", cwd: path.dirname(flow) }, 2);
  await until("the dataset run to start", () => launched, 10000);
  sub.dispose();
  assert.deepStrictEqual(launched.args, [flow, "--dataset", "applicants", "--row", "2"]);
  await until("the dataset run to end", () => !vscode.debug.activeDebugSession);

  // The agent socket: what an MCP host sees of this window.
  const root = vscode.workspace.workspaceFolders[0].uri.fsPath;
  const ok = async (action) => {
    const r = await agent(root, action);
    assert.ok(r.ok, `${action.kind}: ${r.error}`);
    return r.result;
  };
  const shown = await ok({ kind: "show_flow", file: "credit_flow/flow.yaml" });
  assert.ok(shown.described, `show_flow: ${shown.error}`);
  assert.strictEqual((await ok({ kind: "highlight", nodes: ["credit/affordable/eligible"] })).highlighted, "credit/affordable/eligible");
  assert.deepStrictEqual((await ok({ kind: "selection" })).structure_selection, ["credit/affordable/eligible"]);
  assert.deepStrictEqual((await ok({ kind: "set_breakpoints", nodes: ["credit/affordable/eligible"] })).breakpoints, ["credit/affordable/eligible"]);
  assert.strictEqual((await ok({ kind: "debug_flow", file: "credit_flow/flow.yaml" })).paused, "credit/affordable/eligible");
  const sel = await ok({ kind: "selection" });
  assert.strictEqual(sel.debug.paused, "credit/affordable/eligible");
  assert.ok((await ok({ kind: "debug_evaluate", expression: "1 + 1" })).result.includes("2"));
  await ok({ kind: "set_breakpoints", nodes: [] });
  await ok({ kind: "debug_control", action: "continue" });
  await until("the agent's run to end", () => !vscode.debug.activeDebugSession);

  const tree = path.join(root, "bank", "trees", "collections_risk.json");
  const doc = await vscode.workspace.openTextDocument(tree);
  const at = doc.positionAt(doc.getText().indexOf('"has_judgement"'));
  await vscode.window.showTextDocument(doc, { selection: new vscode.Range(at, at) });
  const cursor = (await ok({ kind: "selection" })).config;
  assert.strictEqual(cursor.nodeId, "has_judgement", JSON.stringify(cursor));
  assert.deepStrictEqual(await agent(root, { kind: "selection" }, "wrong"), { ok: false, error: "unauthorized" });
};

