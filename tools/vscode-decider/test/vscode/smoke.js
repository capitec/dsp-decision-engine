// Runs inside VS Code: `pnpm test:vscode`. Shows the YAML example, sets a breakpoint on one use of a
// shared step, debugs the flow under debugpy, and checks the pause is mapped to that node.
const assert = require("node:assert");
const path = require("node:path");
const vscode = require("vscode");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function until(what, fn, ms = 30000) {
  for (const end = Date.now() + ms; Date.now() < end; await sleep(200)) if (fn()) return;
  throw new Error(`timed out waiting for ${what}`);
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
  await sleep(1000);
  await vscode.commands.executeCommand("workbench.action.acceptSelectedQuickOpenItem");
  await debugging;
  await until("the pause", () => api.state().paused);
  assert.strictEqual(api.state().paused, "credit/affordable/eligible");

  await vscode.commands.executeCommand("workbench.action.debug.continue");
  await until("the run to end", () => !vscode.debug.activeDebugSession);
  vscode.debug.removeBreakpoints(vscode.debug.breakpoints);
};
