// Runs inside the extension host: CodeLens detection, the views and a debug session end to end.
import * as assert from "node:assert";
import * as path from "node:path";
import Mocha from "mocha";
import * as vscode from "vscode";

export function run(): Promise<void> {
  const mocha = new Mocha({ ui: "bdd", timeout: 60000, color: true });
  mocha.suite.emit("pre-require", globalThis, "decider", mocha);
  defineTests();
  return new Promise((resolve, reject) => mocha.run((failures) => (failures ? reject(new Error(`${failures} tests failed`)) : resolve())));
}

function defineTests() {
  const loan = vscode.Uri.file(path.join(__dirname, "..", "..", "examples", "loan.py"));

  describe("decider extension", () => {
    it("puts Run and Visualise lenses above the pipeline", async () => {
      const doc = await vscode.workspace.openTextDocument(loan);
      await vscode.window.showTextDocument(doc);
      const lenses = await waitFor(async () => {
        const l = (await vscode.commands.executeCommand<vscode.CodeLens[]>("vscode.executeCodeLensProvider", loan, 10)) ?? [];
        return l.some((x) => x.command?.command === "decider.runFlow") ? l : undefined;
      });
      const titles = lenses.map((l) => l.command!.title);
      assert.ok(titles.some((t) => t.includes("Run flow")), titles.join());
      assert.ok(titles.some((t) => t.includes("Visualise flow")), titles.join());
      assert.strictEqual(lenses[0].range.start.line, 55);
    });

    it("visualises the flow without an error", async () => {
      await vscode.commands.executeCommand("decider.visualise", loan, "pipeline");
      await waitFor(async () => (vscode.window.tabGroups.all.some((g) => g.tabs.some((t) => t.label === "decider: flow")) ? true : undefined), 5000);
    });

    it("runs a session that stops on entry and answers custom requests", async () => {
      const started = await vscode.debug.startDebugging(undefined, {
        type: "decider",
        request: "launch",
        name: "test",
        program: loan.fsPath,
        stopOnEntry: true,
      });
      assert.ok(started);
      const session = await waitFor(async () => vscode.debug.activeDebugSession?.type === "decider" ? vscode.debug.activeDebugSession : undefined);
      const d = (await session.customRequest("decider.describe")) as { pipeline: string };
      assert.strictEqual(d.pipeline, "pipeline");
      const info = (await session.customRequest("decider.info")) as { current: { path: string } | null; debugpyPort?: number };
      assert.deepStrictEqual(info.current, { path: "", phase: "start", depth: 0 });
      await vscode.debug.stopDebugging(session);
    });
  });
}

async function waitFor<T>(probe: () => Promise<T | undefined>, ms = 30000): Promise<T> {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    const v = await probe();
    if (v !== undefined) return v;
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error("timed out");
}
