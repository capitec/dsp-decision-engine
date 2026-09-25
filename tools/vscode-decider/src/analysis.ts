import * as path from "node:path";
import * as vscode from "vscode";
import { withBridge } from "./bridge";
import { type DescribeResult } from "@decider/ui";
import { pythonCommand } from "./python";

const DECIDER_IMPORT = /^\s*(from|import)\s+decider\b/m;

const cache = new Map<string, { version: number; result: Promise<DescribeResult> }>();

/**
 * Import the file in a short-lived bridge and describe its pipelines. Nothing
 * runs on data, but module-level code does execute, like any import.
 */
export function analyse(doc: vscode.TextDocument): Promise<DescribeResult> {
  const key = doc.uri.toString();
  const hit = cache.get(key);
  if (hit && hit.version === doc.version) return hit.result;
  const result = withBridge({ python: pythonCommand(), cwd: path.dirname(doc.fileName) }, (b) =>
    b.request<DescribeResult>("describe", { file: doc.fileName }),
  );
  cache.set(key, { version: doc.version, result });
  result.catch(() => cache.delete(key));
  return result;
}

/** CodeLenses above every module-level pipeline: run it, or draw it. */
export class PipelineCodeLens implements vscode.CodeLensProvider {
  private changed = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this.changed.event;

  async provideCodeLenses(doc: vscode.TextDocument): Promise<vscode.CodeLens[]> {
    if (!DECIDER_IMPORT.test(doc.getText())) return [];
    try {
      const d = await analyse(doc);
      return d.pipelines.flatMap((p) => {
        const range = new vscode.Range(Math.max((p.line ?? 1) - 1, 0), 0, Math.max((p.line ?? 1) - 1, 0), 0);
        const args = [doc.uri, p.name];
        return [
          new vscode.CodeLens(range, { title: "$(play) Run flow", command: "decider.runFlow", arguments: args }),
          new vscode.CodeLens(range, { title: "$(type-hierarchy) Visualise flow", command: "decider.visualise", arguments: args }),
          new vscode.CodeLens(range, { title: "$(beaker) What-if", command: "decider.whatIf", arguments: [doc.uri] }),
          new vscode.CodeLens(range, { title: "$(git-compare) Compare with…", command: "decider.compareRevision", arguments: [doc.uri] }),
        ];
      });
    } catch (e) {
      const range = new vscode.Range(0, 0, 0, 0);
      return [new vscode.CodeLens(range, { title: `decider: cannot analyse (${firstLine((e as Error).message)})`, command: "" })];
    }
  }
}

export function firstLine(s: string): string {
  return s.split("\n")[0].slice(0, 120);
}
