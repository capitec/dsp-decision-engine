import * as fs from "node:fs";
import * as path from "node:path";
import * as vscode from "vscode";
import { NODE_TYPES, type NodeType } from "./structure";

const NAME = /^[A-Za-z_]\w*$/;

/** Ask which node type to insert. */
export async function pickNodeType(): Promise<NodeType["kind"] | undefined> {
  const picked = await vscode.window.showQuickPick(
    NODE_TYPES.map((t) => ({ label: `$(${t.icon.id}) ${t.label}`, detail: t.detail, type: t.kind })),
    { placeHolder: "Node type to insert" },
  );
  return picked?.type;
}

/**
 * Ask for what a new node needs and return it as flow-file data, or `undefined`
 * if the user cancels. `names` lists the steps the flow's components offer.
 */
export async function promptNode(kind: NodeType["kind"], flowFile: string, names: () => Promise<string[]>): Promise<object | undefined> {
  if (kind === "step") {
    const step = await pickStep(names, "Step to run");
    return step ? { step } : undefined;
  }
  if (kind === "subflow") return pickSubflow(flowFile, names);
  const name = await ask(`Name of the new ${kind}`);
  if (!name) return undefined;
  if (kind === "flow") {
    const first = await pickStep(names, `First step in ${name}`);
    return first ? { flow: name, steps: [{ step: first }] } : undefined;
  }
  const cond = await pickCondition(names, kind === "branch" ? "Rows take `then` when…" : "Repeat while…");
  if (cond === undefined) return undefined;
  const cols = await ask(kind === "branch" ? "Columns the arms write (modifies), comma separated" : "Columns carried between iterations, comma separated", true);
  if (!cols) return undefined;
  const list = cols.split(",").map((s) => s.trim()).filter(Boolean);
  if (kind === "branch") {
    const then = await pickStep(names, "Step the `then` arm runs");
    return then ? { branch: name, when: cond, modifies: list, then: [{ step: then }] } : undefined;
  }
  const max = await vscode.window.showInputBox({ prompt: "Most iterations (max)", value: "10", validateInput: (v) => (/^\d+$/.test(v) ? null : "A whole number") });
  if (!max) return undefined;
  const body = await pickStep(names, "Step the loop repeats");
  return body ? { loop: name, while: cond, carries: list, max: Number(max), steps: [{ step: body }] } : undefined;
}

async function ask(prompt: string, list = false): Promise<string | undefined> {
  return vscode.window.showInputBox({
    prompt,
    validateInput: (v) => (list ? (v.trim() ? null : "Name at least one column") : NAME.test(v) ? null : "Letters, digits and underscores"),
  });
}

async function pickStep(names: () => Promise<string[]>, title: string): Promise<string | undefined> {
  return pickName(names, title, [{ label: "$(edit) Another name…", value: "" }]);
}

async function pickCondition(names: () => Promise<string[]>, title: string): Promise<string | { expr: string } | undefined> {
  const picked = await pickName(names, `${title} (a bool column, a step, or an expression)`, [
    { label: "$(symbol-operator) Expression…", value: "expr" },
    { label: "$(edit) A bool column…", value: "" },
  ]);
  if (picked !== "\0expr") return picked;
  const expr = await vscode.window.showInputBox({ prompt: "Condition, e.g. score >= 600 and band == \"A\"", validateInput: (v) => (v.trim() ? null : "Write an expression") });
  return expr ? { expr } : undefined;
}

/** A component name from a quick pick; an `extra` with value "" asks for a name, any other returns "\0<value>". */
async function pickName(names: () => Promise<string[]>, title: string, extra: { label: string; value: string }[]): Promise<string | undefined> {
  const qp = vscode.window.createQuickPick<vscode.QuickPickItem & { value: string; special?: boolean }>();
  qp.title = title;
  qp.placeholder = "Loading the flow's components…";
  qp.busy = true;
  qp.items = extra.map((e) => ({ ...e, special: true }));
  qp.show();
  names().then(
    (found) => {
      qp.items = [...found.map((n) => ({ label: `$(symbol-function) ${n}`, value: n })), ...qp.items];
      qp.placeholder = "Pick one, or type to filter";
    },
    (e) => (qp.placeholder = `Couldn't list components: ${(e as Error).message}`),
  ).finally(() => (qp.busy = false));
  const choice = await new Promise<(typeof qp.items)[number] | undefined>((resolve) => {
    // A typed name that matches nothing is taken as is.
    qp.onDidAccept(() => resolve(qp.selectedItems[0] ?? (NAME.test(qp.value) ? { label: qp.value, value: qp.value } : undefined)));
    qp.onDidHide(() => resolve(undefined));
  });
  qp.dispose();
  if (!choice) return undefined;
  if (!choice.special) return choice.value;
  if (choice.value) return `\0${choice.value}`;
  return ask("Name");
}

async function pickSubflow(flowFile: string, names: () => Promise<string[]>): Promise<object | undefined> {
  const dir = path.dirname(flowFile);
  const flows = path.join(path.basename(dir) === "flows" ? path.dirname(dir) : dir, "flows");
  const existing = fs.existsSync(flows) ? fs.readdirSync(flows).filter((f) => /\.(ya?ml|json)$/.test(f)).map((f) => f.replace(/\.\w+$/, "")) : [];
  const picked = await vscode.window.showQuickPick(
    [...existing.map((n) => ({ label: `$(file-symlink-file) ${n}`, value: n })), { label: "$(new-file) New flows/ file…", value: "" }],
    { placeHolder: "Subflow to reference" },
  );
  if (!picked) return undefined;
  if (picked.value) return { flow: picked.value };
  const name = await ask("Name of the new subflow (flows/<name>.yaml)");
  const first = name && (await pickStep(names, `First step in ${name}`));
  if (!name || !first) return undefined;
  fs.mkdirSync(flows, { recursive: true });
  fs.writeFileSync(path.join(flows, `${name}.yaml`), `flow: ${name}\nsteps:\n  - step: ${first}\n`);
  return { flow: name };
}
