import * as path from "node:path";
import { isMap, isSeq, LineCounter, parseDocument, type Document, type Node } from "yaml";
import { lastSegment, type IRNodeJson } from "@decikit/ui";

export type Kind = "step" | "flow" | "branch" | "loop" | "condition" | "arm";

/** One row of the Structure tree: a node of the flow, and where it is written. */
export interface Item {
  /** Unique in the tree: the node's path, or `<branch path>#then` for an arm. */
  id: string;
  /** The node's path in the described flow (an arm carries its branch's). */
  path: string;
  kind: Kind;
  name: string;
  detail?: string;
  file: string | null;
  /** 1-based. */
  line: number | null;
  /** Flow files only: the node's key path inside `file`, for editing. */
  yamlPath?: (string | number)[];
  /** Flow files only: where a node added inside this one goes (a `flows/` reference's own file). */
  into?: { file: string; yamlPath: (string | number)[] };
  children: Item[];
}

/** Reads a file's text, or `undefined` when there is none. */
export type Read = (file: string) => string | undefined;

const KINDS = ["step", "flow", "branch", "loop"] as const;
const FLOW_EXTS = ["yaml", "yml", "json"];

/** Whether `text` is a decikit flow file: a top-level `flow:` name and `steps:` list. */
export function isFlowFile(text: string): boolean {
  return /^flow:\s*\S/m.test(text) && /^steps:/m.test(text);
}

/** A flow file's outline, `flow: name` references followed into `flows/<name>.yaml`. */
export function yamlOutline(file: string, read: Read): Item {
  const ctx: Ctx = { read, seen: new Set(), chain: [] };
  const parsed = parse(file, read)!;
  const name = String(parsed.doc.get("flow") ?? path.basename(file));
  const root = make(ctx, { path: name, kind: "flow", name, file, line: 1, yamlPath: [], into: { file, yamlPath: [] } });
  ctx.chain.push(name);
  root.children = list(ctx, parsed, file, ["steps"], name);
  return root;
}

/** The outline of a described pipeline (a `.py` one, or a flow file Python could load). */
export function irOutline(n: IRNodeJson, name?: string): Item {
  const kind: Kind = n.kind === "call" ? "step" : n.kind === "sequence" ? "flow" : n.kind;
  return {
    id: n.path || "<root>",
    path: n.path,
    kind,
    name: n.path === "" ? name ?? "pipeline" : lastSegment(n.path),
    file: n.file ?? (n.kind === "call" ? n.python?.file ?? null : null),
    line: n.line ?? (n.kind === "call" ? n.python?.line ?? null : null),
    children: n.kind === "call" ? [] : n.children.map((c) => irOutline(c)),
  };
}

/** The flow file a `flow: name` with no `steps:` loads, as the loader looks for it. */
export function subflowFile(file: string, name: string, read: Read): string | undefined {
  const dir = path.dirname(file);
  const project = path.basename(dir) === "flows" ? path.dirname(dir) : dir;
  return FLOW_EXTS.map((e) => path.join(project, "flows", `${name}.${e}`)).find((f) => read(f) !== undefined);
}

export function walkItems(item: Item, fn: (i: Item, parent: Item | undefined) => void, parent?: Item) {
  fn(item, parent);
  for (const c of item.children) walkItems(c, fn, item);
}

interface Ctx {
  read: Read;
  seen: Set<string>;
  chain: string[];
}

interface Parsed {
  doc: Document;
  lines: LineCounter;
}

function parse(file: string, read: Read): Parsed | undefined {
  const text = read(file);
  if (text === undefined) return undefined;
  const lines = new LineCounter();
  return { doc: parseDocument(text, { lineCounter: lines }), lines };
}

function make(ctx: Ctx, it: Omit<Item, "id" | "children">, suffix = ""): Item {
  let id = it.path + suffix;
  for (let i = 2; ctx.seen.has(id); i++) id = `${it.path}${suffix}~${i}`;
  ctx.seen.add(id);
  return { ...it, id, children: [] };
}

function lineOf(p: Parsed, n: Node | null | undefined): number | null {
  return n?.range ? p.lines.linePos(n.range[0]).line : null;
}

function list(ctx: Ctx, p: Parsed, file: string, keyPath: (string | number)[], prefix: string): Item[] {
  const seq = p.doc.getIn(keyPath, true);
  if (!isSeq(seq)) return [];
  return seq.items.flatMap((n, i) => {
    const it = isMap(n) ? node(ctx, p, file, [...keyPath, i], n, prefix) : undefined;
    return it ? [it] : [];
  });
}

function node(ctx: Ctx, p: Parsed, file: string, yamlPath: (string | number)[], n: Node, prefix: string): Item | undefined {
  if (!isMap(n)) return undefined;
  const kind = KINDS.find((k) => n.has(k));
  if (!kind) return undefined;
  const name = String(n.get(kind === "step" && n.has("as") ? "as" : kind));
  const at = `${prefix}/${name}`;
  const it = make(ctx, { path: at, kind, name, file, line: lineOf(p, n), yamlPath });
  const js = n.toJSON() as Record<string, unknown>;
  if (kind === "step") {
    if (js.as) it.detail = String(js.step);
  } else if (kind === "flow") {
    if (n.has("steps")) {
      it.children = list(ctx, p, file, [...yamlPath, "steps"], at);
      it.into = { file, yamlPath };
    } else {
      const sub = ctx.chain.includes(name) ? undefined : subflowFile(file, name, ctx.read);
      const subParsed = sub ? parse(sub, ctx.read) : undefined;
      it.detail = sub ? path.relative(path.dirname(file), sub) : "no flows/ file";
      if (sub && subParsed) {
        it.into = { file: sub, yamlPath: [] };
        ctx.chain.push(name);
        it.children = list(ctx, subParsed, sub, ["steps"], at);
        ctx.chain.pop();
      }
    }
  } else if (kind === "branch") {
    it.detail = `modifies ${joined(js.modifies)}`;
    it.children.push(condition(ctx, p, file, n, "when", at, name));
    for (const arm of ["then", "else"]) {
      const armPath = [...yamlPath, arm];
      const a = make(ctx, { path: at, kind: "arm", name: arm, file, line: lineOf(p, n.get(arm, true) as Node), into: { file, yamlPath: armPath } }, `#${arm}`);
      if (arm === "else" && !n.has("else")) a.detail = "false rows pass through";
      a.children = list(ctx, p, file, armPath, at);
      it.children.push(a);
    }
  } else {
    it.detail = `max ${js.max} · carries ${joined(js.carries)}`;
    it.into = { file, yamlPath };
    it.children = [condition(ctx, p, file, n, "while", at, name), ...list(ctx, p, file, [...yamlPath, "steps"], at)];
  }
  return it;
}

function condition(ctx: Ctx, p: Parsed, file: string, n: Node, key: "when" | "while", at: string, name: string): Item {
  const raw = isMap(n) ? (n.toJSON() as Record<string, unknown>)[key] : undefined;
  const expr = raw && typeof raw === "object" ? String((raw as { expr?: unknown }).expr) : undefined;
  const condName = expr === undefined ? String(raw) : `${name}_${key}`;
  const line = isMap(n) ? lineOf(p, n.get(key, true) as Node) : null;
  return make(ctx, { path: `${at}/${condName}`, kind: "condition", name: expr === undefined ? condName : key, detail: expr ?? key, file, line });
}

function joined(v: unknown): string {
  return Array.isArray(v) ? v.join(", ") : String(v ?? "?");
}
