import { isMap, isScalar, isSeq, parseDocument, visit, type Document } from "yaml";

export type Where = "before" | "after" | "inside";

/**
 * `text` with `node` inserted next to the node at `yamlPath`, or appended inside
 * it (a flow's or loop's `steps`, or a branch arm). Comments and layout survive.
 */
export function insertNode(text: string, yamlPath: (string | number)[], where: Where, node: object, json = false): string {
  const doc = parseDocument(text);
  const created = doc.createNode(node);
  visit(created, { Seq: (_, s) => void (s.items.every(isScalar) && (s.flow = true)), Map: (_, m) => void (m.has("expr") && (m.flow = true)) });
  if (where === "inside") {
    const target = isSeq(doc.getIn(yamlPath, true)) || isArm(yamlPath) ? yamlPath : [...yamlPath, "steps"];
    const seq = doc.getIn(target, true);
    if (isSeq(seq)) seq.items.push(created);
    else doc.setIn(target, doc.createNode([node]));
  } else {
    const seq = doc.getIn(yamlPath.slice(0, -1), true);
    if (!isSeq(seq)) throw new Error(`no list at ${yamlPath.slice(0, -1).join(".")}`);
    seq.items.splice((yamlPath.at(-1) as number) + (where === "after" ? 1 : 0), 0, created);
  }
  return dump(doc, json);
}

/** `text` without the node at `yamlPath`. */
export function deleteNode(text: string, yamlPath: (string | number)[], json = false): string {
  const doc = parseDocument(text);
  doc.deleteIn(yamlPath);
  return dump(doc, json);
}

/** The root flow's name and its `components:` module, for a new `flows/` file. */
export function rootInfo(text: string): { flow?: string; components?: string } {
  const doc = parseDocument(text);
  return isMap(doc.contents) ? { flow: doc.get("flow") as string, components: doc.get("components") as string } : {};
}

function isArm(yamlPath: (string | number)[]): boolean {
  const last = yamlPath.at(-1);
  return last === "then" || last === "else";
}

function dump(doc: Document, json: boolean): string {
  return json ? JSON.stringify(doc.toJS(), null, 2) + "\n" : doc.toString({ lineWidth: 0 });
}
