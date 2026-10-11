import { isMap, isScalar, isSeq, parseDocument } from "yaml";

export type JsonPath = (string | number)[];

export interface ConfigAt {
  /** Keys and indexes from the document root to the value under the cursor. */
  path: JsonPath;
  /** The document's `type`, e.g. "tree". */
  type?: string;
  /** Tree documents: the id of the node (or the name of the rule) the cursor is in. */
  nodeId?: string;
}

const covers = (n: any, offset: number) => !!n?.range && offset >= n.range[0] && offset <= n.range[1];

/** Where `offset` sits in a JSON or YAML config document; undefined when the text has no mapping or list. */
export function configAt(text: string, offset: number): ConfigAt | undefined {
  const doc = parseDocument(text);
  if (!isMap(doc.contents) && !isSeq(doc.contents)) return undefined;
  const path: JsonPath = [];
  let node: unknown = doc.contents;
  while (isMap(node) || isSeq(node)) {
    if (isSeq(node)) {
      const i = node.items.findIndex((it) => covers(it, offset));
      if (i < 0) break;
      path.push(i);
      node = node.items[i];
    } else {
      const pair = node.items.find((p) => covers(p.key, offset) || covers(p.value, offset));
      if (!pair) break;
      path.push(String(isScalar(pair.key) ? pair.key.value : pair.key));
      node = covers(pair.value, offset) ? pair.value : undefined;
    }
  }
  const js = doc.toJS() as any;
  const [root, section, i] = path;
  const at = root === "tree" && typeof i === "number" ? js?.tree?.[section]?.[i] : undefined;
  return { path, type: typeof js?.type === "string" ? js.type : undefined, nodeId: section === "nodes" ? at?.id : section === "rules" ? at?.meta?.name : undefined };
}
