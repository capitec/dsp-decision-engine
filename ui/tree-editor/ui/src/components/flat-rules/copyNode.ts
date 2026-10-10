import type { RuleTableRow } from './types';

/** A deep copy of `node` with fresh ids. A cases node's `otherwise` is a branch index, so it is kept as is. */
export function copyWithNewIds(node: RuleTableRow): RuleTableRow {
  const copy: any = { ...node, id: crypto.randomUUID() };
  if (node.type === 'cases') copy.branches = node.branches?.map(copyWithNewIds);
  if (node.type === 'unary' || node.type === 'composite') {
    copy.then = node.then ? copyWithNewIds(node.then) : undefined;
    copy.otherwise = node.otherwise ? copyWithNewIds(node.otherwise) : undefined;
  }
  return copy;
}
