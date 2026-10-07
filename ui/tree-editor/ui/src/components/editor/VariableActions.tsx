import { Node } from './types';
import { collectParamRefs, renameParamRefs, clearParamRefs } from '@/utils/param-refs';

/** Rename a parameter across all nodes (thresholds, bounds, isin values, patterns, composites). */
export function renameParameterInNodes(
  nodes: Node[],
  oldName: string,
  newName: string
): Node[] {
  return nodes.map(node =>
    countParameterUsages([node], oldName) ? { ...node, data: renameParamRefs(node.data, oldName, newName) } : node
  );
}

/**
 * Delete a parameter by removing it from all nodes.
 * Thresholds reset to 0, range bounds to null, isin values and patterns to empty.
 */
export function deleteParameterFromNodes(
  nodes: Node[],
  paramName: string
): Node[] {
  return nodes.map(node =>
    countParameterUsages([node], paramName) ? { ...node, data: clearParamRefs(node.data, paramName) } : node
  );
}

/** Extract all parameter names referenced by nodes. */
export function extractParameterReferences(nodes: Node[]): string[] {
  return Array.from(collectParamRefs(nodes.map(n => n.data))).sort();
}

/** Count how many nodes use a specific parameter (each node once). */
export function countParameterUsages(
  nodes: Node[],
  paramName: string
): number {
  return nodes.filter(node => collectParamRefs(node.data).has(paramName)).length;
}
