/**
 * V3 Tree type definitions - edge-based structure
 *
 * Key differences from v2:
 * - Nodes don't have embedded `then`/`otherwise`/`branches`
 * - Tree structure defined by edges with `sourceIndex`
 * - Nodes have optional `meta` field for UI position
 * - Node types: leaf, unary, cases_ranges, cases_string_match, cases_isin, composite
 *
 * Edge mapping convention:
 * - UnaryNode: sourceIndex=0 (then), sourceIndex=1 (otherwise)
 * - CompositeNode: sourceIndex=0 (then), sourceIndex=1 (otherwise)
 * - Cases nodes: sourceIndex=0,1,2... (branches), sourceIndex=N (otherwise)
 */

// Re-export API-generated types
export type {
  V3TreeDocument,
  PositionedNode as PositionedNodeV3,
  LeafNode as LeafNodeV3,
  UnaryNode as UnaryNodeV3,
  CompositeCondition as CompositeNodeV3,
  CasesRanges as CasesRangesV3,
  CasesStringMatch as CasesStringMatchV3,
  CasesIsIn as CasesIsInV3,
  IsInCondition,
  ParamRef,
  MultiSourceEdge as EdgeV3,
  MultiEdgeData as EdgeDataV3,
  StringMatchType as TStringMatchType,
  LogicOp as TLogicOp,
  RangeEndLogic,
  Feature,
} from '../api/types.gen';

// ============================================================================
// Constants
// ============================================================================

export const TREE_V3_FORMAT_VERSION = 3;
export const TREE_V3_TYPE = 'ui-tree-v3' as const;

// ============================================================================
// Node Type Discriminators
// ============================================================================

export type NodeTypeV3 =
  | 'leaf'
  | 'unary'
  | 'cases_ranges'
  | 'cases_string_match'
  | 'cases_isin'
  | 'composite';

// ============================================================================
// Source Index Mapping
// ============================================================================

/**
 * Source index constants for unary and composite nodes
 */
export const UnaryNodeSourceIndex = {
  THEN: 0,
  OTHERWISE: 1,
} as const;

/**
 * Source index constants for composite nodes (same as unary)
 */
export const CompositeNodeSourceIndex = {
  THEN: 0,
  OTHERWISE: 1,
} as const;

/**
 * Get source index for cases node branch
 * @param conditionIndex Index of the condition (0-based)
 * @returns Source index for the branch
 */
export function getCasesNodeBranchSourceIndex(conditionIndex: number): number {
  return conditionIndex;
}

/**
 * Get source index for cases node otherwise branch
 * @param numConditions Number of conditions in the cases node
 * @returns Source index for the otherwise branch
 */
export function getCasesNodeOtherwiseSourceIndex(numConditions: number): number {
  return numConditions;
}

// ============================================================================
// Type Guards
// ============================================================================

export { paramRefName, isParamRef as isInputRef } from '@/utils/param-refs';
export type { ParamRefValue as InputRef } from '@/utils/param-refs';

/**
 * Type guard for ComputedFeature
 */
export function isComputedFeature(
  feature: unknown
): feature is { type: 'computed'; expression: string } {
  return (
    typeof feature === 'object' &&
    feature !== null &&
    'type' in feature &&
    (feature as any).type === 'computed' &&
    'expression' in feature &&
    typeof (feature as any).expression === 'string'
  );
}

/**
 * Extract feature name from Feature type (string or ComputedFeature)
 */
export function getFeatureName(feature: string | { type: 'computed'; expression: string }): string {
  if (typeof feature === 'string') {
    return feature;
  }
  return feature.expression;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get all edges from a source node
 */
export function getEdgesFromNode(
  edges: Array<{ source: string; target: string; data: { sourceIndex: number[] } }>,
  nodeId: string
): Array<{ source: string; target: string; data: { sourceIndex: number[] } }> {
  return edges.filter((edge) => edge.source === nodeId);
}

/**
 * Get all edges to a target node
 */
export function getEdgesToNode(
  edges: Array<{ source: string; target: string; data: { sourceIndex: number[] } }>,
  nodeId: string
): Array<{ source: string; target: string; data: { sourceIndex: number[] } }> {
  return edges.filter((edge) => edge.target === nodeId);
}

/**
 * Get target node ID for a specific source index
 */
export function getTargetForSourceIndex(
  edges: Array<{ source: string; target: string; data: { sourceIndex: number[] } }>,
  sourceNodeId: string,
  sourceIndex: number
): string | undefined {
  const edge = edges.find(
    (e) => e.source === sourceNodeId && e.data.sourceIndex.includes(sourceIndex)
  );
  return edge?.target;
}

/**
 * Check if node has any children
 */
export function hasChildren(
  edges: Array<{ source: string; target: string; data: { sourceIndex: number[] } }>,
  nodeId: string
): boolean {
  return edges.some((edge) => edge.source === nodeId);
}

/**
 * Check if node is a root node (no incoming edges)
 */
export function isRootNode(
  edges: Array<{ source: string; target: string; data: { sourceIndex: number[] } }>,
  nodeId: string
): boolean {
  return !edges.some((edge) => edge.target === nodeId);
}

/**
 * Get all root node IDs
 */
export function getRootNodeIds(
  nodes: Array<{ id: string }>,
  edges: Array<{ source: string; target: string; data: { sourceIndex: number[] } }>
): string[] {
  const targetIds = new Set(edges.map((e) => e.target));
  return nodes.filter((n) => !targetIds.has(n.id)).map((n) => n.id);
}

// ============================================================================
// Node Data Unions
// ============================================================================
