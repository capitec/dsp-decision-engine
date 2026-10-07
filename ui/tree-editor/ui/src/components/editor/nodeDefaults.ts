/**
 * Centralized default values for node types
 * Used by both display components and editors to ensure consistency
 */

import {
  UnaryNodeData,
  CasesRangesNodeData,
  CasesStringMatchNodeData,
  CasesIsInNodeData,
  LeafNodeData,
} from './types';

// Default values for unary nodes (v3)
export const UNARY_NODE_DEFAULTS = {
  condition: {
    op: '<=' as const,
    threshold: 0,
    feature: '',
  }
} satisfies Partial<UnaryNodeData>;

// Default values for cases_isin nodes (v3 - replaces categorical)
export const CASES_ISIN_DEFAULTS = {
  feature: '',
  conditions: [],
} satisfies Partial<CasesIsInNodeData>;

// Default values for cases_ranges nodes (v3)
export const CASES_RANGES_DEFAULTS = {
  feature: '',
  conditions: [],
} satisfies Partial<CasesRangesNodeData>;

// Default values for cases_string_match nodes (v3)
export const CASES_STRING_MATCH_DEFAULTS = {
  feature: '',
  conditions: [],
} satisfies Partial<CasesStringMatchNodeData>;

// Default values for leaf nodes
export const LEAF_DEFAULTS = {
  result_idx: -1, // -1 = use default output
} satisfies Partial<LeafNodeData>;

/**
 * Get default node data for a specific node type (v3)
 * For cases nodes, pass "cases:op" format (e.g., "cases:ranges")
 */
export function getNodeDefaults(nodeType: string): Partial<any> {
  switch (nodeType) {
    case 'unary':
      return UNARY_NODE_DEFAULTS;
    case 'cases:isin':
    case 'cases_isin': // backwards compat
      return CASES_ISIN_DEFAULTS;
    case 'cases:ranges':
    case 'cases_ranges': // backwards compat
      return CASES_RANGES_DEFAULTS;
    case 'cases:string_match':
    case 'cases_string_match': // backwards compat
      return CASES_STRING_MATCH_DEFAULTS;
    case 'leaf':
      return LEAF_DEFAULTS;
    default:
      return {};
  }
}

/**
 * Merge node data with defaults (node data takes precedence)
 */
export function withDefaults<T extends { type: string }>(nodeData: T): T {
  const defaults = getNodeDefaults(nodeData.type);
  return { ...defaults, ...nodeData } as T;
}
