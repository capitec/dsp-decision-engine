/**
 * Shared type definitions for tree-based editors (v2 format)
 *
 * These types mirror the legacy v2 backend structures (current documents: see api/types.gen.ts).
 *
 * Both the tree editor and flat rules editor import from here.
 */

import type { PolarsSchema, PolarsType } from './datasource-types';
import { isParamRef, type ParamRefValue } from '@/utils/param-refs';

// Re-export for convenience so consumers don't need two imports
export type { PolarsSchema, PolarsType };

// ============================================================================
// Format Version
// ============================================================================

/**
 * Tree format version constant.
 * Frontend only supports v2. Backend converts v1→v2 on fetch.
 */
export const TREE_FORMAT_VERSION = 2;

// ============================================================================
// Parameter System (v2)
// ============================================================================

/**
 * Parameter reference used throughout the application.
 * At runtime, values come from the parameters column.
 *
 * Backend: legacy v2 model ParameterReference
 */
export type ParameterReference = ParamRefValue;

/** Type guard for ParameterReference (reads `{param}` and legacy `{key}`) */
export const isParameterReference = isParamRef;

// ============================================================================
// Range End Logic (Shared)
// ============================================================================

/**
 * Range boundary logic for multi-way range branching.
 * Determines which branch a boundary value belongs to.
 *
 * Backend: legacy v2 model RangeEndLogic (enum)
 */
export type RangeEndLogic = 'lower_inclusive' | 'upper_inclusive';

// ============================================================================
// Custom Type Definitions (v2)
// ============================================================================

/**
 * Struct type definition with data records.
 *
 * Backend: legacy v2 model StructTypeDef
 * UI addition: display_field (not in backend, used for dropdown rendering)
 */
export interface UITypeDefStruct {
  name: string;
  type: 'struct';
  definition: {
    /** Ordered field definitions: [field_name, polars_type] */
    fields: Array<[string, PolarsType]>;
    /** Available struct instances for selection */
    data: Array<Record<string, any>>;
    /** Field to use for dropdown display (UI-only, not in backend model) */
    display_field: string;
  };
}

/**
 * Categorical type definition with string categories.
 *
 * Backend: legacy v2 model CategoricalTypeDef
 */
export interface UITypeDefCategorical {
  name: string;
  type: 'categorical';
  definition: {
    categories: string[];
  };
}

/** Union of all custom type definitions, discriminated by `type` */
export type UITypeDef = UITypeDefStruct | UITypeDefCategorical;

/** Dictionary of custom type definitions keyed by UUID */
export type UITypeDefs = Record<string, UITypeDef>;

// ============================================================================
// Tree Output (v2)
// ============================================================================

/**
 * Unified tree output structure (v2 format).
 * Single source of truth for output schema, data, and defaults.
 *
 * Backend: legacy v2 model TreeOutput
 * Inherits dtypes/type_defs from ContainsDtypes
 */
export interface TreeOutput {
  /** Output records that leaf nodes reference by index */
  data: Array<Record<string, any>>;
  /** Default output when no leaf matches (leaf_value = -1) */
  default: Record<string, any> | null;
  /** Output schema as ordered [field_name, type] tuples */
  dtypes: PolarsSchema;
  /** Custom type definitions keyed by UUID */
  type_defs: UITypeDefs;
  /** Display format template for output labels (e.g., "${Action}: ${Description}"). Defaults to first column if not set. */
  display_fmt?: string;
}

// ============================================================================
// Parameter System
// ============================================================================

/**
 * Parameter definition with type and default value.
 * Parameters provide runtime values for ParameterReferences.
 *
 * Backend: legacy v2 model ParameterInfo
 */
export interface ParameterInfo {
  type: PolarsType;
  default_value: any;
}

/** Parameters keyed by name */
export type ParameterMap = Record<string, ParameterInfo>;

// ============================================================================
// Node Types (v2)
// ============================================================================

/** Statistics data optionally attached to nodes (e.g. from training) */
interface BaseNodeData {
  data_count?: number;
  sum_hess?: number;
  gain?: number;
}

/**
 * Numerical range test node - multi-way branching on sorted thresholds.
 * Creates N+1 branches for N thresholds.
 *
 * Backend: legacy v2 model RangeNode
 */
export interface V2RangeNodeData extends BaseNodeData {
  node_type: 'numerical_range_test_node';
  feature: string;
  thresholds: number[] | ParameterReference[];
  end_logic: RangeEndLogic;
}

/**
 * Numerical comparison node - binary branch on single threshold.
 *
 * Backend: legacy v2 model NumericalNode
 */
export interface V2NumericalNodeData extends BaseNodeData {
  node_type: 'numerical_test_node';
  feature: string;
  comparison_op: '<=' | '<' | '==' | '>' | '>=' | '!=';
  threshold: number | ParameterReference;
}

/**
 * Categorical membership test node - binary branch.
 * Match (sourceIndex 0) / no-match (sourceIndex 1).
 *
 * Backend: legacy v2 model CategoricalNode
 */
export interface V2CategoricalNodeData extends BaseNodeData {
  node_type: 'categorical_test_node';
  feature: string;
  category_list: number[] | ParameterReference;
}

/**
 * String pattern matching node.
 * match_any=true: binary branch (OR logic)
 * match_any=false: multi-way branch (one per pattern + default)
 *
 * Backend: legacy v2 model StringMatchNode
 */
export interface V2StringMatchNodeData extends BaseNodeData {
  node_type: 'string_match_node';
  feature: string;
  patterns: string[];
  match_type: 'exact' | 'starts_with' | 'contains' | 'ends_with' | 'regex';
  case_sensitive: boolean;
  match_any: boolean;
}

/**
 * Leaf node - terminal node referencing an output record.
 * leaf_value is an index into TreeOutput.data (-1 = use default).
 *
 * Backend: legacy v2 model LeafNode
 */
export interface V2LeafNodeData extends BaseNodeData {
  node_type: 'leaf';
  leaf_value: number;
}

/** Union of all v2 node data types */
export type V2TreeNodeData =
  | V2RangeNodeData
  | V2NumericalNodeData
  | V2CategoricalNodeData
  | V2StringMatchNodeData
  | V2LeafNodeData;

// ============================================================================
// Tree Structure (v2)
// ============================================================================

/** Node position on canvas */
export interface NodePosition {
  x: number;
  y: number;
}

/**
 * Positioned node combining layout and data.
 * Backend: legacy v2 model PositionedNode
 */
export interface V2PositionedNode {
  id: string;
  position: NodePosition;
  data: V2TreeNodeData;
}

/**
 * Edge data for multi-source edges.
 * Backend: legacy v2 model MultiEdgeData (reused in v2)
 */
export interface MultiEdgeData {
  sourceIndex: number[];
}

/** Edge connecting two nodes */
export interface TreeEdge {
  id: string;
  source: string;
  target: string;
  data: MultiEdgeData;
}

/**
 * Subtree metadata (v2 simplified).
 * Order is determined by list position, not an explicit field.
 *
 * Backend: legacy v2 model SubTree
 */
export interface V2Subtree {
  /** Root node ID */
  id?: string;
  /** Display name */
  name?: string;
}

/** Project metadata */
export interface TreeMetadata {
  name: string;
  description: string;
}

/**
 * Complete v2 tree structure.
 * Backend: legacy v2 model Tree
 */
export interface V2Tree {
  type: 'ui-tree';
  format_version: 2;
  metadata: TreeMetadata | null;
  nodes: V2PositionedNode[];
  edges: TreeEdge[];
  subtrees: V2Subtree[];
  output: TreeOutput;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Validates tree structure is v2 format.
 * Frontend only supports v2. Backend converts v1→v2 on fetch.
 */
export function isV2Tree(tree: any): tree is V2Tree {
  return tree?.type === 'ui-tree' && tree?.format_version === 2;
}

export function isLeafNode(data: V2TreeNodeData): data is V2LeafNodeData {
  return data.node_type === 'leaf';
}

export function isNumericalNode(data: V2TreeNodeData): data is V2NumericalNodeData {
  return data.node_type === 'numerical_test_node';
}

export function isCategoricalNode(data: V2TreeNodeData): data is V2CategoricalNodeData {
  return data.node_type === 'categorical_test_node';
}

export function isRangeNode(data: V2TreeNodeData): data is V2RangeNodeData {
  return data.node_type === 'numerical_range_test_node';
}

export function isStringMatchNode(data: V2TreeNodeData): data is V2StringMatchNodeData {
  return data.node_type === 'string_match_node';
}

export function isStructTypeDef(def: UITypeDef): def is UITypeDefStruct {
  return def.type === 'struct';
}

export function isCategoricalTypeDef(def: UITypeDef): def is UITypeDefCategorical {
  return def.type === 'categorical';
}

// ============================================================================
// Utility Functions
// ============================================================================

/** Extract unique feature names from all v2 nodes */
export function extractFeatureNames(nodes: V2PositionedNode[]): string[] {
  const features = new Set<string>();
  for (const node of nodes) {
    if ('feature' in node.data) {
      features.add((node.data as { feature: string }).feature);
    }
  }
  return Array.from(features).sort();
}

/** Find all variable names referenced across v2 nodes */
export function extractVariableNames(nodes: V2PositionedNode[]): string[] {
  const names = new Set<string>();

  for (const node of nodes) {
    const { data } = node;
    switch (data.node_type) {
      case 'numerical_test_node':
        if (isParameterReference(data.threshold)) {
          names.add(data.threshold.param);
        }
        break;
      case 'categorical_test_node':
        if (isParameterReference(data.category_list)) {
          names.add(data.category_list.param);
        }
        break;
      case 'numerical_range_test_node':
        for (const t of data.thresholds) {
          if (isParameterReference(t)) names.add(t.param);
        }
        break;
      // string_match_node and leaf have no variable references in v2
    }
  }

  return Array.from(names).sort();
}

/** Create an empty TreeOutput with the given schema */
export function createEmptyTreeOutput(dtypes: PolarsSchema = []): TreeOutput {
  return {
    data: [],
    default: null,
    dtypes,
    type_defs: {},
  };
}
