// Types for flat rule system using document-based JSONB structure

// Shared v2 types - canonical definitions in tree-types.ts
import type {
  TreeOutput,
  UITypeDef,
  UITypeDefStruct,
  UITypeDefCategorical,
  UITypeDefs,
  ParameterInfo,
  ParameterReference,
  PolarsSchema,
  RangeEndLogic,
} from '@/types/tree-types';

// Re-export shared types so existing consumers of './types' don't break
export type {
  TreeOutput,
  UITypeDef,
  UITypeDefStruct,
  UITypeDefCategorical,
  UITypeDefs,
  ParameterInfo,
  ParameterReference,
  RangeEndLogic,
};

export interface FlatRuleProject {
  id: string;
  name: string;
  description?: string;
  created_by: string;
  created_at: string;
  updated_at: string;
  version: number;
}

export interface FlatRuleProjectDetail extends FlatRuleProject {
  project_data: PEDPrioritizedTree;
  validation_errors?: any[];
  live_collab?: boolean;
  // output_types removed - now using project_data.ui_type_defs instead
}

// PED-compatible prioritized_flat_rule structure
export interface PEDPrioritizedTree {
  type: 'prioritized_flat_rule';
  input_schema?: PolarsSchema; // Optional input schema for casting inputs at runtime
  output: TreeOutput; // Nested output structure containing data, dtypes, type_defs, default
  rules: PEDTreeNode[];
  parameters_col: string;
  parameters?: Record<string, ParameterInfo>; // Parameters with type and default value
  mode: string;
  use_optimized_execution?: boolean;
  output_fn?: any;
  post_process_fn?: any;
  format_prioritized_fn?: any;
}

// UI Schema field types
export type UISchemaField =
  | UISchemaFieldStruct
  | UISchemaFieldCategorical
  | UISchemaFieldNumerical
  | UISchemaFieldString
  | UISchemaFieldBoolean;

export interface UISchemaFieldBase {
  label?: string;
  description?: string;
  required?: boolean;
}

export interface UISchemaFieldStruct extends UISchemaFieldBase {
  type: 'struct';
  type_id: string;
}

export interface UISchemaFieldCategorical extends UISchemaFieldBase {
  type: 'categorical';
  type_id: string;
}

export interface UISchemaFieldNumerical extends UISchemaFieldBase {
  type: 'numerical';
  min?: number;
  max?: number;
  allow_decimals?: boolean;
  allow_nans?: boolean;
}

export interface UISchemaFieldString extends UISchemaFieldBase {
  type: 'string';
  min_length?: number;
  max_length?: number;
}

export interface UISchemaFieldBoolean extends UISchemaFieldBase {
  type: 'boolean';
}

// PED Tree node structure (simplified for now)
export interface PEDTreeNode {
  meta?: {
    name?: string;
    description?: string;
    position?: { x: number; y: number };
  };
  rule: any; // Full rule structure - can be expanded later
}

// API request types  
export interface CreateFlatRuleProjectRequest {
  name: string;
  description?: string;
  project_data?: PEDPrioritizedTree;
}

export interface UpdateFlatRuleProjectRequest {
  name?: string;
  description?: string;
  project_data?: PEDPrioritizedTree;
}

// Legacy interfaces - will be replaced with PED-based operations in Phase 3
export interface FlatRule {
  id: string;
  name: string;
  priority: number;
  result_output_id: string;
  project_id: string;
  created_at: string;
  updated_at: string;
}

export interface FlatRuleOutput {
  id: string;
  name: string;
  output_data: Record<string, any>;
  project_id: string;
  created_at: string;
  updated_at: string;
}

export interface FlatRuleParameter {
  id: string;
  name: string;
  parameter_type: 'string' | 'number' | 'boolean' | 'array';
  value: any;
  project_id: string;
  created_at: string;
  updated_at: string;
}

export interface FlatRuleCondition {
  id: string;
  field_path: string;
  operator: 'equals' | 'not_equals' | 'greater_than' | 'less_than' | 'in' | 'not_in' | 'contains' | 'not_contains';
  value: any;
  logical_group_id?: string;
  rule_id: string;
  created_at: string;
  updated_at: string;
}
export interface CreateFlatRuleRequest {
  name: string;
  priority: number;
  result_output_id: string;
}

export interface UpdateFlatRuleRequest {
  name?: string;
  priority?: number;
  result_output_id?: string;
}

export interface CreateFlatRuleOutputRequest {
  name: string;
  output_data: Record<string, any>;
}

export interface CreateFlatRuleParameterRequest {
  name: string;
  parameter_type: 'string' | 'number' | 'boolean' | 'array';
  value: any;
}

export interface CreateFlatRuleConditionRequest {
  field_path: string;
  operator: 'equals' | 'not_equals' | 'greater_than' | 'less_than' | 'in' | 'not_in' | 'contains' | 'not_contains';
  value: any;
  logical_group_id?: string;
}

// Export request and response
export interface FlatRuleExportRequest {
  include_ui_metadata?: boolean;
}

export interface FlatRuleExportResponse {
  project_id: string;
  project_name: string;
  exported_at: string;
  prioritized_tree: Record<string, any>;
}

// Paginated response wrapper
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

// Import output types from the output-types module

// Flat rules UI structure - uses BOTH conditions array (with indices) AND branches array (actual nodes)
export type RuleTableRow =
  | LeafRuleNode
  | UnaryRuleNode
  | CasesNode
  | CompositeRuleNode;

// Leaf node - terminal node returning a result
export interface LeafRuleNode {
  id: string;
  type: 'leaf';
  result_idx: number; // Index into output table (-1 = default)

  // UI state
  editMode?: boolean;
  isCollapsed?: boolean;
}

// Unary condition types - discriminated by operator (uses ParameterReference for flat-rules UI)
export type UnaryCondition =
  | ({ op: '<=' } & { feature: string; threshold: number | ParameterReference })
  | ({ op: '<' } & { feature: string; threshold: number | ParameterReference })
  | ({ op: '==' } & { feature: string; threshold: number | ParameterReference })
  | ({ op: '>' } & { feature: string; threshold: number | ParameterReference })
  | ({ op: '>=' } & { feature: string; threshold: number | ParameterReference })
  | ({ op: '!=' } & { feature: string; threshold: number | ParameterReference })
  | ({ op: 'between' } & { feature: string; min: number | ParameterReference; max: number | ParameterReference })
  | ({ op: 'isin' } & { feature: string; values: number[] | ParameterReference })
  | ({ op: 'string_match' } & {
      feature: string;
      pattern: string;
      match_type: 'exact' | 'starts_with' | 'contains' | 'ends_with' | 'regex';
      case_sensitive?: boolean;
    })
  | ({ op: 'is_true' } & { feature: string })
  | ({ op: 'is_false' } & { feature: string });

// Unary node - binary branching with single condition
export interface UnaryRuleNode {
  id: string;
  type: 'unary';
  condition: UnaryCondition;
  then?: RuleTableRow;
  otherwise?: RuleTableRow;

  // UI state
  editMode?: boolean;
  isCollapsed?: boolean;
}

// Cases node - multi-way branching with different operators
export type CasesNode =
  | CasesRangesNode
  | CasesStringMatchNode
  | CasesIsInNode;

// Cases Ranges - multi-way branching on numeric ranges
export interface CasesRangesNode {
  id: string;
  type: 'cases';
  op: 'ranges';
  feature: string;
  strict?: boolean;
  conditions: Array<{
    when: {
      min: number | ParameterReference | null;
      max: number | ParameterReference | null;
      end_logic?: RangeEndLogic;
    };
    then: number; // Index into branches array
  }>;
  branches: RuleTableRow[]; // Array of actual branch nodes
  otherwise?: number; // Index into branches array

  // UI state
  editMode?: boolean;
  isCollapsed?: boolean;
}

// Cases String Match - multi-way branching on string patterns
export interface CasesStringMatchNode {
  id: string;
  type: 'cases';
  op: 'string_match';
  feature: string;
  conditions: Array<{
    when: {
      patterns: string[]; // Note: plural - array of patterns
    };
    then: number; // Index into branches array
  }>;
  branches: RuleTableRow[]; // Array of actual branch nodes
  otherwise?: number; // Index into branches array

  // UI state
  editMode?: boolean;
  isCollapsed?: boolean;
}

// Cases IsIn - multi-way branching on category membership
export interface CasesIsInNode {
  id: string;
  type: 'cases';
  op: 'isin';
  feature: string;
  conditions: Array<{
    when: {
      values: number[] | ParameterReference;
    };
    then: number; // Index into branches array
  }>;
  branches: RuleTableRow[]; // Array of actual branch nodes
  otherwise?: number; // Index into branches array

  // UI state
  editMode?: boolean;
  isCollapsed?: boolean;
}

// Composite node - AND/OR/NOT logic combining multiple conditions
export interface CompositeRuleNode {
  id: string;
  type: 'composite';
  op: 'and' | 'or' | 'not';
  conditions: any[]; // Nested conditions (can be other nodes) - NOT must have exactly 1
  then?: RuleTableRow;
  otherwise?: RuleTableRow;

  // UI state
  editMode?: boolean;
  isCollapsed?: boolean;
}

// Type guard helpers
export function isCasesNode(node: RuleTableRow): node is CasesNode {
  return node.type === 'cases';
}

export function isCasesRangesNode(node: RuleTableRow): node is CasesRangesNode {
  return node.type === 'cases' && (node as CasesNode).op === 'ranges';
}

export function isCasesIsInNode(node: RuleTableRow): node is CasesIsInNode {
  return node.type === 'cases' && (node as CasesNode).op === 'isin';
}

export function isCasesStringMatchNode(node: RuleTableRow): node is CasesStringMatchNode {
  return node.type === 'cases' && (node as CasesNode).op === 'string_match';
}

// Legacy types - kept for migration compatibility
export interface RuleBranch {
  id: string;
  when?: RuleCondition | RuleCondition[];
  then: RuleTableRow;
}

export type RuleCondition =
  | NumericalCondition
  | CategoricalCondition
  | StringCondition
  | RangeCondition;

export interface NumericalCondition {
  op: '<=' | '<' | '==' | '>' | '>=' | '!=';
  threshold: number | string | ParameterReference;
}

export interface CategoricalCondition {
  categories: number[] | ParameterReference;
}

export interface StringCondition {
  pattern: string | string[] | ParameterReference;
  matchType: 'exact' | 'starts_with' | 'contains' | 'ends_with' | 'regex';
  caseSensitive?: boolean;
}

export interface RangeCondition {
  min?: number | string | null | ParameterReference;
  max?: number | string | null | ParameterReference;
}

// Rule group structure for managing nested logic
export interface RuleGroup {
  id: string;
  parentId?: string;
  indentLevel: number;
  logicalOperator: 'AND' | 'OR';
  conditions: LegacyRuleCondition[];
  childGroups: RuleGroup[];
  output?: RuleOutput;
}

// Legacy individual condition within a group - DEPRECATED
export interface LegacyRuleCondition {
  id: string;
  field: string;
  operator: string;
  value: any;
  valueType?: 'literal' | 'parameter' | 'variable';
}

// Output definition for rules
export interface RuleOutput {
  id: string;
  name: string;
  data: Record<string, any>;
}

// Prioritized tree structure (matches PED format)
export interface PrioritizedTree {
  id: string;
  name: string;
  priority: number;
  rootNode: RuleTableRow;
  description?: string;
}

// Editor state for the flat rule interface
export interface FlatRuleEditorState {
  project: FlatRuleProjectDetail | null;
  selectedTreeId: string | null;
  selectedNodeId: string | null;
  trees: PrioritizedTree[];
  isLoading: boolean;
  error: string | null;
  unsavedChanges: boolean;
}