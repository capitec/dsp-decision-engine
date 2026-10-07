import {
    Node as xyNode,
    Edge as xyEdge,
} from '@xyflow/react';
import { VariableMap } from './variable-types';
import { ProjectOutputField, CustomType } from '@/types/output-field-types';
import {
    ParameterMap,
    PolarsSchema
} from '@/types/tree-types';
import type {
    InputRef,
    RangeEndLogic,
    TStringMatchType,
    TLogicOp,
} from '@/types/tree-v3-types';

// Fix Feature type - make computed type field required
export type Feature = string | { type: 'computed'; expression: string };

// Subtree management types
export interface Subtree {
  id: string;           // Root node ID
  name: string;         // Display name (max 40 chars)
  order: number;        // Priority order (0 = highest)
  rootNodeId: string;   // Reference to root node
  isActive?: boolean;   // Currently selected/focused
  hidden?: boolean;     // Hidden when node becomes non-root but has custom name
}

export interface SubtreeState {
  subtrees: Subtree[];
  activeSubtreeId: string | null;
}

// v3 Node Types - matching the decider V3TreeDocument node types
// Structure managed by edges, no embedded then/otherwise/branches

export type LeafNodeData = {
    type: "leaf";
    result_idx: number; // Index into output.data array (-1 = use default)
};

export type UnaryNodeData = {
    type: "unary";
    id?: string;
    condition: {
        op: "<=" | "<" | "==" | ">" | ">=" | "!=";
        feature: Feature;
        threshold: number | InputRef;
    };
};

export type CasesRangesNodeData = {
    type: "cases";
    op: "ranges";
    id?: string;
    feature: Feature;
    conditions: Array<{
        min: number | null;
        max: number | null;
        end_logic?: RangeEndLogic;
    }>;
};

export type CasesStringMatchNodeData = {
    type: "cases";
    op: "string_match";
    id?: string;
    feature: Feature;
    conditions: Array<{
        pattern: string;
        match_type: TStringMatchType;
        case_sensitive?: boolean;
    }>;
};

export type CasesIsInNodeData = {
    type: "cases";
    op: "isin";
    id?: string;
    feature: Feature;
    conditions: Array<{
        values: number[] | InputRef;
    }>;
};

// Recursive type for composite conditions
export interface CompositeCondition {
    logic_op: TLogicOp;
    conditions: (UnaryCondition | CompositeCondition)[];
}

export interface UnaryCondition {
    op: "<=" | "<" | "==" | ">" | ">=" | "!=";
    feature: Feature;
    threshold: number | InputRef;
}

export type CompositeNodeData = {
    type: "composite";
    id?: string;
    condition: CompositeCondition;
};

export type TreeNode = LeafNodeData | UnaryNodeData | CasesRangesNodeData | CasesStringMatchNodeData | CasesIsInNodeData | CompositeNodeData;

// v3 TreeOutput (same as v2)
export interface TreeOutput {
    dtypes: PolarsSchema;
    type_defs: Record<string, any>;
    data: Record<string, any>[];
    default: Record<string, any> | null;
    display_fmt?: string;
}

// v3 Edge data structure
export interface EdgeData extends Record<string, unknown> {
    sourceIndex: number[]; // Always an array in v3
}

export type NodeData<T = TreeNode> = T & {};

export type Node<T = TreeNode> = xyNode<NodeData<T>>;
export type Edge = xyEdge<EdgeData>;

export interface ParentIdentifier {
    parentNodeId: string;
    parentHandle: string;
    parentNode?: Node;
}

export interface ProjectMetadata {
    name: string;
    description: string;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface ProjectData {
    id: string;
    name: string;
    description: string | null;
    created_by: string;
    created_at: string;
    updated_at: string;
    is_owner: boolean;
    access_mode: string;
}

/**
 * Editor state structure - v3 format only
 */
export interface EditorState {
    nodes: Node[];
    edges: Edge[];
    input_schema: PolarsSchema;
    subtrees?: Subtree[];
    output: TreeOutput;
    parameters: ParameterMap;
}

export interface ProjectRevision {
    id: string;
    project_id: string;
    revision_number: number;
    full_state: EditorState;
    created_at: string;
    created_by: string;
}

export interface ProjectWithRevisionResponse {
    // Project info (always present)
    project_id: string;
    name: string;
    description: string | null;

    // Latest revision info (null if no revisions exist)
    latest_revision: ProjectRevision | null;

    // Editor state (from revision or defaults)
    editor_state: EditorState;
}

export interface SessionInfo {
    recovered_from: string;
    pending_operations: number;
    active_users: string[];
}

export interface ProjectStateResponse {
    latest_revision: ProjectRevision;
    compiled_state: EditorState;
    session_info: SessionInfo;
}