import {
    Node as xyNode,
    Edge as xyEdge,
} from '@xyflow/react';

type BaseNodeData = {
    data_count?: number;
    sum_hess?: number;
    gain?: number;
};

export interface withChildren {
  left_child: string;
  right_child: string;
}

interface withPosition {
    position: {x: number, y: number};
}

export type ForkNodeData = BaseNodeData & {
    split_feature_id: number;
    default_left: boolean;
};

export type NumericalNodeData = ForkNodeData & {
    node_type: "numerical_test_node";
    comparison_op:  "<=" | "<" | "==" | ">" | ">=";
    threshold: number;
};

export type CategoricalNodeData = ForkNodeData & {
    node_type: "categorical_test_node";
    category_list_right_child: boolean;
    category_list: number[];
};

export type LeafNodeData = Partial<ForkNodeData> & {
    node_type: "leaf";
    leaf_value: number;
};

export type TreeNode = (NumericalNodeData | CategoricalNodeData | LeafNodeData)

export interface TreeOutput {
    data: string[][]
    columns: string[]
}

export interface SourceData {
    features: string[];
    nodes: Record<string, TreeNode & withChildren & Partial<withPosition>>;
    leafOrder?: string[];
    metadata?: ProjectMetadata;
    treeOutput?: TreeOutput;
}

export type NodeData<T = TreeNode> = T & {};

export type Node<T = TreeNode> = xyNode<NodeData<T>>;
export type Edge = xyEdge;

export interface ParentIdentifier {
    parentNodeId: string;
    parentHandle: string;
}

export interface ProjectMetadata {
    name: string;
    description: string;
}