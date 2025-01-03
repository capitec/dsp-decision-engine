import {
    Node as xyNode,
    Edge as xyEdge,
} from '@xyflow/react';

type BaseNode = {
    data_count?: number;
    sum_hess?: number;
    gain?: number;
};

interface withChildren {
  left_child: string;
  right_child: string;
}

export type ForkNode = BaseNode & {
    split_feature_id: number;
    default_left: boolean;
};

export type NumericalNode = ForkNode & {
    node_type: "numerical_test_node";
    comparison_op:  "<=" | "<" | "==" | ">" | ">=";
    threshold: number;
};

export type CategoricalNode = ForkNode & {
    node_type: "categorical_test_node";
    category_list_right_child: boolean;
    category_list: number[];
};

export type LeafNode = Partial<ForkNode> & {
    node_type: "leaf";
    leaf_value: number;
};

export type TreeNode = (NumericalNode | CategoricalNode | LeafNode)

export interface SourceData {
    features: string[];
    nodes: Record<string, TreeNode & withChildren>;
}

export type NodeData = TreeNode & {label: string}
export type Node = xyNode<NodeData>;
export type Edge = xyEdge;

export interface ParentIdentifier {
    parentNodeId: string;
    parentHandle: string;
}