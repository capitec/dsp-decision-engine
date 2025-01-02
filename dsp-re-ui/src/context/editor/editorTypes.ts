// Base structure shared between both types of nodes
type BaseNode = {
  data_count?: number;
  sum_hess?: number;
  gain?: number;
};

type ForkNode = BaseNode & {
  left_child: string;
  right_child: string;
  split_feature_id: number;
  default_left: boolean;
}

type NumericalNode = ForkNode & {
  node_type: "numerical_test_node";
  comparison_op:  "<=" | "<" | "==" | ">" | ">=";
  threshold: number;
};

type CategoricalNode = ForkNode & {
  node_type: "categorical_test_node";
  category_list_right_child: boolean;
  category_list: number[];
};

export type LeafNode = Partial<ForkNode> & {
  node_type: "leaf";
  leaf_value: number;
};

export type TreeNode = NumericalNode | CategoricalNode | LeafNode;


export interface FlattenedNodes {
  features: string[];
  nodes: Record<string, TreeNode>;
}


export interface EditorState extends FlattenedNodes{
    projectId: string | null;
    selectedNodeId: string | null;
    loading: boolean;
    error: string | null;
}
  
export type EditorAction = 
    | { type: 'SET_PROJECT_ID'; payload: string }
    | { type: 'SELECT_NODE'; payload: string | null }
    | { type: 'FETCH_NODES_START' }
    | { type: 'FETCH_NODES_SUCCESS'; payload: FlattenedNodes }
    | { type: 'FETCH_NODES_FAILURE'; payload: string }
    | { type: 'CLEAR_PROJECT' }
    // | { type: 'ADD_NODE'; payload: { parentKey: string; valueIdx: number; node: TreeNode } }
    | { type: 'UPDATE_NODE'; payload: { key: string; node: Partial<TreeNode> } }
    // | { type: 'DELETE_NODE'; payload: string }
;