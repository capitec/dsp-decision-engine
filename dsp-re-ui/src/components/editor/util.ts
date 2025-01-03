import { Edge, Node, SourceData, TreeNode, LeafNode, NumericalNode, CategoricalNode, ForkNode, NodeData } from "./types";

const getLabel = (node: TreeNode, features: string[]): string => {
  if (node.node_type === 'leaf') {
    return node.leaf_value.toString();
  }
  const feature = features[node.split_feature_id];
  if (node.node_type === "numerical_test_node") {
    return `${feature} ${node.comparison_op} ${node.threshold}`;
  }
  return `${feature} in set`;
};

export const formatNodes = (data: SourceData): { nodes: Node[], edges: Edge[] } => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    
    // Find root nodes (nodes that aren't children of any other node)
    const childNodes = new Set<string>();
    Object.values(data.nodes).forEach(node => {
      if ('left_child' in node) {
        if (node.left_child) childNodes.add(node.left_child);
        if (node.right_child) childNodes.add(node.right_child);
      }
    });
    
    const rootNodes = Object.keys(data.nodes).filter(id => !childNodes.has(id));
  
    const getPosition = (
      nodeId: string, 
      treeIndex: number,
      parentPos?: { x: number, y: number }, 
      isRight = false
    ): { x: number, y: number } => {
      if (!parentPos) {
        // Root nodes are positioned horizontally with equal spacing
        const treeSpacing = 300;
        return { x: treeIndex * treeSpacing, y: 0 };
      }
      return {
        x: parentPos.x + (isRight ? 100 : -100),
        y: parentPos.y + 100
      };
    };
  
    const processNode = (
      nodeId: string, 
      treeIndex: number,
      parentPos?: { x: number, y: number }, 
      isRight = false
    ) => {
      const nodeData = data.nodes[nodeId];
      if (!nodeData) return;
  
      const position = getPosition(nodeId, treeIndex, parentPos, isRight);
      const label = getLabel(nodeData, data.features);
  
      nodes.push({
        id: nodeId,
        data: { ...nodeData, label },
        position,
        type: nodeData.node_type === "leaf"? "output": "leftRightNode"
      });
  
      if ('left_child' in nodeData && nodeData.node_type !== 'leaf') {
        if (nodeData.left_child) {
          edges.push({
            id: `${nodeId}:${nodeData.left_child}`,
            source: nodeId,
            sourceHandle: 'left',
            type: 'step',
            target: nodeData.left_child
          });
          processNode(nodeData.left_child, treeIndex, position, false);
        }
        if (nodeData.right_child) {
          edges.push({
            id: `${nodeId}:${nodeData.right_child}`,
            source: nodeId,
            sourceHandle: 'right',
            type: 'step',
            target: nodeData.right_child
          });
          processNode(nodeData.right_child, treeIndex, position, true);
        }
      }
    };
  
    rootNodes.forEach((rootId, index) => {
      processNode(rootId, index);
    });
  
    return { nodes, edges };
};

const defaultForkNode: ForkNode = {
    left_child: null,
    right_child: null,
    split_feature_id: 0,
    default_left: false,
}

const defaultNumericalNode: NumericalNode = {
    ...defaultForkNode,
    node_type: "numerical_test_node",
    comparison_op: "==",
    threshold: 0,
  };
  
const defaultCategoricalNode: CategoricalNode = {
    ...defaultForkNode,
    node_type: "categorical_test_node",
    category_list_right_child: false,
    category_list: [],
};

export const defaultLeafNode: LeafNode = {
    node_type: "leaf",
    leaf_value: 0,
};

const defaultNodeLookup: Record<string, TreeNode> = {
  "numerical_test_node": defaultNumericalNode,
  "categorical_test_node": defaultCategoricalNode,
  "leaf": defaultLeafNode,
};

export const updateNodeData = (node: Node, newData: Partial<Node["data"]>, features: string[], nodes: Node[]) => {
  const nodeIndex = nodes.findIndex(n => {
    return n.id === node.id
  });
  if (nodeIndex === -1) {
    return nodes;
  }
  const baseNode: TreeNode = defaultNodeLookup[newData.node_type || nodes[nodeIndex].data.node_type] || defaultLeafNode;


  const updatedNodes = [...nodes];

  //@ts-ignore
  const updatedNode: NodeData = {
    ...baseNode,
    ...updatedNodes[nodeIndex].data,
    ...newData,
  }

  updatedNodes[nodeIndex] = {
    ...updatedNodes[nodeIndex],
    data: {...updatedNode, label: getLabel(updatedNode, features)},
    type: updatedNode.node_type === "leaf"? "output": "leftRightNode"
  };
  return updatedNodes;
};

export const addNode = (position: Node["position"], parentNodeId: string, parentHandle: string, features: string[], nodes: Node[]) => {

}