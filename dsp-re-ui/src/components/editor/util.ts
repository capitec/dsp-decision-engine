import { Edge, Node, SourceData, TreeNode, LeafNode, NumericalNode, CategoricalNode, ForkNode, NodeData, ParentIdentifier } from "./types";

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

const defaultForkNode: ForkNode = {
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

const getUpdatedNode = (node: Node, updates: Partial<NodeData>, features?: string[]): Node => {
  const baseNode: TreeNode = defaultNodeLookup[updates.node_type || node.data.node_type] || defaultLeafNode;
  //@ts-ignore
  const updatedNodeData: NodeData = {
    ...baseNode,
    ...node.data,
    ...updates,
  };
  if (features) {
    updatedNodeData.label = getLabel(updatedNodeData, features)
  }
  return {
    ...node,
    data: updatedNodeData,
    type: updatedNodeData.node_type === "leaf"? "output": "leftRightNode"
  };
};


const getNodeIndex = (nodeId: string, nodes: Node[]): number => 
  nodes.findIndex(node => node.id === nodeId);

export const updateNodeData = (node: Node, newData: Partial<Node["data"]>, features: string[], nodes: Node[]) => {
  const nodeIndex = getNodeIndex(node.id, nodes);
  if (nodeIndex === -1) {
    return nodes;
  }
  const updatedNodes = [...nodes];
  updatedNodes[nodeIndex] = getUpdatedNode(updatedNodes[nodeIndex], newData, features)

  return updatedNodes;
};

const createEdge = (sourceId: string, targetId: string, sourceHandle: 'left' | 'right'): Edge => ({
  id: `${sourceId}:${targetId}`,
  source: sourceId,
  sourceHandle,
  type: 'step',
  target: targetId
});

const removeNodeConnections = (
  nodeId: string, 
  edges: Edge[]
): Edge[] => {
  const resEdges = edges.filter(edge => 
    edge.target !== nodeId
  );
  return resEdges;
};


export const addNode = (
  position: Node["position"],
  nodeId: string = `node_${Date.now()}`,
  data: any = defaultLeafNode,
  features: string[],
  nodes: Node[] = [],
  edges: Edge[] = [],
  parent: ParentIdentifier | undefined = undefined,
): [Node[], Edge[]] => {
  const newNode: Node = {
    id: nodeId,
    data: { 
      ...data,
      label: getLabel(data, features)
    },
    position,
    type: data.node_type === "leaf" ? "output" : "leftRightNode"
  };

  let updatedNodes = [...nodes, newNode];
  let updatedEdges = edges;

  if (parent) {
    updatedEdges = linkNodeToParent(
      nodeId,
      parent,
      updatedEdges
    );
  }

  return [updatedNodes, updatedEdges];
};

export const linkNodeToParent = (
  nodeId: string, 
  parent: ParentIdentifier, 
  edges: Edge[]
): Edge[] => {
  const edgesWithoutConnections = removeNodeConnections(nodeId, edges);

  const newEdge = createEdge(
    parent.parentNodeId, 
    nodeId, 
    parent.parentHandle as 'left' | 'right'
  );

  return [...edgesWithoutConnections, newEdge];
};

export const formatNodes = (data: SourceData): { nodes: Node[], edges: Edge[] } => {
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
      return { x: treeIndex * 300, y: 0 };
    }
    return {
      x: parentPos.x + (isRight ? 100 : -100),
      y: parentPos.y + 100
    };
  };

  let nodes: Node[] = [];
  let edges: Edge[] = [];

  const processNode = (
    nodeId: string, 
    treeIndex: number,
    parentPos?: { x: number, y: number }, 
    isRight = false
  ) => {
    const nodeData = data.nodes[nodeId];
    if (!nodeData) return;

    const position = getPosition(nodeId, treeIndex, parentPos, isRight);
    [nodes, edges] = addNode(position, nodeId, nodeData, data.features, nodes, edges, undefined);

    if ('left_child' in nodeData && nodeData.node_type !== 'leaf') {
      if (nodeData.left_child) {
        edges.push(createEdge(nodeId, nodeData.left_child, 'left'));
        processNode(nodeData.left_child, treeIndex, position, false);
      }
      if (nodeData.right_child) {
        edges.push(createEdge(nodeId, nodeData.right_child, 'right'));
        processNode(nodeData.right_child, treeIndex, position, true);
      }
    }
  };

  rootNodes.forEach((rootId, index) => {
    processNode(rootId, index);
  });

  return { nodes, edges };
};