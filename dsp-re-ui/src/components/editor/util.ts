import { Edge, Node, SourceData, TreeNode, LeafNodeData, NumericalNodeData, CategoricalNodeData, ForkNodeData, NodeData, ParentIdentifier, withChildren, TreeOutput, ProjectMetadata, withPosition, BranchingNodeDataWithChildren } from "./types";

import {
  Connection,
  getOutgoers,
  MarkerType,
} from '@xyflow/react';

import ELK, {ElkExtendedEdge, ElkNode} from 'elkjs';

const elk = new ELK();


const elkOptions = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  "elk.spacing.nodeNode": "25", 
  "elk.layered.spacing.edgeNodeBetweenLayers": "25",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.crossingMinimization.semiInteractive": "true",
  "elk.portConstraints": "FIXED_SIDE",
  "elk.layered.wrapping.strategy": "SINGLE_EDGE",
  "elk.ordering.strategy": "MANUAL",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.nodePlacement.favorStraightEdges": "true",
  "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
};

function getNodeOrder(nodes: Node[], edges: Edge[]) {
  const order = new Map<string, number>();
  let counter = 0;

  function visit(nodeId: string, isLeft: boolean) {
    const leftChild = edges.find(e => e.source === nodeId && e.sourceHandle === 'left')?.target;
    const rightChild = edges.find(e => e.source === nodeId && e.sourceHandle === 'right')?.target;
    
    if (leftChild) visit(leftChild, true);
    order.set(nodeId, counter++);
    if (rightChild) visit(rightChild, false);
  }

  // Find all root nodes (nodes with no incoming edges)
  const rootNodes = nodes.filter(n => !edges.some(e => e.target === n.id));

  // Visit each root node
  rootNodes.forEach(root => {
    visit(root.id, true)
  });

  return {order, rootNodes};
}

export const formatTree = async (nodes: Node[], edges: Edge[]) => {
  // Pre-process to determine node ordering
  const {order:nodeOrder, rootNodes} = getNodeOrder(nodes, edges);

  const graph = {
    id: 'root',
    layoutOptions: elkOptions,
    children: 
    [
      ...nodes.map(node => ({
        ...node,
        layoutOptions: {
          "elk.position": nodeOrder.has(node.id) ? 
            `(${nodeOrder.get(node.id)! * 100},0)` : undefined
        },
        width: node.width || node.initialWidth || 172,
        height: node.height || node.initialHeight || 36,
      } as ElkNode)),
      {
        id: "__*root*__",
        layoutOptions: {
          "elk.position": "(0,0)"
        },
        width: 1,
        height: 1,
      }
    ],
    edges: [
      ...edges.map(edge => ({
        id: edge.id,
        sources: [`${edge.source}`],
        targets: [`${edge.target}`]
      })),
      ...rootNodes.map(n => ({
        id: `__*root*__${n.id}`,
        sources: [`__*root*__`],
        targets: [`${n.id}`]
      })),

    ]
  };

  const layout = await elk.layout(graph);
  return nodes.map(node => {
    const elkNode = layout.children?.find(n => n.id === node.id);
    return {
      ...node,
      position: {
        x: (elkNode?.x || 0) - (elkNode?.width || 0)/2,
        y: elkNode?.y || 0 - (elkNode?.height || 0)/2
      }
    };
  });
};

export const getLabel = (node: TreeNode, features: string[]): string => {
  if (node.node_type === 'leaf') {
    return node.leaf_value.toString();
  }
  const feature = features[node.split_feature_id];
  if (node.node_type === "numerical_test_node") {
    return `${feature} ${node.comparison_op} ${node.threshold}`;
  }
  return `${feature} in set`;
};

const defaultForkNode: ForkNodeData = {
    split_feature_id: 0,
    default_left: false,
}

const defaultNumericalNode: NumericalNodeData = {
    ...defaultForkNode,
    node_type: "numerical_test_node",
    comparison_op: "==",
    threshold: 0,
  };
  
const defaultCategoricalNode: CategoricalNodeData = {
    ...defaultForkNode,
    node_type: "categorical_test_node",
    category_list_right_child: false,
    category_list: [],
};

export const defaultLeafNode: LeafNodeData = {
    node_type: "leaf",
    leaf_value: -1,
};

const defaultNodeLookup: Record<string, TreeNode> = {
  "numerical_test_node": defaultNumericalNode,
  "categorical_test_node": defaultCategoricalNode,
  "leaf": defaultLeafNode,
};

const createsCycle = (newEdge: Connection | Edge, edges: Edge[]) => {
  const hasCycle = (edge: Connection | Edge, visited = new Set()): boolean => {
    if (visited.has(edge.target)) return false;
    visited.add(edge.target);

    const found: Edge | undefined = edges
      .filter(v=>v.source == edge.target)
      .find(v => {
        return (newEdge.source == v.target) || hasCycle(v, visited)
      })
    return !!found;
  };
  const res = hasCycle(newEdge)
  return res;
}

const getUpdatedNode = (node: Node, updates: Partial<NodeData>, features?: string[]): Node => {
  const baseNode: TreeNode = defaultNodeLookup[updates.node_type || node.data.node_type] || defaultLeafNode;
  //@ts-ignore
  const updatedNodeData: NodeData = {
    ...baseNode,
    ...node.data,
    ...updates,
  };

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
  target: targetId,
  markerEnd: {
    type: MarkerType.ArrowClosed,
  },
});

const removeNodeConnections = (
  nodeId: string, 
  parent: ParentIdentifier, 
  edges: Edge[]
): Edge[] => {
  console.log({nodeId, parent,edges})
  const resEdges = edges.filter(edge => 
    (edge.target !== nodeId) && (edge.source !== parent.parentNodeId || edge.sourceHandle !== parent.parentHandle)
  )
  return resEdges;
};


export const addNode = (
  position: Node["position"],
  nodeId: string = `node_${Date.now()}`,
  data: any = defaultLeafNode,
  nodes: Node[] = [],
  edges: Edge[] = [],
  parent: ParentIdentifier | undefined = undefined,
): [Node[], Edge[]] => {
  const newNode: Node = {
    id: nodeId,
    data,
    position,
    type: data.node_type === "leaf" ? "output" : "leftRightNode"
  };

  let updatedNodes = [...nodes, newNode];
  let updatedEdges = edges;

  if (parent) {
    updatedEdges = linkNodeToParent(
      nodeId,
      parent,
      removeNodeConnections(nodeId, parent, updatedEdges)
    );
  }

  return [updatedNodes, updatedEdges];
};

export const linkNodeToParent = (
  nodeId: string, 
  parent: ParentIdentifier, 
  edges: Edge[]
): Edge[] => {
  const newEdge = createEdge(
    parent.parentNodeId, 
    nodeId, 
    parent.parentHandle as 'left' | 'right'
  );
  if (createsCycle(newEdge, edges)) {return edges};
  const edgesWithoutConnections = removeNodeConnections(nodeId, parent, edges);
  return [...edgesWithoutConnections, newEdge];
};


const getDefaultFeatures = (nodes: SourceData["nodes"]): string[] => {
  const maxFeatureId = Math.max(
    ...Object.values(nodes)
      .filter((node): node is (NumericalNodeData | CategoricalNodeData) & withChildren => 
        'split_feature_id' in node
      )
      .map(node => node.split_feature_id),
    -1
  );
  
  return Array.from({ length: maxFeatureId + 1 }, (_, i) => `feature_${i}`);
};

const compactLeafValues = (nodes: Record<string, TreeNode>): Map<number, number> => {
  const uniqueValues = Array.from(new Set<number>(
    Object.values(nodes)
      .filter((node): node is LeafNodeData => node.node_type === 'leaf')
      .filter((node): node is LeafNodeData => node.leaf_value != -1)
      .map(node => node.leaf_value)
  )).sort((a, b) => a - b);
  
  const uniqueMap = new Map(uniqueValues.map((value, index) => [value, index]));
  uniqueMap.set(-1, -1);
  return uniqueMap;
};

const validateNodes = (
  nodes: SourceData["nodes"], 
  features: SourceData["features"]
): void => {
  // TODO
  // for (const [nodeId, node] of Object.entries(nodes)) {
  //   if ('split_feature_id' in node && node.split_feature_id >= features.length) {
  //     throw new Error(
  //       `Node ${nodeId} references feature index ${node.split_feature_id} but only ${features.length} features exist`
  //     );
  //   }
  // }
};



export const loadState = async (data: SourceData, filename?: string): Promise<{ 
  nodes: Node[],
  edges: Edge[], 
  features: string[],
  leafOrder: string[],
  metadata: ProjectMetadata,
  treeOutput: TreeOutput
}> => {
  const edges: Edge[] = [];
  const valueMapping = compactLeafValues(data.nodes);
  let seenPosition = false;

  const nodes: Node[] = Object.keys(data.nodes).map(nodeId=>{
    const {position, ...nodeData} = data.nodes[nodeId];
    if (position) { seenPosition = true }
    const [[newNode]] = addNode(
      position || {x:0,y:0}, 
      nodeId, 
      nodeData.node_type === "leaf"? {...nodeData, leaf_value:valueMapping.get(nodeData.leaf_value)||nodeData.leaf_value}:nodeData, 
      [], [], 
      undefined)
    if (nodeData.node_type === 'leaf') { return newNode };
    if (nodeData.left_child) {
      edges.push(createEdge(nodeId, nodeData.left_child, 'left'));
    }
    if (nodeData.right_child) {
      edges.push(createEdge(nodeId, nodeData.right_child, 'right'));
    }
    return newNode;
  });

  const features = data.features ?? getDefaultFeatures(data.nodes);
  return {
    nodes: seenPosition?nodes: await formatTree(nodes,edges), 
    edges, 
    features,
    leafOrder: data.leafOrder ?? nodes.filter(n => n.data.node_type === 'leaf').map(n => n.id),
    metadata: data.metadata ?? { name: (filename||"Untitled"), description: "" },
    treeOutput: data.treeOutput ?? { data: Object.keys(valueMapping).map(v=>[]), columns:[]}
  }
}

export const exportState = (
  nodes: Node[],
  edges: Edge[],
  features: string[],
  leafOrder: string[],
  metadata: ProjectMetadata,
  treeOutput: TreeOutput
): SourceData => {
  const edgeMap = new Map<string, {left?: string, right?: string}>();
  
  for (const edge of edges) {
    const entry = edgeMap.get(edge.source) ?? {};
    if (edge.sourceHandle?.includes('left')) {
      entry.left = edge.target;
    } else {
      entry.right = edge.target;
    }
    edgeMap.set(edge.source, entry);
  }

  const exportNodes: SourceData["nodes"] = {};
  
  nodes.forEach((node)=>{
    if (node.data.node_type === 'leaf') {
      exportNodes[node.id] = ({
        ...(node.data as LeafNodeData),
        position: node.position
      } as LeafNodeData & withPosition)
      return
    }
    const children = edgeMap.get(node.id) ;
    if (!children || !children.left || !children.right) {
      throw new Error(`Non-leaf node ${node.id} missing children`);
    }
    exportNodes[node.id] = {
      ...node.data,
      position: node.position,
      left_child: children.left,
      right_child: children.right
    };
  })

  validateNodes(exportNodes, features);

  return {
    features,
    nodes: exportNodes,
    leafOrder,
    metadata,
    treeOutput
  };
};