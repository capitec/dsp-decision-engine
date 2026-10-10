import { paramRefName } from '@/utils/param-refs';
import {
  Edge, Node, TreeNode, LeafNodeData,
  UnaryNodeData, CasesIsInNodeData, CasesStringMatchNodeData, CasesRangesNodeData,
  NodeData, ParentIdentifier, TreeOutput, Subtree
} from "./types";

import {
  Connection,
  MarkerType,
} from '@xyflow/react';
import Dagre from '@dagrejs/dagre';


export const OP_SYMBOL_MAP: Record<string, string> = {
  "<=": "≤", 
  "<": "<", 
  "==": "=", 
  ">": ">", 
  ">=": "≥",
}


function sortedIndex(array: number[], value: number, edges: Edge[]) {
	var low = 0,
		high = array.length;

	while (low < high) {
		var mid = low + high >>> 1;
		const sourceIndex = edges[array[mid]].data!.sourceIndex;
		const compareValue = Array.isArray(sourceIndex) ? Math.min(...sourceIndex) : sourceIndex;
		if (compareValue < value) low = mid + 1;
		else high = mid;
	}
	return low;
}

// Commented out for future use - node ordering functionality
// function getNodeOrder(nodes: Node[], edges: Edge[]) {
//   const order = new Map<string, number>();
//   const parentChildMap = new Map<string, number[]>();
//   edges.forEach((e, i) => {
//     if (!parentChildMap.has(e.source)) {
//       parentChildMap.set(e.source, [i]);
//     } else {
//       const arr = parentChildMap.get(e.source)!;
//       const insertIdx = sortedIndex(arr, e.data!.sourceIndex, edges)
//       arr.splice(insertIdx, 0, i);
//     }
//   })
//   let counter = 0;

//   function visit(nodeId: string) {
//     parentChildMap.get(nodeId)?.forEach(childIdx => {
//       visit(edges[childIdx]!.target);
//       order.set(nodeId, counter++);
//     });
//   }

//   // Find all root nodes (nodes with no incoming edges)
//   const rootNodes = nodes.filter(n => !edges.some(e => e.target === n.id));

//   // Visit each root node
//   rootNodes.forEach(root => visit(root.id));

//   return {order, rootNodes};
// }

export const formatTree = async (nodes: Node[], edges: Edge[]) => {
  const g = new Dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB" });
 
  // Sort edges by sourceIndex to respect threshold ordering
  const sortedEdges = [...edges].sort((a, b) => {
    // First sort by source node
    if (a.source !== b.source) {
      return a.source.localeCompare(b.source);
    }
    // Then sort by sourceIndex (threshold index)
    // For arrays, use the first (minimum) value for sorting
    const aSourceIndex = a.data?.sourceIndex ?? [0];
    const bSourceIndex = b.data?.sourceIndex ?? [0];
    const aIndex = Array.isArray(aSourceIndex) ? Math.min(...aSourceIndex) : aSourceIndex;
    const bIndex = Array.isArray(bSourceIndex) ? Math.min(...bSourceIndex) : bSourceIndex;
    return aIndex - bIndex;
  });

  sortedEdges.forEach((edge) => g.setEdge(edge.source, edge.target));
  nodes.forEach((node) =>
    g.setNode(node.id, {
      ...node,
      width: node.measured?.width ?? 0,
      height: node.measured?.height ?? 0,
    }),
  );

  Dagre.layout(g);
  let nodesChanged = false;
  const newNodes = nodes.map((node) => {
      const position = g.node(node.id);
      // We are shifting the dagre node position (anchor=center center) to the top left
      // so it matches the React Flow node anchor point (top left).
      const x = position.x - (node.measured?.width ?? 0) / 2;
      const y = position.y - (node.measured?.height ?? 0) / 2;
      if (x !== node.position.x || y !== node.position.y) {
        nodesChanged = true;
      }
 
      return { ...node, position: { x, y } };
    })
  // Avoid unnecessary rerenders
  // Also prevents auto rearrange loop
  return nodesChanged? newNodes: nodes
};

export const getLabel = (node: TreeNode, features: string[]): string => {
  if (node.type === 'leaf') {
    return node.result_idx.toString();
  }
  if ('feature' in node) {
    const feature = node.feature;
    if (typeof feature === 'string') {
      return feature || "Unselected";
    }
    // ComputedFeature
    return feature.expression || "Unselected";
  }
  return "Unselected";
};

// v3 default node templates

const defaultUnaryNode: UnaryNodeData = {
    type: "unary",
    condition: {
      op: "==",
      feature: '',
      threshold: 0,
    }
  };

const defaultCasesRangesNode: CasesRangesNodeData = {
    type: "cases",
    op: "ranges",
    feature: '',
    conditions: [],
};

const defaultCasesIsInNode: CasesIsInNodeData = {
    type: "cases",
    op: "isin",
    feature: '',
    conditions: [],
};

const defaultCasesStringMatchNode: CasesStringMatchNodeData = {
    type: "cases",
    op: "string_match",
    feature: '',
    conditions: [],
};

export const defaultLeafNode: LeafNodeData = {
    type: "leaf",
    result_idx: -1,
};

const defaultNodeLookup: Record<string, TreeNode> = {
  "unary": defaultUnaryNode,
  "cases:ranges": defaultCasesRangesNode,
  "cases:isin": defaultCasesIsInNode,
  "cases:string_match": defaultCasesStringMatchNode,
  "leaf": defaultLeafNode,
};

export const createsCycle = (newEdge: Connection | Edge, edges: Edge[]) => {
  if (newEdge.source === newEdge.target) return true;
  const hasCycle = (edge: Connection | Edge, visited = new Set()): boolean => {
    if (visited.has(edge.target)) return false;
    visited.add(edge.target);

    const found: Edge | undefined = edges
      .filter(v=>v.source === edge.target)
      .find(v => {
        return (newEdge.source === v.target) || hasCycle(v, visited)
      })
    return !!found;
  };
  const res = hasCycle(newEdge)
  return res;
}

const getUpdatedNode = (node: Node, updates: Partial<NodeData>, features?: string[]): Node => {
  const nodeType = updates.type || node.data.type;
  const baseNode: TreeNode = defaultNodeLookup[nodeType] || defaultLeafNode;

  // Merge with proper type
  const updatedNodeData = {
    ...baseNode,
    ...node.data,
    ...updates,
  } as NodeData;

  return {
    ...node,
    data: updatedNodeData,
    type: "default",
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

const createEdge = (sourceId: string, targetId: string, sourceHandle: string, sourceIndex: number): Edge => ({
  id: `${sourceId}:${targetId}`,
  source: sourceId,
  sourceHandle,
  type: 'labeledEdge',
  target: targetId,
  markerEnd: {
    type: MarkerType.ArrowClosed,
  },
  data: {
    sourceIndex: [sourceIndex] // v3 uses arrays
  }
});

const removeNodeConnections = (
  nodeId: string, 
  parent: ParentIdentifier, 
  edges: Edge[]
): Edge[] => {
  console.log({nodeId, parent, edges})
  const resEdges = edges.filter(edge => 
    (edge.target !== nodeId) // && (edge.source !== parent.parentNodeId || edge.sourceHandle !== parent.parentHandle)
  )
  return resEdges;
};


export const addNode = (
  position: Node["position"],
  nodeId: string = `node_${Date.now()}`,
  data: TreeNode = defaultLeafNode,
  nodes: Node[] = [],
  edges: Edge[] = [],
  parent: ParentIdentifier | undefined = undefined,
): [Node[], Edge[]] => {
  const newNode: Node = {
    id: nodeId,
    data,
    position,
    type: "default",
    // type: data.node_type === "leaf" ? "out" : "processNode"
  };
  console.log({newNode})
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

export const getNextEdge = (edges: Edge[], parentNodeId: string): number => {
  const usedIndices = new Set<number>();
  
  edges
    .filter(edge => edge.source === parentNodeId)
    .forEach(edge => {
      const sourceIndex = edge.data?.sourceIndex;
      if (Array.isArray(sourceIndex)) {
        sourceIndex.forEach(idx => usedIndices.add(idx));
      } else if (typeof sourceIndex === 'number') {
        usedIndices.add(sourceIndex);
      }
    });

  return usedIndices.size === 0
    ? 0
    : Array.from({ length: usedIndices.size + 1 }, (_, i) => i).find(i => !usedIndices.has(i))!;
};

export const linkNodeToParent = (
  nodeId: string, 
  parent: ParentIdentifier, 
  edges: Edge[]
): Edge[] => {
  const newEdge = createEdge(
    parent.parentNodeId, 
    nodeId, 
    parent.parentHandle,
    getNextEdge(edges, parent.parentNodeId),
  );
  if (createsCycle(newEdge, edges)) {return edges};
  const edgesWithoutConnections = removeNodeConnections(nodeId, parent, edges);
  return [...edgesWithoutConnections, newEdge];
};


// Helper to extract feature name from Feature type
const extractFeatureName = (feature: string | { type?: 'computed'; expression: string }): string | null => {
  if (typeof feature === 'string') return feature;
  if (feature && typeof feature === 'object' && 'expression' in feature) {
    return feature.expression;
  }
  return null;
};

const getDefaultFeatures = (nodes: Node[]): string[] => {
  const features = new Set<string>();
  for (const node of nodes) {
    if ('feature' in node.data) {
      const featureName = extractFeatureName(node.data.feature);
      if (featureName) {
        features.add(featureName);
      }
    }
  }
  return Array.from(features).sort();
};


// ===============================
// SUBTREE MANAGEMENT UTILITIES
// ===============================

/**
 * Find all root nodes (nodes with no incoming edges)
 */
export const findRootNodes = (nodes: Node[], edges: Edge[]): Node[] => {
  const nodesWithIncomingEdges = new Set(edges.map(edge => edge.target));
  return nodes.filter(node => !nodesWithIncomingEdges.has(node.id));
};

/**
 * Get feature display name from Feature type (helper for subtree management)
 */
const getFeatureDisplay = (feature: string | { type?: 'computed'; expression: string } | undefined): string => {
  if (typeof feature === 'string') return feature || 'Unselected';
  if (feature && typeof feature === 'object' && 'expression' in feature) {
    return feature.expression || 'Unselected';
  }
  return 'Unselected';
};

/**
 * Generate default name for a node/subtree based on existing naming logic (v3)
 */
export const generateDefaultNodeName = (node: Node): string => {
  const data = node.data;

  if (data.type === 'leaf') {
    return `Leaf ${data.result_idx}`;
  }

  if (data.type === "unary") {
    const feature = getFeatureDisplay(data.condition.feature);
    const op = OP_SYMBOL_MAP[data.condition.op] || data.condition.op;
    const threshold = data.condition.threshold;
    const refName = paramRefName(threshold);
    const thresholdDisplay = refName !== null ? `$${refName}` : String(threshold);
    return `${feature} ${op} ${thresholdDisplay}`;
  } else if (data.type === "cases") {
    const feature = getFeatureDisplay(data.feature);
    const conditions = data.conditions || [];
    if (data.op === "isin") {
      return `${feature} in {${conditions.length} cases}`;
    } else if (data.op === "ranges") {
      return `${feature} [${conditions.length} ranges]`;
    } else if (data.op === "string_match") {
      return `${feature} match(${conditions.length})`;
    }
    return `${feature} cases(${conditions.length})`;
  } else if (data.type === "composite") {
    return `Composite (${data.op})`;
  }

  return 'Unselected';
};

/**
 * Check if a subtree name matches the default name for its node
 */
export const isDefaultSubtreeName = (subtree: Subtree, node: Node): boolean => {
  const defaultName = generateDefaultNodeName(node);
  return subtree.name === defaultName;
};

/**
 * Check if a subtree was automatically generated (has default naming pattern)
 * This is different from isDefaultSubtreeName - this checks if it was ever manually renamed
 */
export const isAutoGeneratedSubtree = (subtree: Subtree, node: Node): boolean => {
  const currentDefaultName = generateDefaultNodeName(node);

  let isAutoGenerated = false;

  // Check common default patterns for this type of node (v3)
  if (node.data.type === 'leaf') {
    isAutoGenerated = /^Leaf -?\d+$/.test(subtree.name);
  } else if (node.data.type === 'unary') {
    const featureDisplay = getFeatureDisplay(node.data.condition.feature);
    isAutoGenerated = subtree.name.startsWith(`${featureDisplay} `) && /[<>=]/.test(subtree.name);
  } else if (node.data.type === 'cases' && node.data.op === 'isin') {
    const featureDisplay = getFeatureDisplay(node.data.feature);
    isAutoGenerated = subtree.name.includes(`${featureDisplay} in {`) || subtree.name === currentDefaultName;
  } else {
    // For other node types, just check exact match
    isAutoGenerated = subtree.name === currentDefaultName;
  }

  return isAutoGenerated;
};

/**
 * Detect all subtrees from current nodes and edges
 * - Show subtrees for root nodes
 * - Hide subtrees for non-root nodes (but preserve them if they have custom names)
 * - Delete subtrees with default names when they become non-root
 */
export const detectSubtrees = (nodes: Node[], edges: Edge[], existingSubtrees: Subtree[] = []): Subtree[] => {
  const rootNodes = findRootNodes(nodes, edges);
  const rootNodeIds = new Set(rootNodes.map(node => node.id));
  const existingSubtreeMap = new Map(existingSubtrees.map(st => [st.rootNodeId, st]));
  const nodeMap = new Map(nodes.map(node => [node.id, node]));
  
  const newSubtrees: Subtree[] = [];
  let maxOrder = Math.max(0, ...existingSubtrees.map(st => st.order));
  
  let hasChanges = false;

  // Process all existing subtrees first
  existingSubtrees.forEach(existing => {
    const node = nodeMap.get(existing.rootNodeId);
    if (!node) {
      // Node was deleted, remove subtree
      hasChanges = true;
      return;
    }

    const isCurrentlyRoot = rootNodeIds.has(existing.rootNodeId);
    const hasCustomName = !isAutoGeneratedSubtree(existing, node);

    if (isCurrentlyRoot) {
      // Node is still a root, keep subtree visible
      if (!hasCustomName) {
        // Auto-update name for subtrees with auto-generated names
        const newDefaultName = generateDefaultNodeName(node);
        if (existing.name !== newDefaultName) {
          hasChanges = true;
        }
        newSubtrees.push({
          ...existing,
          name: newDefaultName,
          hidden: false
        });
      } else {
        // Preserve custom name
        newSubtrees.push({
          ...existing,
          hidden: false
        });
      }
    } else if (hasCustomName) {
      // Node is no longer root but has custom name, hide but preserve
      newSubtrees.push({
        ...existing,
        hidden: true
      });
      if (!existing.hidden) hasChanges = true; // Mark as changed if was visible before
    } else {
      // Node is no longer root and has auto-generated name, delete it
      hasChanges = true;
      // Don't add to newSubtrees (effectively deletes it)
    }
  });

  // Create new subtrees for root nodes that don't have subtrees yet
  rootNodes.forEach(rootNode => {
    const existing = existingSubtreeMap.get(rootNode.id);
    
    if (!existing) {
      hasChanges = true;
      // Create new subtree with default name
      const defaultName = generateDefaultNodeName(rootNode);
      newSubtrees.push({
        id: `${rootNode.id}`,
        name: defaultName,
        order: maxOrder + 1,
        rootNodeId: rootNode.id,
        isActive: false,
        hidden: false
      });
      maxOrder++;
    }
  });
  
  // Sort by order for consistent display
  const sortedSubtrees = newSubtrees.sort((a, b) => a.order - b.order);
  
  // Return original array if no changes to prevent useEffect loops
  return hasChanges ? sortedSubtrees : existingSubtrees;
};

/**
 * Handle root node deletion - promote child nodes to new subtrees
 */
export const handleRootNodeDeletion = (
  deletedNodeId: string,
  nodes: Node[],
  edges: Edge[],
  currentSubtrees: Subtree[]
): Subtree[] => {
  // Find the subtree being deleted
  const deletedSubtree = currentSubtrees.find(st => st.rootNodeId === deletedNodeId);
  if (!deletedSubtree) {
    // Not a root node deletion, just refresh subtrees
    return detectSubtrees(nodes, edges, currentSubtrees);
  }
  
  // Find child nodes of the deleted root
  const childEdges = edges.filter(edge => edge.source === deletedNodeId);
  
  if (childEdges.length === 0) {
    // No children, just remove the subtree
    return currentSubtrees.filter(st => st.rootNodeId !== deletedNodeId);
  }
  
  // Sort child edges by sourceIndex to determine priority
  const sortedChildEdges = childEdges.sort((a, b) => {
    const aSourceIndex = a.data?.sourceIndex ?? [0];
    const bSourceIndex = b.data?.sourceIndex ?? [0];
    const aIndex = Array.isArray(aSourceIndex) ? Math.min(...aSourceIndex) : aSourceIndex;
    const bIndex = Array.isArray(bSourceIndex) ? Math.min(...bSourceIndex) : bSourceIndex;
    return aIndex - bIndex;
  });
  
  const updatedSubtrees = currentSubtrees.filter(st => st.rootNodeId !== deletedNodeId);
  const baseName = deletedSubtree.name;
  const insertIndex = deletedSubtree.order;
  
  // Create new subtrees for each child
  sortedChildEdges.forEach((edge, index) => {
    const childNode = nodes.find(n => n.id === edge.target);
    if (!childNode) return;
    
    const newSubtree: Subtree = {
      id: `${childNode.id}`,
      name: index === 0 ? baseName : `${baseName}.${index}`,
      order: insertIndex + index,
      rootNodeId: childNode.id,
      isActive: false
    };
    
    updatedSubtrees.push(newSubtree);
  });
  
  // Adjust order of subtrees that came after the deleted one
  const adjustedSubtrees = updatedSubtrees.map(st => {
    if (st.order > insertIndex) {
      return { ...st, order: st.order + sortedChildEdges.length - 1 };
    }
    return st;
  });
  
  return adjustedSubtrees.sort((a, b) => a.order - b.order);
};

/**
 * Update subtree order after drag and drop
 */
export const reorderSubtrees = (subtrees: Subtree[], fromIndex: number, toIndex: number): Subtree[] => {
  const reordered = [...subtrees];
  const [movedItem] = reordered.splice(fromIndex, 1);
  reordered.splice(toIndex, 0, movedItem);

  // Update order values to match new positions
  return reordered.map((subtree, index) => ({
    ...subtree,
    order: index
  }));
};

// ===============================
// OUTPUT LABEL FORMATTING
// ===============================

/**
 * Formats a field value for display, handling custom types, arrays, etc.
 */
function formatFieldValue(value: unknown): string {
  if (value === undefined || value === null) {
    return '';
  }

  // Handle struct references: { $key: "display_value" }
  if (typeof value === 'object' && '$key' in value) {
    return String(value.$key);
  }

  // Handle arrays (List, Set, Array types)
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return '[]';
    }
    // Format each element and join
    const formattedElements = value.map(item => {
      if (typeof item === 'object' && '$key' in item) {
        return String(item.$key);
      }
      return String(item);
    });
    return `[${formattedElements.join(', ')}]`;
  }

  // Handle plain objects (shouldn't happen with proper typing, but fallback)
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }

  return String(value);
}

/**
 * Formats an output record using a display format template.
 *
 * @param record - The output record to format
 * @param displayFmt - Template string like "${Action}: ${Description}"
 * @param fieldNames - Ordered list of field names (for fallback to first column)
 * @returns Formatted string
 *
 * @example
 * formatOutputLabel({ action: "Allow", description: "Default" }, "${action}: ${description}", ["action", "description"])
 * // Returns: "Allow: Default"
 */
export function formatOutputLabel(
  record: Record<string, any>,
  displayFmt: string | undefined,
  fieldNames: string[]
): string {
  // If no display format, use first column
  if (!displayFmt || displayFmt.trim() === '') {
    const firstFieldName = fieldNames[0];
    if (firstFieldName && record[firstFieldName] !== undefined) {
      return formatFieldValue(record[firstFieldName]);
    }
    // Fallback to JSON if no fields
    return JSON.stringify(record).substring(0, 50);
  }

  // Replace ${fieldName} with actual values
  return displayFmt.replace(/\$\{(\w+)\}/g, (match, fieldName) => {
    const value = record[fieldName];
    return formatFieldValue(value);
  });
}

/**
 * Extracts field names used in a display format template.
 *
 * @param displayFmt - Template string like "${Action}: ${Description}"
 * @returns Array of field names used in the template
 *
 * @example
 * extractFieldNamesFromFormat("${action}: ${description}")
 * // Returns: ["action", "description"]
 */
export function extractFieldNamesFromFormat(displayFmt: string): string[] {
  const matches = displayFmt.matchAll(/\$\{(\w+)\}/g);
  return Array.from(matches, match => match[1]);
}

/**
 * Validates a display format template against available field names.
 *
 * @param displayFmt - Template string to validate
 * @param availableFields - Available field names
 * @returns Error message if invalid, undefined if valid
 */
export function validateDisplayFormat(
  displayFmt: string,
  availableFields: string[]
): string | undefined {
  if (!displayFmt || displayFmt.trim() === '') {
    return undefined; // Empty is valid (uses default)
  }

  const usedFields = extractFieldNamesFromFormat(displayFmt);
  const invalidFields = usedFields.filter(f => !availableFields.includes(f));

  if (invalidFields.length > 0) {
    return `Unknown fields: ${invalidFields.join(', ')}`;
  }

  return undefined;
}