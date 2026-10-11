import { Node, Edge, TreeOutput, Subtree } from './types';
import { withNodeFeatures } from './FeatureActions';
import { detectSubtrees, generateDefaultNodeName } from './util';
import { TreeEdge, V2Subtree, PolarsSchema, ParameterMap } from '@/types/tree-types';
import { normalizeTreeOutput } from '@/utils/normalize-dtypes';
import { sameJson } from '@/lib/utils';

// v2 default output with sample data
export const defaultOutput: TreeOutput = {
  dtypes: [
    ['action', 'String'],
    ['description', 'String'],
  ],
  type_defs: {},
  data: [
    { action: 'Allow', description: 'Default Action' },
    { action: 'Alert', description: 'A rule suggests that there is an alert' },
    { action: 'Block', description: 'This is more serious and must be blocked' },
  ],
  default: { action: 'Allow', description: 'Default Action' },
};

/** The editor's own copy of a document: what it edits, and the keys it writes back unchanged. */
export interface EditorDocument {
  input_schema: PolarsSchema;
  nodes: Node[];
  edges: Edge[];
  subtrees: Subtree[];
  output: TreeOutput;
  parameters: ParameterMap;
  formatVersion: number;
  documentRest: Record<string, any>;
}

// Helper function to normalize nodes from imports/API
const normalizeNode = (node: any): Node => {
  // Legacy documents stored the ranges strictness as `strict_mode`
  if (node.data && 'strict_mode' in node.data) {
    const { strict_mode, ...data } = node.data;
    node = { ...node, data: { ...data, strict: data.strict ?? strict_mode } };
  }
  return {
    ...node,
    type: "default",
    measured: node.measured || undefined,
    selected: false,
    dragging: false,
    ...node,
    // decikit lets a document leave out positions (they default to 0, 0); the canvas is laid out on load then
    position: node.position ?? { x: 0, y: 0 },
  };
};

const EDITOR_KEYS = new Set(['type', 'nodes', 'edges', 'subtrees', 'input_schema', 'output', 'parameters', 'formatVersion', 'format_version']);

// decikit also accepts edges without an id and a single sourceIndex instead of a list
const normalizeEdge = (edge: any): Edge => ({
  ...edge,
  id: edge.id ?? `${edge.source}:${edge.target}`,
  data: { ...edge.data, sourceIndex: [edge.data?.sourceIndex ?? 0].flat() },
});

const savedNode = (node: Node) => ({ id: node.id, position: node.position, data: node.data });

const savedEdge = (edge: Edge): TreeEdge => {
  const sourceIndex = edge.data?.sourceIndex;
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    data: { sourceIndex: Array.isArray(sourceIndex) ? sourceIndex : [sourceIndex || 0] },
  };
};

/**
 * Read a stored document into the editor. With `previous` (the editor's state), a node or edge the document leaves
 * as it was keeps its object, and a changed one keeps its canvas state (selection, measured size).
 */
export function fromDocument(projectData: any, previous?: Pick<EditorDocument, 'nodes' | 'edges'>): EditorDocument {
  const oldNodes = new Map(previous?.nodes.map((n) => [n.id, n]));
  const nodes: Node[] = (projectData.nodes || []).map((raw: any) => {
    const node = normalizeNode(raw);
    const old = oldNodes.get(node.id);
    if (!old) return node;
    return sameJson(savedNode(old), savedNode(node)) ? old : { ...old, position: node.position, data: node.data };
  });
  const oldEdges = new Map(previous?.edges.map((e) => [e.id, e]));
  const edges: Edge[] = (projectData.edges || []).map((raw: any) => {
    const edge = normalizeEdge(raw);
    const old = oldEdges.get(edge.id);
    if (!old) return edge;
    return sameJson(savedEdge(old), savedEdge(edge)) ? old : { ...old, ...edge };
  });
  // A saved subtree is {id: its root node's id, name}, and decikit runs the first one listed, so the
  // editor's subtrees start from them, in their order, rather than being rebuilt in node order.
  const nodeById = new Map<string, Node>(nodes.map((n) => [n.id, n]));
  const savedSubtrees: Subtree[] = (projectData.subtrees || [])
    .filter((st: any) => nodeById.has(st.rootNodeId ?? st.id))
    .map((st: any, order: number) => {
      const rootNodeId = st.rootNodeId ?? st.id;
      return { ...st, id: st.id ?? rootNodeId, rootNodeId, order: st.order ?? order,
               name: st.name || generateDefaultNodeName(nodeById.get(rootNodeId)!) };
    });

  return {
    input_schema: withNodeFeatures(projectData.input_schema || [], nodes),
    nodes,
    edges,
    subtrees: detectSubtrees(nodes, edges, savedSubtrees),
    // Normalize output dtypes to handle both array and object formats from backend
    output: projectData.output ? normalizeTreeOutput(projectData.output) : defaultOutput,
    parameters: projectData.parameters || {},
    formatVersion: projectData.formatVersion || projectData.format_version || 3,
    documentRest: Object.fromEntries(Object.entries(projectData).filter(([k]) => !EDITOR_KEYS.has(k))),
  };
}

/** The document the editor holds, as Save sends it (no canvas state: selection, measured sizes, edge styles). */
export function toDocument(state: Omit<EditorDocument, 'formatVersion'>) {
  const subtrees: V2Subtree[] = (state.subtrees || []).map((st) => ({ id: st.id || st.rootNodeId, name: st.name }));
  return {
    ...state.documentRest,
    type: "ui-tree", // Type discriminator (stays as "ui-tree" for all versions)
    nodes: state.nodes.map(savedNode),
    edges: state.edges.map(savedEdge),
    subtrees,
    input_schema: state.input_schema,
    output: state.output || defaultOutput,
    parameters: state.parameters || {},
    formatVersion: 3,
  } as any;
}
