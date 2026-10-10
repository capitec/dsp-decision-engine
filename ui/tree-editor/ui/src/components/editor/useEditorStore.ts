import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { Node, Edge, TreeOutput, Subtree } from './types';
import {
  getProjectApiV1ProjectsProjectIdGet,
  updateProjectApiV1ProjectsProjectIdPut
} from '@/api';
import type { ProjectDetailResponse } from '@/api/types.gen';
import { ValidationResult } from '../validation/ValidationResultsDialog';
import { extractFeatures, renameFeature } from './FeatureActions';
import { detectSubtrees, generateDefaultNodeName } from './util';
import { ParameterMap, TreeEdge, V2Subtree, PolarsSchema } from '@/types/tree-types';
import { normalizeTreeOutput } from '@/utils/normalize-dtypes';
import { ValidationError, parseValidationErrors, formatValidationErrors, getInvalidNodeIds, getSectionErrors } from '@/utils/validation-error-parser';
import { toast } from '@/hooks/use-toast';
import { apiErrorMessage } from '@/lib/utils';
import { subscribeToStoreActivity } from '@/lib/background-token-refresh';

// v2 default output with sample data
const defaultOutput: TreeOutput = {
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

// decikit also accepts edges without an id and a single sourceIndex instead of a list
const normalizeEdge = (edge: any): Edge => ({
  ...edge,
  id: edge.id ?? `${edge.source}:${edge.target}`,
  data: { ...edge.data, sourceIndex: [edge.data?.sourceIndex ?? 0].flat() },
});

// Helper function to extract feature name from v3 Feature type
const getFeatureName = (feature: string | { type?: 'computed'; expression: string } | undefined): string | null => {
  if (typeof feature === 'string') return feature;
  if (feature && typeof feature === 'object' && 'expression' in feature) {
    return feature.expression;
  }
  return null;
};

// Helper function to get node type string from v3 nodes
const getNodeTypeString = (node: Node): string => {
  return node.data.type;
};

// Helper function to infer Polars type from node type (v3)
const inferPolarsTypeFromNode = (nodeType: string, op?: string): 'String' | 'Float64' | 'Int64' => {
  switch (nodeType) {
    case 'unary':
      return 'Float64';
    case 'cases':
      // Need to check op to determine type
      if (op === 'ranges') return 'Float64';
      if (op === 'isin') return 'Int64';
      if (op === 'string_match') return 'String';
      return 'String'; // default
    default:
      return 'String';
  }
};

// Map decikit's validation_errors onto node highlighting and section flags
const applyValidationErrors = (errors: ValidationError[], nodes: Array<{ id: string }>) => {
  const parsed = parseValidationErrors(errors, nodes);
  return {
    state: {
      validationErrors: errors,
      invalidNodeIds: getInvalidNodeIds(parsed),
      ...getSectionErrors(parsed),
    },
    message: formatValidationErrors(parsed),
  };
};

interface EditorStoreState {
  // Core editor data
  input_schema: PolarsSchema; // Input schema with types (snake_case for Python compatibility)
  nodes: Node[];
  edges: Edge[];
  subtrees: Subtree[];

  // v2 format
  output: TreeOutput;
  parameters: ParameterMap;

  // Project management
  loading: boolean;
  saving: boolean;
  error: string | null;
  isDirty: boolean;
  projectId: string | null;
  projectName: string;
  projectDescription: string | null;
  autoRearrange: boolean;
  formatVersion: number;

  // Validation state
  validationErrors: ValidationError[];
  invalidNodeIds: Set<string>;
  hasFeatureErrors: boolean;
  hasOutputErrors: boolean;
  hasParameterErrors: boolean;
  hasSubtreeErrors: boolean;

  // Actions
  setInputSchema: (schema: PolarsSchema | ((prev: PolarsSchema) => PolarsSchema)) => void;
  setNodes: (nodes: Node[] | ((prev: Node[]) => Node[])) => void;
  setEdges: (edges: Edge[] | ((prev: Edge[]) => Edge[])) => void;
  setSubtrees: (subtrees: Subtree[] | ((prev: Subtree[]) => Subtree[])) => void;

  // v2 setters
  setOutput: (output: TreeOutput | ((prev: TreeOutput) => TreeOutput)) => void;
  setParameters: (parameters: ParameterMap | ((prev: ParameterMap) => ParameterMap)) => void;

  setProjectName: (name: string) => void;
  setProjectDescription: (description: string | null) => void;
  setAutoRearrange: (autoRearrange: boolean) => void;
  setIsDirty: (dirty: boolean) => void;

  // Node operations
  updateNodeData: (nodeId: string, updates: Partial<any>) => void;
  addNode: (node: Node) => void;
  removeNode: (nodeId: string) => void;

  // Edge operations
  addEdge: (edge: Edge) => void;
  removeEdge: (edgeId: string) => void;
  updateEdge: (edgeId: string, updates: Partial<Edge>) => void;

  // Feature operations (v2: name-based)
  renameFeature: (oldName: string, newName: string) => void;

  // Project management
  loadProject: (projectId: string) => Promise<void>;
  saveProject: () => Promise<void>;
  validateProject: () => Promise<ValidationResult | null>;

  // Initialization
  initializeFromProject: (response: ProjectDetailResponse) => void;
  reset: () => void;
}

const editorStore = create<EditorStoreState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    input_schema: [], // Input schema with types (snake_case for Python compatibility)
    nodes: [],
    edges: [],
    subtrees: [],

    // v2 format
    output: defaultOutput,
    parameters: {},

    loading: false,
    saving: false,
    error: null,
    isDirty: false,
    projectId: null,
    projectName: '',
    projectDescription: null,
    autoRearrange: false,
    formatVersion: 3,

    // Validation state
    validationErrors: [],
    invalidNodeIds: new Set<string>(),
    hasFeatureErrors: false,
    hasOutputErrors: false,
    hasParameterErrors: false,
    hasSubtreeErrors: false,

    // Basic setters with dirty flag
    setInputSchema: (schema) => set((state) => ({
      input_schema: typeof schema === 'function' ? schema(state.input_schema) : schema,
      isDirty: true
    })),

    setNodes: (nodes) => set((state) => {
      const updatedNodes = typeof nodes === 'function' ? nodes(state.nodes) : nodes;
      const extractedFeatures = extractFeatures(updatedNodes);

      // Auto-sync new features to input_schema with inferred types
      const updatedInputSchema = [...state.input_schema];
      const existingFeatureNames = new Set(updatedInputSchema.map(([name]) => name));

      // Build a map of feature name to node type for type inference
      const featureToNodeType = new Map<string, string>();
      updatedNodes.forEach(node => {
        if ('feature' in node.data) {
          const featureName = getFeatureName(node.data.feature);
          if (featureName) {
            featureToNodeType.set(featureName, getNodeTypeString(node));
          }
        }
      });

      extractedFeatures.forEach(featureName => {
        if (!existingFeatureNames.has(featureName)) {
          const nodeType = featureToNodeType.get(featureName) || 'cases:string_match';
          const inferredType = inferPolarsTypeFromNode(
            nodeType.startsWith('cases:') ? 'cases' : nodeType,
            nodeType.split(':')[1]
          );
          updatedInputSchema.push([featureName, inferredType]);
          existingFeatureNames.add(featureName);
        }
      });

      return {
        nodes: updatedNodes,
        input_schema: updatedInputSchema,
        isDirty: true,
        // Don't clear validation errors here - they persist until save succeeds or user edits the invalid node
      };
    }),

    setEdges: (edges) => set((state) => ({
      edges: typeof edges === 'function' ? edges(state.edges) : edges,
      isDirty: true
    })),

    setSubtrees: (subtrees) => set((state) => ({
      subtrees: typeof subtrees === 'function' ? subtrees(state.subtrees) : subtrees,
      isDirty: true
    })),

    // v2 setters
    setOutput: (output) => set((state) => ({
      output: typeof output === 'function' ? output(state.output) : output,
      isDirty: true
    })),

    setParameters: (parameters) => set((state) => ({
      parameters: typeof parameters === 'function' ? parameters(state.parameters) : parameters,
      isDirty: true
    })),

    setProjectName: (name) => set({ projectName: name }),
    setProjectDescription: (description) => set({ projectDescription: description }),
    setAutoRearrange: (autoRearrange) => set({ autoRearrange }),
    setIsDirty: (dirty) => set({ isDirty: dirty }),

    // Node operations
    updateNodeData: (nodeId, updates) => {
      set((state) => {
        const updatedNodes = state.nodes.map(node => {
          if (node.id !== nodeId) return node;

          const newData = { ...node.data, ...updates };

          // Validation based on node type (v3)
          if (newData.type === 'cases') {
            // Ensure conditions array exists
            if (!Array.isArray(newData.conditions)) {
              newData.conditions = [];
            }
            // Ensure feature is valid
            if (typeof newData.feature !== 'string' && (!newData.feature || typeof newData.feature !== 'object')) {
              newData.feature = '';
            }
          }

          return {
            ...node,
            data: newData
          };
        });

        const extractedFeatures = extractFeatures(updatedNodes);

        // Auto-sync new features to input_schema with inferred types
        const updatedInputSchema = [...state.input_schema];
        const existingFeatureNames = new Set(updatedInputSchema.map(([name]) => name));

        // Build a map of feature name to node type for type inference
        const featureToNodeType = new Map<string, string>();
        updatedNodes.forEach(n => {
          if ('feature' in n.data) {
            const featureName = getFeatureName(n.data.feature);
            if (featureName) {
              featureToNodeType.set(featureName, getNodeTypeString(n));
            }
          }
        });

        extractedFeatures.forEach(featureName => {
          if (!existingFeatureNames.has(featureName)) {
            const nodeType = featureToNodeType.get(featureName) || 'cases:string_match';
            const inferredType = inferPolarsTypeFromNode(
              nodeType.startsWith('cases:') ? 'cases' : nodeType,
              nodeType.split(':')[1]
            );
            updatedInputSchema.push([featureName, inferredType]);
            existingFeatureNames.add(featureName);
          }
        });

        // Clear validation error for this specific node since it's being edited
        const newInvalidNodeIds = new Set(state.invalidNodeIds);
        newInvalidNodeIds.delete(nodeId);

        return {
          nodes: updatedNodes,
          input_schema: updatedInputSchema,
          isDirty: true,
          invalidNodeIds: newInvalidNodeIds
        };
      });
    },

    addNode: (node) => {
      set((state) => {
        const updatedNodes = [...state.nodes, node];
        const extractedFeatures = extractFeatures(updatedNodes);

        // Auto-sync new features to input_schema with inferred types
        const updatedInputSchema = [...state.input_schema];
        const existingFeatureNames = new Set(updatedInputSchema.map(([name]) => name));

        // Build a map of feature name to node type for type inference
        const featureToNodeType = new Map<string, string>();
        updatedNodes.forEach(n => {
          if ('feature' in n.data) {
            const featureName = getFeatureName(n.data.feature);
            if (featureName) {
              featureToNodeType.set(featureName, getNodeTypeString(n));
            }
          }
        });

        extractedFeatures.forEach(featureName => {
          if (!existingFeatureNames.has(featureName)) {
            const nodeType = featureToNodeType.get(featureName) || 'string_match_node';
            const inferredType = inferPolarsTypeFromNode(nodeType);
            updatedInputSchema.push([featureName, inferredType]);
            existingFeatureNames.add(featureName);
          }
        });

        return {
          nodes: updatedNodes,
          input_schema: updatedInputSchema,
          isDirty: true
        };
      });
    },

    removeNode: (nodeId) => {
      set((state) => {
        const updatedNodes = state.nodes.filter(n => n.id !== nodeId);
        const extractedFeatures = extractFeatures(updatedNodes);

        // Auto-sync new features to input_schema with inferred types
        const updatedInputSchema = [...state.input_schema];
        const existingFeatureNames = new Set(updatedInputSchema.map(([name]) => name));

        // Build a map of feature name to node type for type inference
        const featureToNodeType = new Map<string, string>();
        updatedNodes.forEach(n => {
          if ('feature' in n.data) {
            const featureName = getFeatureName(n.data.feature);
            if (featureName) {
              featureToNodeType.set(featureName, getNodeTypeString(n));
            }
          }
        });

        extractedFeatures.forEach(featureName => {
          if (!existingFeatureNames.has(featureName)) {
            const nodeType = featureToNodeType.get(featureName) || 'string_match_node';
            const inferredType = inferPolarsTypeFromNode(nodeType);
            updatedInputSchema.push([featureName, inferredType]);
            existingFeatureNames.add(featureName);
          }
        });

        return {
          nodes: updatedNodes,
          edges: state.edges.filter(e => e.source !== nodeId && e.target !== nodeId),
          input_schema: updatedInputSchema,
          isDirty: true
        };
      });
    },

    // Edge operations
    addEdge: (edge) => {
      set((state) => ({
        edges: [...state.edges, edge],
        isDirty: true
      }));
    },

    removeEdge: (edgeId) => {
      set((state) => ({
        edges: state.edges.filter(e => e.id !== edgeId),
        isDirty: true
      }));
    },

    updateEdge: (edgeId, updates) => {
      set((state) => ({
        edges: state.edges.map(edge =>
          edge.id === edgeId ? { ...edge, ...updates } : edge
        ),
        isDirty: true
      }));
    },

    // Feature operations (v2: name-based)
    renameFeature: (oldName, newName) => {
      set((state) => {
        const updatedNodes = renameFeature(state.nodes, oldName, newName);
        const extractedFeatures = extractFeatures(updatedNodes);

        // Update input_schema - rename the feature but keep the type
        const updatedInputSchema = state.input_schema.map(([name, type]) =>
          name === oldName ? [newName, type] : [name, type]
        ) as PolarsSchema;

        // Add new features if any with inferred types
        const existingFeatureNames = new Set(updatedInputSchema.map(([name]) => name));

        // Build a map of feature name to node type for type inference
        const featureToNodeType = new Map<string, string>();
        updatedNodes.forEach(n => {
          if ('feature' in n.data) {
            const featureName = getFeatureName(n.data.feature);
            if (featureName) {
              featureToNodeType.set(featureName, getNodeTypeString(n));
            }
          }
        });

        extractedFeatures.forEach(featureName => {
          if (!existingFeatureNames.has(featureName)) {
            const nodeType = featureToNodeType.get(featureName) || 'string_match_node';
            const inferredType = inferPolarsTypeFromNode(nodeType);
            updatedInputSchema.push([featureName, inferredType]);
            existingFeatureNames.add(featureName);
          }
        });

        return {
          nodes: updatedNodes,
          input_schema: updatedInputSchema,
          isDirty: true
        };
      });
    },

    // Project management
    loadProject: async (projectId) => {
      try {
        set({ loading: true, error: null, projectId });

        const response = await getProjectApiV1ProjectsProjectIdGet({
          path: { project_id: projectId }
        });

        if (response.data) {
          get().initializeFromProject(response.data as ProjectDetailResponse);
        }

      } catch (err: any) {
        console.error('Failed to load project:', err);
        set({ error: apiErrorMessage(err, 'Failed to load project') });
      } finally {
        set({ loading: false });
      }
    },

    saveProject: async () => {
      const state = get();
      if (!state.projectId || state.saving) return;

      try {
        set({ saving: true, error: null });

        // Convert nodes/edges/subtrees for API
        const v3Nodes = state.nodes.map(node => ({
          id: node.id,
          position: node.position,
          data: node.data
        }));

        const v3Edges: TreeEdge[] = state.edges.map(edge => {
          const sourceIndex = edge.data?.sourceIndex;
          return {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            data: {
              sourceIndex: Array.isArray(sourceIndex) ? sourceIndex : [sourceIndex || 0]
            }
          };
        });

        const v3Subtrees: V2Subtree[] = (state.subtrees || []).map(st => ({
          id: st.id || st.rootNodeId,
          name: st.name
        }));

        // Prepare project data in v3 format
        const projectData = {
          type: "ui-tree", // Type discriminator (stays as "ui-tree" for all versions)
          nodes: v3Nodes,
          edges: v3Edges,
          subtrees: v3Subtrees,

          // V3 uses input_schema instead of features
          input_schema: state.input_schema,

          // V3 format (required)
          output: state.output || defaultOutput,
          parameters: state.parameters || {},

          // Format version discriminator
          formatVersion: 3
        } as any;

        const response = await updateProjectApiV1ProjectsProjectIdPut({
          path: { project_id: state.projectId },
          body: { project_data: projectData }
        });

        // Validation never blocks a save: the server returns what decikit found
        const errors = (response.data?.validation_errors ?? []) as unknown as ValidationError[];
        const formatted = applyValidationErrors(errors, state.nodes);
        set({ isDirty: false, ...formatted.state });

        toast(errors.length === 0
          ? { title: "Project Saved", description: "Your changes have been saved successfully." }
          : {
              title: "Saved with validation issues",
              description: formatted.message,
              duration: 10000,
            });

      } catch (error: any) {
        console.error('Failed to save project:', error);
        toast({
          title: "Save Failed",
          description: apiErrorMessage(error, "Failed to save project. Please try again."),
          variant: "destructive",
        });
      } finally {
        set({ saving: false });
      }
    },

    validateProject: async () => {
      const { projectId, validationErrors, nodes, input_schema, subtrees } = get();
      if (!projectId) return null;
      return {
        valid: validationErrors.length === 0,
        errors: validationErrors.map(e => e.msg),
        warnings: [],
        node_count: nodes.length,
        feature_count: input_schema.length,
        subtree_count: subtrees.length,
        format_version: 3,
        node_results: []
      };
    },

    // Initialization
    initializeFromProject: (response) => {
      const projectData = response.project_data as any; // Type as any since it's a union type
      const normalizedNodes = (projectData.nodes || []).map(normalizeNode);
      const edges = (projectData.edges || []).map(normalizeEdge);
      // A saved subtree is {id: its root node's id, name}, and decikit runs the first one listed, so the
      // editor's subtrees start from them, in their order, rather than being rebuilt in node order.
      const nodeById = new Map<string, Node>(normalizedNodes.map((n: Node) => [n.id, n]));
      const savedSubtrees: Subtree[] = (projectData.subtrees || [])
        .filter((st: any) => nodeById.has(st.rootNodeId ?? st.id))
        .map((st: any, order: number) => {
          const rootNodeId = st.rootNodeId ?? st.id;
          return { ...st, id: st.id ?? rootNodeId, rootNodeId, order: st.order ?? order,
                   name: st.name || generateDefaultNodeName(nodeById.get(rootNodeId)!) };
        });

      // Get input_schema from project data
      let inputSchema: PolarsSchema = projectData.input_schema || [];

      // Auto-sync any features from nodes that aren't in schema with inferred types
      const extractedFeatures = extractFeatures(normalizedNodes);
      const existingFeatureNames = new Set(inputSchema.map(([name]) => name));

      // Build a map of feature name to node type for type inference
      const featureToNodeType = new Map<string, string>();
      normalizedNodes.forEach((n: any) => {
        if ('feature' in n.data && n.data.feature) {
          featureToNodeType.set(n.data.feature, n.data.node_type);
        }
      });

      extractedFeatures.forEach(featureName => {
        if (!existingFeatureNames.has(featureName)) {
          const nodeType = featureToNodeType.get(featureName) || 'string_match_node';
          const inferredType = inferPolarsTypeFromNode(nodeType);
          inputSchema.push([featureName, inferredType]);
          existingFeatureNames.add(featureName);
        }
      });

      // Normalize output dtypes to handle both array and object formats from backend
      const normalizedOutput = projectData.output
        ? normalizeTreeOutput(projectData.output)
        : defaultOutput;

      set({
        projectName: response.name,
        projectDescription: response.description,
        input_schema: inputSchema,
        nodes: normalizedNodes,
        edges,
        ...((projectData.nodes || []).some((n: any) => !n.position) && { autoRearrange: true }),
        subtrees: detectSubtrees(normalizedNodes, edges, savedSubtrees),

        // v2 format - backend handles backward compatibility
        output: normalizedOutput,
        parameters: projectData.parameters || {},

        // Extract format version from loaded project
        formatVersion: projectData.formatVersion || projectData.format_version || 3,

        isDirty: false,
        error: null,
        ...applyValidationErrors(
          (response.validation_errors ?? []) as unknown as ValidationError[],
          projectData.nodes || []
        ).state
      });
    },

    reset: () => {
      set({
        input_schema: [],
        nodes: [],
        edges: [],
        subtrees: [],

        // v2 format
        output: defaultOutput,
        parameters: {},

        loading: false,
        saving: false,
        error: null,
        isDirty: false,
        projectId: null,
        projectName: '',
        projectDescription: null,
        autoRearrange: false,
        formatVersion: 3,
        validationErrors: [],
        invalidNodeIds: new Set<string>(),
        hasFeatureErrors: false,
        hasOutputErrors: false,
        hasParameterErrors: false,
        hasSubtreeErrors: false,
      });
    }
  }))
);

// Enable background token refresh on store activity
subscribeToStoreActivity(editorStore);

export const useEditorStore = editorStore;

// Selector hooks for better performance
export const useEditorInputSchema = () => editorStore((state) => state.input_schema);
export const useEditorNodes = () => editorStore((state) => state.nodes);
export const useEditorEdges = () => editorStore((state) => state.edges);
export const useEditorSubtrees = () => editorStore((state) => state.subtrees);
export const useEditorOutput = () => editorStore((state) => state.output);
export const useEditorParameters = () => editorStore((state) => state.parameters);

// Project management selectors
export const useEditorLoading = () => editorStore((state) => state.loading);
export const useEditorSaving = () => editorStore((state) => state.saving);
export const useEditorError = () => editorStore((state) => state.error);
export const useEditorIsDirty = () => editorStore((state) => state.isDirty);
export const useEditorProjectId = () => editorStore((state) => state.projectId);
export const useEditorProjectName = () => editorStore((state) => state.projectName);
export const useEditorProjectDescription = () => editorStore((state) => state.projectDescription);
export const useEditorAutoRearrange = () => editorStore((state) => state.autoRearrange);
export const useEditorFormatVersion = () => editorStore((state) => state.formatVersion);

// Action selectors
export const useEditorSetInputSchema = () => editorStore((state) => state.setInputSchema);
export const useEditorSetNodes = () => editorStore((state) => state.setNodes);
export const useEditorSetEdges = () => editorStore((state) => state.setEdges);
export const useEditorSetSubtrees = () => editorStore((state) => state.setSubtrees);
export const useEditorSetOutput = () => editorStore((state) => state.setOutput);
export const useEditorSetParameters = () => editorStore((state) => state.setParameters);

// Project management action selectors
export const useEditorSetProjectName = () => editorStore((state) => state.setProjectName);
export const useEditorSetProjectDescription = () => editorStore((state) => state.setProjectDescription);
export const useEditorSetAutoRearrange = () => editorStore((state) => state.setAutoRearrange);
export const useEditorSetIsDirty = () => editorStore((state) => state.setIsDirty);

// Node/Edge operations
export const useEditorUpdateNodeData = () => editorStore((state) => state.updateNodeData);
export const useEditorAddNode = () => editorStore((state) => state.addNode);
export const useEditorRemoveNode = () => editorStore((state) => state.removeNode);
export const useEditorAddEdge = () => editorStore((state) => state.addEdge);
export const useEditorRemoveEdge = () => editorStore((state) => state.removeEdge);
export const useEditorUpdateEdge = () => editorStore((state) => state.updateEdge);
export const useEditorRenameFeature = () => editorStore((state) => state.renameFeature);

// Project management operations
export const useEditorLoadProject = () => editorStore((state) => state.loadProject);
export const useEditorSaveProject = () => editorStore((state) => state.saveProject);
export const useEditorValidateProject = () => editorStore((state) => state.validateProject);
