import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { Node, Edge, TreeOutput, Subtree } from './types';
import {
  getProjectApiV1ProjectsProjectIdGet,
  updateProjectApiV1ProjectsProjectIdPut
} from '@/api';
import type { ProjectDetailResponse } from '@/api/types.gen';
import { ValidationResult } from '../validation/ValidationResultsDialog';
import { withNodeFeatures, renameFeature } from './FeatureActions';
import { ParameterMap, PolarsSchema } from '@/types/tree-types';
import { defaultOutput, fromDocument, toDocument } from './document';
import { ValidationError, parseValidationErrors, formatValidationErrors, getInvalidNodeIds, getSectionErrors } from '@/utils/validation-error-parser';
import { toast } from '@/hooks/use-toast';
import { isStaleSave, staleSaveToast } from '@/lib/stale-save';
import { apiErrorMessage } from '@/lib/utils';
import { subscribeToStoreActivity } from '@/lib/background-token-refresh';

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
  // The server version this editor last loaded or saved; sent with a save so a stale one is refused
  version: number | null;
  autoRearrange: boolean;
  formatVersion: number;
  // The document's keys the editor does not edit (metadata, name, ...), written back unchanged on save
  documentRest: Record<string, any>;

  // Validation state
  validationErrors: ValidationError[];
  invalidNodeIds: Set<string>;
  hasFeatureErrors: boolean;
  hasOutputErrors: boolean;
  hasParameterErrors: boolean;
  hasSubtreeErrors: boolean;

  // Actions
  setInputSchema: (schema: PolarsSchema | ((prev: PolarsSchema) => PolarsSchema)) => void;
  setNodes: (nodes: Node[] | ((prev: Node[]) => Node[]), dirty?: boolean) => void;
  setEdges: (edges: Edge[] | ((prev: Edge[]) => Edge[]), dirty?: boolean) => void;
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
    version: null,
    autoRearrange: false,
    formatVersion: 3,
    documentRest: {},

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

    setNodes: (nodes, dirty = true) => set((state) => {
      const updatedNodes = typeof nodes === 'function' ? nodes(state.nodes) : nodes;
      const updatedInputSchema = withNodeFeatures(state.input_schema, updatedNodes);

      return {
        nodes: updatedNodes,
        input_schema: updatedInputSchema,
        isDirty: dirty || state.isDirty,
        // Don't clear validation errors here - they persist until save succeeds or user edits the invalid node
      };
    }),

    setEdges: (edges, dirty = true) => set((state) => ({
      edges: typeof edges === 'function' ? edges(state.edges) : edges,
      isDirty: dirty || state.isDirty
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

        const updatedInputSchema = withNodeFeatures(state.input_schema, updatedNodes);

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
        const updatedInputSchema = withNodeFeatures(state.input_schema, updatedNodes);

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
        const updatedInputSchema = withNodeFeatures(state.input_schema, updatedNodes);

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
        const renamedSchema = state.input_schema.map(([name, type]) =>
          name === oldName ? [newName, type] : [name, type]
        ) as PolarsSchema;
        const updatedInputSchema = withNodeFeatures(renamedSchema, updatedNodes);

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

        const projectData = toDocument(state);

        const response = await updateProjectApiV1ProjectsProjectIdPut({
          path: { project_id: state.projectId },
          body: { project_data: projectData, base_version: state.version }
        });

        // Validation never blocks a save: the server returns what decikit found
        const errors = (response.data?.validation_errors ?? []) as unknown as ValidationError[];
        const formatted = applyValidationErrors(errors, state.nodes);
        // An edit made while the save was in flight replaced one of these and is not in what was saved
        set((now) => ({
          version: response.data?.version ?? now.version,
          isDirty: (['nodes', 'edges', 'subtrees', 'output', 'parameters', 'input_schema'] as const)
            .some((key) => now[key] !== state[key]),
          ...formatted.state,
        }));

        toast(errors.length === 0
          ? { title: "Project Saved", description: "Your changes have been saved successfully." }
          : {
              title: "Saved with validation issues",
              description: formatted.message,
              duration: 10000,
            });

      } catch (error: any) {
        console.error('Failed to save project:', error);
        if (isStaleSave(error)) staleSaveToast();
        else toast({
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

      set({
        projectName: response.name,
        projectDescription: response.description,
        version: response.version,
        ...fromDocument(projectData),
        ...((projectData.nodes || []).some((n: any) => !n.position) && { autoRearrange: true }),
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
        version: null,
        autoRearrange: false,
        formatVersion: 3,
    documentRest: {},
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
