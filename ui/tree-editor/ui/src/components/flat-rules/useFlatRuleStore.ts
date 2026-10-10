import React from 'react';
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { PrioritizedTree, RuleTableRow, FlatRuleProjectDetail, RuleCondition } from './types';
import { flatRuleApi } from './api';
import { normalizeDtypes } from '@/utils/normalize-dtypes';
import { parseFlatRuleValidationErrors, groupErrorsByRule, getSectionErrors, formatRuleErrors, type ParsedFlatRuleError } from '@/utils/flat-rule-validation-parser';
import { subscribeToStoreActivity } from '@/lib/background-token-refresh';
import { toast } from '@/hooks/use-toast';
import { apiErrorMessage } from '@/lib/utils';
import { collectParamRefs } from '@/utils/param-refs';

interface FlatRuleState {
  // Core data
  trees: PrioritizedTree[];
  project: FlatRuleProjectDetail;
  
  // UI state
  expandedTrees: Set<string>;
  editingTreeName: string | null;
  copiedNode: RuleTableRow | null;
  collapsedNodes: Set<string>;
  focusedNode: {treeId: string, nodeId: string, depth: number} | null;
  
  // Available data (computed)
  availableFields: string[];
  outputOptions: Array<{ id: number; name: string; description: string }>;
  
  
  // Save state
  saving: boolean;
  // What was last loaded or saved: every edit replaces `trees` or `project.project_data`, so a different
  // reference means unsaved changes
  savedTrees: PrioritizedTree[];
  savedProjectData: any;
  error: string | null;
  ruleErrors: Map<string, string>; // Map of treeId to error message

  // Validation errors (detailed)
  validationErrors: Map<number, any[]>; // Map of rule index to parsed errors
  sectionErrors: any[]; // Errors not tied to a specific rule (input_schema, output, parameters)
  hasInputSchemaErrors: boolean;
  hasOutputErrors: boolean;
  hasParameterErrors: boolean;
  
  // Actions
  setTrees: (trees: PrioritizedTree[] | ((prev: PrioritizedTree[]) => PrioritizedTree[])) => void;
  setProject: (project: FlatRuleProjectDetail | ((prev: FlatRuleProjectDetail) => FlatRuleProjectDetail)) => void;
  setExpandedTrees: (expanded: Set<string> | ((prev: Set<string>) => Set<string>)) => void;
  setEditingTreeName: (name: string | null) => void;
  setCopiedNode: (node: RuleTableRow | null) => void;
  setCollapsedNodes: (collapsed: Set<string> | ((prev: Set<string>) => Set<string>)) => void;
  setFocusedNode: (node: {treeId: string, nodeId: string, depth: number} | null) => void;
  
  
  // Node operations
  updateNodeValue: (treeId: string, nodeId: string, field: string, value: any) => void;
  updateConditionInGroup: (treeId: string, branchId: string, conditionIndex: number, updates: Partial<RuleCondition>) => void;
  saveAsNewOutput: (formData: Record<string, any>) => void;
  saveToExistingOutput: (outputIndex: number, formData: Record<string, any>) => void;
  
  // Parameter management
  createParameter: (name: string, type: any, value: any) => void;
  updateParameter: (name: string, updates: { name?: string; type?: any; value?: any }) => void;
  deleteParameter: (name: string) => void;
  isParameterInUse: (parameterName: string) => boolean;
  
  // Save functionality; resolves to whether the server stored the document
  saveProject: () => Promise<boolean>;
  
  
  // Initialization
  initializeFromProject: (project: FlatRuleProjectDetail) => void;
  
  // Internal helpers
  updateProjectData: (updates: any) => void;
}

function containsNode(node: any, id: string): boolean {
  if (!node || typeof node !== 'object') return false;
  if (node.id === id) return true;
  return [node.then, node.otherwise, ...(node.branches ?? [])].some(child => containsNode(child, id));
}

// decikit reports a document that is valid but cannot be built with no location: the message names the node
// ("tree node '<id>': ...") or the output, so place it on that rule or on the output section.
function locateBuildError(error: ParsedFlatRuleError, trees: PrioritizedTree[]): ParsedFlatRuleError {
  if (error.type !== 'build_error') return error;
  const nodeId = /^tree node '([^']+)': /.exec(error.message)?.[1];
  const ruleIndex = nodeId ? trees.findIndex(tree => containsNode(tree.rootNode, nodeId)) : -1;
  if (ruleIndex >= 0) {
    return { ...error, ruleIndex, section: 'rule', conditionPath: [], message: error.message.replace(/^tree node '[^']+': /, '') };
  }
  return /\boutput\b/.test(error.message) ? { ...error, section: 'output' } : error;
}

// Map decikit's validation_errors onto per-rule highlighting and section flags
function buildErrorState(rawErrors: any[] | undefined, trees: PrioritizedTree[]) {
  const ruleErrors = new Map<string, string>();
  const validationErrors = new Map<number, any[]>();
  const parsed = parseFlatRuleValidationErrors(rawErrors ?? []).map(e => locateBuildError(e, trees));
  groupErrorsByRule(parsed).forEach((errors, ruleIndex) => {
    if (ruleIndex < trees.length) {
      ruleErrors.set(trees[ruleIndex].id, formatRuleErrors(errors));
      validationErrors.set(ruleIndex, errors);
    }
  });
  return {
    ruleErrors,
    validationErrors,
    sectionErrors: parsed.filter(e => e.ruleIndex === undefined),
    ...getSectionErrors(parsed),
  };
}

// Helper to create default leaf node
function createDefaultLeafNode(): RuleTableRow {
  return {
    id: crypto.randomUUID(),
    type: 'leaf',
    result_idx: -1,
    editMode: false
  };
}

// Helper to compute available fields from input schema
const computeAvailableFields = (projectData: any): string[] => {
  const inputSchema = projectData?.input_schema || [];
  if (Array.isArray(inputSchema) && inputSchema.length > 0) {
    return inputSchema.map(([fieldName]: [string, any]) => fieldName);
  }
  return [];
};

// Helper to compute output options (now accessing nested output structure)
const computeOutputOptions = (projectData: any): Array<{ id: number; name: string; description: string }> => {
  const outputData = projectData?.output || {};
  // Normalize dtypes to handle both array and object formats from backend
  const dtypes = normalizeDtypes(outputData.dtypes || []);
  const output = outputData.data || [];
  const defaultValue = outputData.default || {};

  // Generate default description from actual default values
  const defaultDescription = Object.keys(defaultValue).length > 0
    ? Object.entries(defaultValue).map(([key, value]) => {
        if (value === null || value === undefined) return `${key}: null`;
        if (typeof value === 'string') return `${key}: ${value}`;
        if (typeof value === 'boolean') return `${key}: ${value}`;
        if (typeof value === 'number') return `${key}: ${value}`;
        if (Array.isArray(value)) return `${key}: [${value.join(', ')}]`;
        if (typeof value === 'object') return `${key}: {...}`;
        return `${key}: ${String(value)}`;
      }).join(', ')
    : 'no default values';

  const options = [
    { id: -1, name: 'Default', description: defaultDescription }
  ];

  output.forEach((record: any, index: number) => {
    const fieldNames = dtypes.map(([name]: [string, any]) => name);

    // Create a clean key: value display format
    const keyValuePairs = Object.entries(record).map(([key, value]) => {
      if (value === null || value === undefined) return `${key}: null`;
      if (typeof value === 'string') return `${key}: ${value}`;
      if (typeof value === 'boolean') return `${key}: ${value}`;
      if (typeof value === 'number') return `${key}: ${value}`;
      if (Array.isArray(value)) return `${key}: [${value.join(', ')}]`;
      if (typeof value === 'object') return `${key}: {...}`;
      return `${key}: ${String(value)}`;
    });

    const description = keyValuePairs.join(', ');

    // Try to find a meaningful display value for the name
    const displayValue = fieldNames.length > 0
      ? (record[fieldNames[0]] || `Record ${index + 1}`)
      : `Record ${index + 1}`;

    options.push({
      id: index,
      name: String(displayValue),
      description: description
    });
  });

  return options;
};

const store = create<FlatRuleState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    trees: [],
    project: {} as FlatRuleProjectDetail,
    expandedTrees: new Set<string>(),
    editingTreeName: null,
    copiedNode: null,
    collapsedNodes: new Set<string>(),
    focusedNode: null,
    availableFields: [],
    outputOptions: [],
    saving: false,
    savedTrees: [],
    savedProjectData: undefined,
    error: null,
    ruleErrors: new Map(),
    validationErrors: new Map(),
    sectionErrors: [],
    hasInputSchemaErrors: false,
    hasOutputErrors: false,
    hasParameterErrors: false,

    // Actions
    setTrees: (trees) => set((state) => ({
      trees: typeof trees === 'function' ? trees(state.trees) : trees
    })),

    setProject: (project) => set((state) => {
      const newProject = typeof project === 'function' ? project(state.project) : project;
      return {
        project: newProject,
        availableFields: computeAvailableFields(newProject.project_data),
        outputOptions: computeOutputOptions(newProject.project_data)
      };
    }),

    setExpandedTrees: (expanded) => set((state) => ({
      expandedTrees: typeof expanded === 'function' ? expanded(state.expandedTrees) : expanded
    })),

    setEditingTreeName: (name) => set({ editingTreeName: name }),

    setCopiedNode: (node) => set({ copiedNode: node }),

    setCollapsedNodes: (collapsed) => set((state) => ({
      collapsedNodes: typeof collapsed === 'function' ? collapsed(state.collapsedNodes) : collapsed
    })),

    setFocusedNode: (node) => set({ focusedNode: node }),

    // Node operations
    updateNodeValue: (treeId, nodeId, field, value) => {
      set((state) => ({
        trees: state.trees.map(tree => {
          if (tree.id !== treeId) return tree;
          
          const updateNodeRecursively = (node: RuleTableRow | undefined): RuleTableRow | undefined => {
            if (!node) return undefined;

            if (node.id === nodeId) {
              return { ...node, [field]: value };
            }

            // Check what needs to be recursed into
            const hasBranches = (node.type === 'cases') && node.branches && node.branches.length > 0;
            const hasThen = (node.type === 'unary' || node.type === 'composite') && node.then !== undefined;
            const hasOtherwise = (node.type === 'unary' || node.type === 'composite' || node.type === 'cases') && node.otherwise !== undefined;

            // If no children, this node doesn't contain our target
            if (!hasBranches && !hasThen && !hasOtherwise) {
              return node;
            }

            // Build updated node with recursively updated children
            if (hasBranches && (node.type === 'cases')) {
              return {
                ...node,
                branches: node.branches.map(branch => updateNodeRecursively(branch) || branch),
              } as RuleTableRow;
            }

            if ((hasThen || hasOtherwise) && (node.type === 'unary' || node.type === 'composite')) {
              return {
                ...node,
                then: node.then ? (updateNodeRecursively(node.then) || node.then) : node.then,
                otherwise: node.otherwise ? (updateNodeRecursively(node.otherwise) || node.otherwise) : node.otherwise
              } as RuleTableRow;
            }

            return node;
          };
          
          return {
            ...tree,
            rootNode: updateNodeRecursively(tree.rootNode) || tree.rootNode
          };
        })
      }));
    },

    updateConditionInGroup: (treeId, branchId, conditionIndex, updates) => {
      // Direct implementation - update condition in specified group
      set((state) => ({
        trees: state.trees.map(tree => {
          if (tree.id !== treeId) return tree;
          
          const updateNodeRecursively = (node: RuleTableRow): RuleTableRow => {
            // Handle cases nodes with branches
            if ((node.type === 'cases') && node.branches) {
              return {
                ...node,
                branches: node.branches.map(branch => updateNodeRecursively(branch)),
              } as RuleTableRow;
            }

            // Handle unary/composite nodes
            if ((node.type === 'unary' || node.type === 'composite')) {
              return {
                ...node,
                then: node.then ? updateNodeRecursively(node.then) : node.then,
                otherwise: node.otherwise ? updateNodeRecursively(node.otherwise) : node.otherwise
              } as RuleTableRow;
            }

            return node;
          };
          
          return {
            ...tree,
            rootNode: updateNodeRecursively(tree.rootNode) || tree.rootNode
          };
        })
      }));
    },

    saveAsNewOutput: (formData) => {
      set((state) => ({
        project: {
          ...state.project,
          project_data: {
            ...state.project.project_data,
            output: {
              ...(state.project.project_data?.output || {}),
              data: [...(state.project.project_data?.output?.data || []), formData]
            }
          }
        }
      }));
      // Recompute output options
      const newProject = get().project;
      set({ outputOptions: computeOutputOptions(newProject.project_data) });
    },

    saveToExistingOutput: (outputIndex, formData) => {
      if (outputIndex === -1) {
        // Update default
        set((state) => ({
          project: {
            ...state.project,
            project_data: {
              ...state.project.project_data,
              output: {
                ...(state.project.project_data?.output || {}),
                default: formData
              }
            }
          }
        }));
      } else {
        // Update existing output record
        set((state) => {
          const currentOutput = state.project.project_data?.output?.data || [];
          const updatedOutput = [...currentOutput];
          updatedOutput[outputIndex] = formData;

          return {
            project: {
              ...state.project,
              project_data: {
                ...state.project.project_data,
                output: {
                  ...(state.project.project_data?.output || {}),
                  data: updatedOutput
                }
              }
            }
          };
        });
      }
      // Recompute output options
      const newProject = get().project;
      set({ outputOptions: computeOutputOptions(newProject.project_data) });
    },

    // Parameter management actions
    createParameter: (name, type, value) => {
      set((state) => ({
        project: {
          ...state.project,
          project_data: {
            ...state.project.project_data,
            parameters: {
              ...state.project.project_data?.parameters,
              [name]: {
                type,
                default_value: value
              }
            }
          }
        }
      }));
    },

    updateParameter: (name, updates) => {
      set((state) => {
        const currentParams = state.project.project_data?.parameters || {};
        const updatedParams = { ...currentParams };

        if (updates.name && updates.name !== name) {
          // Rename parameter - delete old key and create new one
          const paramInfo = updatedParams[name];
          if (paramInfo) {
            updatedParams[updates.name] = {
              ...paramInfo,
              ...(updates.type !== undefined && { type: updates.type }),
              ...(updates.value !== undefined && { default_value: updates.value })
            };
            delete updatedParams[name];
          }
        } else {
          // Update existing parameter
          const paramInfo = updatedParams[name];
          if (paramInfo) {
            updatedParams[name] = {
              ...paramInfo,
              ...(updates.type !== undefined && { type: updates.type }),
              ...(updates.value !== undefined && { default_value: updates.value })
            };
          }
        }

        return {
          project: {
            ...state.project,
            project_data: {
              ...state.project.project_data,
              parameters: updatedParams
            }
          }
        };
      });
    },

    deleteParameter: (name) => {
      set((state) => {
        const currentParams = state.project.project_data?.parameters || {};
        const updatedParams = { ...currentParams };
        delete updatedParams[name];

        return {
          project: {
            ...state.project,
            project_data: {
              ...state.project.project_data,
              parameters: updatedParams
            }
          }
        };
      });
    },

    isParameterInUse: (parameterName) =>
      get().trees.some(tree => collectParamRefs(tree.rootNode).has(parameterName)),

    // Save functionality
    saveProject: async () => {
      const state = get();
      if (state.saving) return false;

      try {
        set({
          saving: true,
          error: null,
          ruleErrors: new Map(),
          validationErrors: new Map(),
          sectionErrors: [],
          hasInputSchemaErrors: false,
          hasOutputErrors: false,
          hasParameterErrors: false
        });

        const updatedProjectData = {
          ...state.project.project_data,
          rules: state.trees.map(tree => ({
            meta: {
              name: tree.name,
              ...(tree.description && { description: tree.description }),
              position: { x: 0, y: 0 }
            },
            rule: tree.rootNode
          }))
        };

        // Validation never blocks a save: the server returns what decikit found
        const saved = await flatRuleApi.updateProject(state.project.id, {
          project_data: updatedProjectData
        });

        const errorState = buildErrorState(saved.validation_errors, state.trees);
        set({ ...errorState, savedTrees: state.trees, savedProjectData: state.project.project_data });
        toast(saved.validation_errors.length === 0
          ? { title: "Project Saved", description: "Your changes have been saved successfully." }
          : {
              title: "Saved with validation issues",
              description: `${saved.validation_errors.length} issue(s) found. Affected rules are highlighted.`,
              duration: 10000,
            });
        return true;
      } catch (err: any) {
        console.error('Failed to save project:', err);
        toast({
          title: "Save Failed",
          description: apiErrorMessage(err, 'Failed to save project. Please try again.'),
          variant: "destructive",
        });
        return false;
      } finally {
        set({ saving: false });
      }
    },

    // Initialization
    initializeFromProject: (project) => {
      // Helper to ensure all nodes have IDs (recursively)
      const ensureNodeIds = (node: any): RuleTableRow => {
        if (!node) return node;

        const nodeWithId = {
          ...node,
          id: node.id || crypto.randomUUID()
        };
        // Legacy documents stored the ranges strictness as `strict_mode`
        if ('strict_mode' in nodeWithId) {
          nodeWithId.strict = nodeWithId.strict ?? nodeWithId.strict_mode;
          delete nodeWithId.strict_mode;
        }

        // Recursively process child nodes based on type
        if (nodeWithId.type === 'unary' || nodeWithId.type === 'composite') {
          return {
            ...nodeWithId,
            then: nodeWithId.then ? ensureNodeIds(nodeWithId.then) : nodeWithId.then,
            otherwise: nodeWithId.otherwise ? ensureNodeIds(nodeWithId.otherwise) : nodeWithId.otherwise,
            // Also ensure nested composite conditions have IDs
            conditions: nodeWithId.type === 'composite' && nodeWithId.conditions
              ? nodeWithId.conditions.map((cond: any) => {
                  if (cond.type === 'composite') {
                    return ensureNodeIds(cond);
                  }
                  return cond;
                })
              : nodeWithId.conditions
          } as RuleTableRow;
        }

        if (nodeWithId.type === 'cases') {
          return {
            ...nodeWithId,
            branches: nodeWithId.branches?.map((branch: any) => ensureNodeIds(branch))
          } as RuleTableRow;
        }

        return nodeWithId as RuleTableRow;
      };

      // Initialize trees directly from PED format - NO CONVERSION!
      const trees: PrioritizedTree[] =
        project.project_data?.rules && project.project_data.rules.length > 0
          ? project.project_data.rules.map((root, index) => ({
              id: crypto.randomUUID(),
              name: root.meta?.name || `Rule ${index + 1}`,
              priority: index + 1,
              rootNode: ensureNodeIds(root.rule) || createDefaultLeafNode(),
              description: root.meta?.description || ''
            }))
          : [{
              id: crypto.randomUUID(),
              name: 'Rule 1',
              priority: 1,
              rootNode: createDefaultLeafNode(),
              description: ''
            }];

      // Initialize with empty parameters if none exist and normalize dtypes
      const enhancedProject = {
        ...project,
        project_data: {
          ...project.project_data,
          parameters: project.project_data?.parameters || {},
          output: project.project_data?.output ? {
            ...project.project_data.output,
            dtypes: normalizeDtypes(project.project_data.output.dtypes || [])
          } : {
            data: [],
            default: null,
            dtypes: [],
            type_defs: {}
          }
        }
      };

      set({
        project: enhancedProject,
        trees,
        expandedTrees: new Set(trees.map(t => t.id)),
        availableFields: computeAvailableFields(enhancedProject.project_data),
        outputOptions: computeOutputOptions(enhancedProject.project_data),
        // Reset other state
        editingTreeName: null,
        copiedNode: null,
        collapsedNodes: new Set(),
        focusedNode: null,
        saving: false,
        savedTrees: trees,
        savedProjectData: enhancedProject.project_data,
        error: null,
        ...buildErrorState(project.validation_errors, trees)
      });
    },

    // Internal helpers
    updateProjectData: (updates) => {
      set((state) => {
        const newProject = {
          ...state.project,
          project_data: {
            ...state.project.project_data,
            ...updates
          }
        };
        return {
          project: newProject,
          availableFields: computeAvailableFields(newProject.project_data),
          outputOptions: computeOutputOptions(newProject.project_data)
        };
      });
    }
  }))
);

// Enable background token refresh on store activity
subscribeToStoreActivity(store);

export const useFlatRuleStore = store;

// Create stable selectors for better performance
export const useFlatRuleTrees = () => useFlatRuleStore((state) => state.trees);
export const useFlatRuleProject = () => useFlatRuleStore((state) => state.project);
export const useFlatRuleExpandedTrees = () => useFlatRuleStore((state) => state.expandedTrees);
export const useFlatRuleEditingTreeName = () => useFlatRuleStore((state) => state.editingTreeName);
export const useFlatRuleCopiedNode = () => useFlatRuleStore((state) => state.copiedNode);
export const useFlatRuleCollapsedNodes = () => useFlatRuleStore((state) => state.collapsedNodes);
export const useFlatRuleFocusedNode = () => useFlatRuleStore((state) => state.focusedNode);
export const useFlatRuleAvailableFields = () => useFlatRuleStore((state) => state.availableFields);
export const useFlatRuleOutputOptions = () => useFlatRuleStore((state) => state.outputOptions);
export const useFlatRuleSaving = () => useFlatRuleStore((state) => state.saving);
export const useFlatRuleDirty = () => useFlatRuleStore(
  (state) => state.trees !== state.savedTrees || state.project.project_data !== state.savedProjectData
);
export const useFlatRuleError = () => useFlatRuleStore((state) => state.error);
export const useFlatRuleRuleErrors = () => useFlatRuleStore((state) => state.ruleErrors);
export const useFlatRuleValidationErrors = () => useFlatRuleStore((state) => state.validationErrors);
export const useFlatRuleSectionErrors = () => useFlatRuleStore((state) => state.sectionErrors);
export const useFlatRuleHasInputSchemaErrors = () => useFlatRuleStore((state) => state.hasInputSchemaErrors);
export const useFlatRuleHasOutputErrors = () => useFlatRuleStore((state) => state.hasOutputErrors);
export const useFlatRuleHasParameterErrors = () => useFlatRuleStore((state) => state.hasParameterErrors);


// Individual action selectors
export const useFlatRuleSetTrees = () => useFlatRuleStore((state) => state.setTrees);
export const useFlatRuleSetProject = () => useFlatRuleStore((state) => state.setProject);
export const useFlatRuleSetExpandedTrees = () => useFlatRuleStore((state) => state.setExpandedTrees);
export const useFlatRuleSetEditingTreeName = () => useFlatRuleStore((state) => state.setEditingTreeName);
export const useFlatRuleSetCopiedNode = () => useFlatRuleStore((state) => state.setCopiedNode);
export const useFlatRuleSetCollapsedNodes = () => useFlatRuleStore((state) => state.setCollapsedNodes);
export const useFlatRuleSetFocusedNode = () => useFlatRuleStore((state) => state.setFocusedNode);
export const useFlatRuleUpdateNodeValue = () => useFlatRuleStore((state) => state.updateNodeValue);
export const useFlatRuleUpdateConditionInGroup = () => useFlatRuleStore((state) => state.updateConditionInGroup);
export const useFlatRuleSaveAsNewOutput = () => useFlatRuleStore((state) => state.saveAsNewOutput);
export const useFlatRuleSaveToExistingOutput = () => useFlatRuleStore((state) => state.saveToExistingOutput);
export const useFlatRuleSaveProject = () => useFlatRuleStore((state) => state.saveProject);
export const useFlatRuleInitializeFromProject = () => useFlatRuleStore((state) => state.initializeFromProject);

// Parameter management selectors
export const useFlatRuleCreateParameter = () => useFlatRuleStore((state) => state.createParameter);
export const useFlatRuleUpdateParameter = () => useFlatRuleStore((state) => state.updateParameter);
export const useFlatRuleDeleteParameter = () => useFlatRuleStore((state) => state.deleteParameter);
export const useFlatRuleIsParameterInUse = () => useFlatRuleStore((state) => state.isParameterInUse);