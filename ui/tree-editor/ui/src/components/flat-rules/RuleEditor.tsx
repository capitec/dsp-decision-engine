import React, { useState, useCallback, useMemo } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Plus, ChevronDown, ChevronRight, MoveUp, MoveDown,
  Edit2, Check, X, Trash2, Settings2, Search, AlertTriangle, Copy
} from "lucide-react";
import { TreeRenderer } from './TreeRenderer';
import { copyWithNewIds } from './copyNode';
import type {
  FlatRuleProjectDetail,
  PrioritizedTree,
  RuleTableRow
} from './types';
import {
  useFlatRuleTrees,
  useFlatRuleExpandedTrees,
  useFlatRuleEditingTreeName,
  useFlatRuleCopiedNode,
  useFlatRuleFocusedNode,
  useFlatRuleRuleErrors,
  useFlatRuleValidationErrors,
  useFlatRuleSectionErrors,
  useFlatRuleSetTrees,
  useFlatRuleSetExpandedTrees,
  useFlatRuleSetEditingTreeName,
  useFlatRuleSetFocusedNode,
  useFlatRuleSetCopiedNode
} from './useFlatRuleStore';
import { ParameterManagementDialog } from '../shared-tree/ParameterManagementDialog';
import { useFlatRuleParameterManagement } from './useFlatRuleParameterManagement';
import { PolarsDataType } from './output-types';

// Error display component (separate to avoid hooks rule violation)
const RuleErrorDisplay: React.FC<{ errorMessage: string }> = ({ errorMessage }) => {
  const errorLines = errorMessage.split('\n').filter(line => line.trim().startsWith('•'));
  const [showAll, setShowAll] = React.useState(false);
  const maxVisible = 3;
  const visibleErrors = showAll ? errorLines : errorLines.slice(0, maxVisible);
  const hasMore = errorLines.length > maxVisible;

  return (
    <div className="mt-1 space-y-1" data-testid="rule-errors">
      <div className="flex items-start gap-1.5 text-red-600 text-sm">
        <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <div className="font-medium mb-1">{errorLines.length} validation {errorLines.length === 1 ? 'error' : 'errors'}:</div>
          <ul className="space-y-0.5 text-xs">
            {visibleErrors.map((line, idx) => (
              <li key={idx}>{line}</li>
            ))}
          </ul>
          {hasMore && !showAll && (
            <button
              onClick={() => setShowAll(true)}
              className="text-xs text-red-700 hover:text-red-800 underline mt-1"
            >
              Show {errorLines.length - maxVisible} more...
            </button>
          )}
          {showAll && hasMore && (
            <button
              onClick={() => setShowAll(false)}
              className="text-xs text-red-700 hover:text-red-800 underline mt-1"
            >
              Show less
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// Helper function to infer parameter type from value
const inferParameterType = (value: any): PolarsDataType => {
  if (Array.isArray(value)) return 'String'; // Default for arrays
  if (typeof value === 'boolean') return 'Boolean';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? 'Int64' : 'Float64';
  }
  return 'String';
};

interface RuleEditorProps {
  // No props needed - context handles everything
}

export const RuleEditor: React.FC<RuleEditorProps> = () => {
  // Use individual Zustand store hooks for all state management
  const trees = useFlatRuleTrees();
  const expandedTrees = useFlatRuleExpandedTrees();
  const editingTreeName = useFlatRuleEditingTreeName();
  const copiedNode = useFlatRuleCopiedNode();
  const focusedNode = useFlatRuleFocusedNode();
  const ruleErrors = useFlatRuleRuleErrors();
  const validationErrors = useFlatRuleValidationErrors();
  const parameterErrors = useFlatRuleSectionErrors().filter(e => e.section === 'parameters');
  const setExpandedTrees = useFlatRuleSetExpandedTrees();
  const setEditingTreeName = useFlatRuleSetEditingTreeName();
  const setFocusedNode = useFlatRuleSetFocusedNode();
  const setTrees = useFlatRuleSetTrees();
  const setCopiedNode = useFlatRuleSetCopiedNode();

  // Parameter management
  const {
    parameters,
    parametersForDialog,
    availableParameters,
    isParameterInUse,
    saveParameters
  } = useFlatRuleParameterManagement();

  // Local UI state for drag and drop
  const [draggedNode, setDraggedNode] = useState<{node: RuleTableRow, treeId: string} | null>(null);
  const [editingTreeDescription, setEditingTreeDescription] = useState<string | null>(null);
  const [parametersDialogOpen, setParametersDialogOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [deleteConfirmation, setDeleteConfirmation] = useState<{ treeId: string; treeName: string } | null>(null);

  const addTree = useCallback(() => {
    setTrees(prev => [
      ...prev,
      {
        id: crypto.randomUUID(),
        name: `Rule ${prev.length + 1}`,
        priority: prev.length + 1,
        rootNode: {
          id: crypto.randomUUID(),
          type: 'leaf',
          result_idx: -1,
          editMode: false
        },
        description: ''
      }
    ]);
  }, [setTrees]);

  const deleteTree = useCallback((treeId: string, treeName: string) => {
    setDeleteConfirmation({ treeId, treeName });
  }, []);

  const confirmDeleteTree = useCallback(() => {
    if (deleteConfirmation) {
      setTrees(prev => prev.filter(t => t.id !== deleteConfirmation.treeId));
      setDeleteConfirmation(null);
    }
  }, [deleteConfirmation, setTrees]);

  const cloneTree = useCallback((treeId: string) => {
    const tree = trees.find(t => t.id === treeId);
    if (!tree) return;

    // Find a unique name by checking for existing copies
    const baseName = tree.name;
    const existingCopyNumbers: number[] = [];

    // Extract copy numbers from existing trees
    trees.forEach(t => {
      // Match "Base Name (Copy N)" or "Base Name (Copy)"
      const copyMatch = t.name.match(new RegExp(`^${baseName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} \\(Copy(?: (\\d+))?\\)$`));
      if (copyMatch) {
        const num = copyMatch[1] ? parseInt(copyMatch[1], 10) : 1;
        existingCopyNumbers.push(num);
      }
    });

    // Find the next available copy number
    const nextCopyNumber = existingCopyNumbers.length > 0
      ? Math.max(...existingCopyNumbers) + 1
      : 1;

    const clonedTree = {
      ...tree,
      id: crypto.randomUUID(),
      name: `${baseName} (Copy ${nextCopyNumber})`,
      priority: trees.length + 1,
      rootNode: copyWithNewIds(tree.rootNode)
    };

    setTrees(prev => [...prev, clonedTree]);
  }, [trees, setTrees]);

  const renameTree = useCallback((treeId: string, newName: string) => {
    setTrees(prev => prev.map(tree => 
      tree.id === treeId ? { ...tree, name: newName } : tree
    ));
  }, [setTrees]);

  // The name is renamed as it is typed; cancelling, or finishing with an empty name, puts the old one back
  const [nameBeforeEdit, setNameBeforeEdit] = useState('');
  const finishRename = (tree: PrioritizedTree, keep: boolean) => {
    if (!keep || !tree.name.trim()) renameTree(tree.id, nameBeforeEdit);
    setEditingTreeName(null);
  };

  const moveTree = useCallback((treeId: string, direction: 'up' | 'down') => {
    setTrees(prev => {
      const index = prev.findIndex(t => t.id === treeId);
      if (index === -1) return prev;
      
      const newIndex = direction === 'up' ? index - 1 : index + 1;
      if (newIndex < 0 || newIndex >= prev.length) return prev;
      
      const newTrees = [...prev];
      [newTrees[index], newTrees[newIndex]] = [newTrees[newIndex], newTrees[index]];
      
      // Update priorities
      return newTrees.map((tree, idx) => ({ ...tree, priority: idx + 1 }));
    });
  }, [setTrees]);

  const toggleTreeExpansion = useCallback((treeId: string) => {
    setExpandedTrees(prev => {
      const updated = new Set(prev);
      if (updated.has(treeId)) {
        updated.delete(treeId);
      } else {
        updated.add(treeId);
      }
      return updated;
    });
  }, [setExpandedTrees]);

  const updateNode = useCallback((treeId: string, nodeId: string, updates: Partial<RuleTableRow>) => {
    setTrees(prev => prev.map(tree => {
      if (tree.id !== treeId) return tree;
      
      const updateNodeRecursively = (node: RuleTableRow): RuleTableRow => {
        if (node.id === nodeId) {
          return { ...node, ...updates } as RuleTableRow;
        }
        
        // Handle cases nodes with branches
        if (node.type === 'cases') {
          return {
            ...node,
            branches: node.branches?.map(branch => updateNodeRecursively(branch)),
          } as RuleTableRow;
        }

        // Handle unary/composite nodes with then/otherwise
        if (node.type === 'unary' || node.type === 'composite') {
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
        rootNode: updateNodeRecursively(tree.rootNode)
      };
    }));
  }, [setTrees]);


  // TODO: Move these to actions once deleteBranchAction and deleteNodeAction are implemented
  const deleteBranch = useCallback((treeId: string, branchId: string) => {
    // Check if this is actually deleting the otherwise branch (branchId === 'otherwise')
    if (branchId === 'otherwise') {
      // Find the parent node and remove its otherwise field
      const tree = trees.find(t => t.id === treeId);
      if (!tree) return;
      
      const removeOtherwiseRecursively = (node: RuleTableRow): RuleTableRow => {
        // Check if this node's otherwise is the one to remove
        if ((node.type === 'unary' || node.type === 'composite') &&
            node.otherwise && node.otherwise.id === branchId.replace('otherwise-', '')) {
          return { ...node, otherwise: undefined } as RuleTableRow;
        }

        // Recurse into cases nodes
        if ((node.type === 'cases') && node.branches) {
          return {
            ...node,
            branches: node.branches.map(branch => removeOtherwiseRecursively(branch)),
          } as RuleTableRow;
        }

        // Recurse into unary/composite nodes
        if ((node.type === 'unary' || node.type === 'composite')) {
          return {
            ...node,
            then: node.then ? removeOtherwiseRecursively(node.then) : node.then,
            otherwise: node.otherwise ? removeOtherwiseRecursively(node.otherwise) : node.otherwise
          } as RuleTableRow;
        }

        return node;
      };
      
      const updatedTree = {
        ...tree,
        rootNode: removeOtherwiseRecursively(tree.rootNode)
      };
      
      updateNode(treeId, tree.rootNode.id, updatedTree.rootNode);
      return;
    }
    
    // Handle regular branch deletion
    const tree = trees.find(t => t.id === treeId);
    if (!tree) return;

    const deleteBranchRecursively = (node: RuleTableRow): RuleTableRow => {
      // Handle cases nodes with branches
      if ((node.type === 'cases') && node.branches) {
        const filteredBranches = node.branches.filter(branch => branch.id !== branchId);
        if (filteredBranches.length !== node.branches.length) {
          return {
            ...node,
            branches: filteredBranches
          } as RuleTableRow;
        }

        return {
          ...node,
          branches: node.branches.map(branch => deleteBranchRecursively(branch)),
        } as RuleTableRow;
      }

      // Handle unary/composite nodes
      if ((node.type === 'unary' || node.type === 'composite')) {
        return {
          ...node,
          then: node.then ? deleteBranchRecursively(node.then) : node.then,
          otherwise: node.otherwise ? deleteBranchRecursively(node.otherwise) : node.otherwise
        } as RuleTableRow;
      }

      return node;
    };

    const updatedTree = {
      ...tree,
      rootNode: deleteBranchRecursively(tree.rootNode)
    };

    updateNode(treeId, tree.rootNode.id, updatedTree.rootNode);
  }, [trees, updateNode]);


  const copyNode = useCallback((node: RuleTableRow) => {
    setCopiedNode(node);
  }, [setCopiedNode]);

  const pasteNode = useCallback((treeId: string, targetBranchId: string, nodeToUse?: RuleTableRow) => {
    const nodeToPaste = nodeToUse || copiedNode;
    if (nodeToPaste) {
      updateNode(treeId, targetBranchId, copyWithNewIds(nodeToPaste));
    }
  }, [updateNode, copiedNode]);




  const handleDragStart = useCallback((e: React.DragEvent, node: RuleTableRow, treeId: string) => {
    setDraggedNode({ node, treeId });
    e.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, targetBranchId: string, treeId: string) => {
    e.preventDefault();
    if (!draggedNode) return;
    
    // Don't allow dropping onto itself or moving to same tree if it's the root
    if (draggedNode.treeId === treeId) {
      const sourceTree = trees.find(t => t.id === draggedNode.treeId);
      if (sourceTree?.rootNode.id === draggedNode.node.id) {
        setDraggedNode(null);
        return;
      }
    }
    
    // First remove the node from its current position
    const copiedDraggedNode = { ...draggedNode.node, id: crypto.randomUUID() };
    
    // Then add it to the new position
    pasteNode(treeId, targetBranchId);
    
    setDraggedNode(null);
  }, [draggedNode, trees, pasteNode]);

  // TODO: Move this to actions once addBranchAction is implemented
  const addBranch = useCallback((treeId: string, nodeId: string) => {
    const tree = trees.find(t => t.id === treeId);
    if (!tree) return;
    
    // Find the target node to get its current branches and type
    const findTargetNode = (node: RuleTableRow): RuleTableRow | null => {
      if (node.id === nodeId) return node;

      // Handle cases nodes with branches
      if ((node.type === 'cases') && node.branches) {
        for (const branch of node.branches) {
          const found = findTargetNode(branch);
          if (found) return found;
        }
      }

      // Handle unary/composite nodes
      if ((node.type === 'unary' || node.type === 'composite')) {
        if (node.then) {
          const found = findTargetNode(node.then);
          if (found) return found;
        }
        if (node.otherwise) {
          const found = findTargetNode(node.otherwise);
          if (found) return found;
        }
      }

      return null;
    };
    
    const targetNode = findTargetNode(tree.rootNode);
    if (!targetNode || targetNode.type === 'leaf') return;

    // Add a new branch node to the cases node
    const newBranchNode: RuleTableRow = {
      id: crypto.randomUUID(),
      type: 'leaf',
      result_idx: -1,
      editMode: false
    };

    const currentBranches = targetNode.type === 'cases'
      ? (targetNode.branches || [])
      : [];
    
    // Add new branch to existing branches (otherwise is handled separately)
    const updatedBranches = [...currentBranches, newBranchNode];

    // Use updateNode with just the branches update
    updateNode(treeId, nodeId, { branches: updatedBranches } as any);
  }, [trees, updateNode]);

  const addOtherwiseBranch = useCallback((treeId: string, nodeId: string) => {
    // Use updateNodeAction to add otherwise field directly to the node
    const createDefaultLeafNode = (): RuleTableRow => ({
      id: crypto.randomUUID(),
      type: 'leaf',
      result_idx: -1,
      editMode: false
    });
    
    updateNode(treeId, nodeId, { otherwise: createDefaultLeafNode() });
  }, [updateNode]);



  // Helper function to find a node by ID in a tree
  const findNodeById = (node: RuleTableRow, targetId: string): RuleTableRow | null => {
    if (node.id === targetId) {
      return node;
    }

    // Handle cases nodes with branches
    if ((node.type === 'cases') && node.branches) {
      for (const branch of node.branches) {
        const found = findNodeById(branch, targetId);
        if (found) return found;
      }
    }

    // Handle unary/composite nodes
    if ((node.type === 'unary' || node.type === 'composite')) {
      if (node.then) {
        const found = findNodeById(node.then, targetId);
        if (found) return found;
      }
      if (node.otherwise) {
        const found = findNodeById(node.otherwise, targetId);
        if (found) return found;
      }
    }

    return null;
  };

  // Filter trees based on search query
  const filteredTrees = useMemo(() => {
    if (!searchQuery.trim()) return trees;

    const query = searchQuery.toLowerCase();
    return trees.filter(tree =>
      tree.name.toLowerCase().includes(query) ||
      (tree.description && tree.description.toLowerCase().includes(query)) ||
      tree.priority.toString().includes(query)
    );
  }, [trees, searchQuery]);


  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Prioritized Decision Trees</h2>
        <div className="flex gap-2 items-center">
          <Button variant="outline" size="sm" onClick={() => setParametersDialogOpen(true)}>
            <Settings2 className="h-4 w-4 mr-2" />
            Manage Parameters
            {parameters.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full text-xs">
                {parameters.length}
              </span>
            )}
          </Button>
          <Button onClick={addTree}>
            <Plus className="w-4 h-4 mr-2" />
            Add Rule
          </Button>
        </div>
      </div>
      {parameterErrors.length > 0 && (
        <div className="text-sm text-red-600" data-testid="parameter-errors">
          {parameterErrors.map(e => <div key={e.path}>{e.message}</div>)}
        </div>
      )}

      {/* Search Bar */}
      {trees.length > 0 && (
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            type="text"
            placeholder="Search rules by name, description, or priority..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-9"
          />
          {searchQuery && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSearchQuery('')}
              className="absolute right-1 top-1/2 transform -translate-y-1/2 h-7 px-2"
            >
              <X className="h-3 w-3" />
            </Button>
          )}
        </div>
      )}

      {/* Show message if no results */}
      {filteredTrees.length === 0 && searchQuery && (
        <div className="text-center py-8 text-gray-500">
          No rules found matching "{searchQuery}"
        </div>
      )}

      {filteredTrees.map((tree, index) => {
        const hasError = ruleErrors.has(tree.id);
        return (
        <Card
          key={tree.id}
          data-testid="rule-card"
          className={`border-2 transition-all duration-500 ease-in-out transform ${hasError ? 'border-red-500 bg-red-50' : ''}`}
          style={{
            transform: `translateY(0px)`,
            transition: 'all 0.5s ease-in-out'
          }}
        >
          <CardHeader className="px-4 py-3 pb-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => toggleTreeExpansion(tree.id)}
                  data-testid="rule-toggle"
                  className="h-7 px-2"
                >
                  {expandedTrees.has(tree.id) ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </Button>

                <span className="px-2.5 py-0.5 bg-gray-100 rounded text-xs font-mono">
                  #{tree.priority}
                </span>

                {editingTreeName === tree.id ? (
                  <div className="flex items-center gap-2">
                    <Input
                      value={tree.name}
                      onChange={(e) => renameTree(tree.id, e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === 'Escape') finishRename(tree, e.key === 'Enter');
                      }}
                      className="w-48 h-8 text-sm"
                      autoFocus
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => finishRename(tree, true)}
                      data-testid="rule-rename-done"
                      className="h-7 px-2"
                    >
                      <Check className="w-3.5 h-3.5 text-green-600" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => finishRename(tree, false)}
                      className="h-7 px-2"
                    >
                      <X className="w-3.5 h-3.5 text-red-600" />
                    </Button>
                  </div>
                ) : (
                  <div className="flex flex-col gap-0.5">
                    <div className="flex items-center gap-1.5">
                      <h3 className="text-base font-semibold" data-testid="rule-name">{tree.name}</h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => { setNameBeforeEdit(tree.name); setEditingTreeName(tree.id); }}
                        data-testid="rule-rename"
                        className="h-6 px-1.5"
                      >
                        <Edit2 className="w-3 h-3" />
                      </Button>
                    </div>

                    {/* Error Display */}
                    {hasError && (
                      <RuleErrorDisplay errorMessage={ruleErrors.get(tree.id) || ''} />
                    )}

                    {/* Editable Description */}
                    {editingTreeDescription === tree.id ? (
                      <div className="flex items-center gap-2">
                        <Input
                          value={tree.description || ''}
                          onChange={(e) => {
                            setTrees(prev => prev.map(t =>
                              t.id === tree.id ? { ...t, description: e.target.value } : t
                            ));
                          }}
                          className="text-xs text-gray-600 w-64 h-7"
                          placeholder="Add description..."
                          autoFocus
                        />
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setEditingTreeDescription(null)}
                          className="h-6 px-1.5"
                        >
                          <Check className="w-3 h-3 text-green-600" />
                        </Button>
                      </div>
                    ) : (
                      <div
                        className="flex items-center gap-1 cursor-pointer group"
                        onClick={() => setEditingTreeDescription(tree.id)}
                      >
                        <p className="text-xs text-gray-600 hover:text-gray-800">
                          {tree.description || 'Add description...'}
                        </p>
                        <Edit2 className="w-2.5 h-2.5 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="flex items-center gap-0.5">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => moveTree(tree.id, 'up')}
                  disabled={index === 0}
                  data-testid="rule-move-up"
                  className="h-7 px-2"
                >
                  <MoveUp className="w-3.5 h-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => moveTree(tree.id, 'down')}
                  disabled={index === trees.length - 1}
                  data-testid="rule-move-down"
                  className="h-7 px-2"
                >
                  <MoveDown className="w-3.5 h-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => cloneTree(tree.id)}
                  className="text-blue-600 hover:text-blue-700 h-7 px-2"
                  title="Clone Rule"
                >
                  <Copy className="w-3.5 h-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => deleteTree(tree.id, tree.name)}
                  disabled={trees.length === 1}
                  data-testid="rule-delete"
                  className="text-red-600 hover:text-red-700 h-7 px-2"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </Button>
              </div>
            </div>
          </CardHeader>

          {expandedTrees.has(tree.id) && (
            <CardContent className="px-4 py-3">
              <div className="text-sm text-gray-600 mb-3">
                Trees execute in priority order using first-match logic. Multiple conditions in a group use OR logic.
                {!focusedNode && <span className="block text-xs mt-1 text-gray-500">💡 Click the # button on inputs to use parameter references</span>}
                {focusedNode && focusedNode.treeId === tree.id && (
                  <span className="block text-xs mt-1 text-blue-600">🔍 Focused on subtree at depth {focusedNode.depth} - showing only this branch</span>
                )}
              </div>
              {focusedNode && focusedNode.treeId === tree.id && (
                <div className="mb-3 p-2 bg-blue-50 border border-blue-200 rounded flex items-center justify-between">
                  <span className="text-sm text-blue-700 font-medium">📍 Focused view - showing subtree from depth {focusedNode.depth}</span>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => setFocusedNode(null)}
                    className="h-6 px-2 text-xs"
                  >
                    Show All
                  </Button>
                </div>
              )}
              
              {/* Render either the full tree or the focused subtree */}
              {(() => {
                // Get validation errors for this rule
                const ruleIndex = trees.findIndex(t => t.id === tree.id);
                const errorsForRule = ruleIndex >= 0 ? validationErrors.get(ruleIndex) || [] : [];

                if (focusedNode && focusedNode.treeId === tree.id) {
                  // Find the focused node and render it as root
                  const focusedTreeNode = findNodeById(tree.rootNode, focusedNode.nodeId);
                  return focusedTreeNode ? (
                    <TreeRenderer
                      rootNode={focusedTreeNode}
                      treeId={tree.id}
                      startDepth={focusedNode.depth}
                      availableParameters={availableParameters}
                      onManageParameters={() => setParametersDialogOpen(true)}
                      onDragStart={handleDragStart}
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      onCopyNode={copyNode}
                      onDeleteBranch={deleteBranch}
                      onAddBranch={addBranch}
                      onAddOtherwiseBranch={addOtherwiseBranch}
                      nodeErrors={errorsForRule}
                    />
                  ) : (
                    <div className="text-red-500 text-sm">Focused node not found</div>
                  );
                } else {
                  // Render the full tree
                  return (
                    <TreeRenderer
                      rootNode={tree.rootNode}
                      treeId={tree.id}
                      startDepth={0}
                      availableParameters={availableParameters}
                      onManageParameters={() => setParametersDialogOpen(true)}
                      onDragStart={handleDragStart}
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      onCopyNode={copyNode}
                      onDeleteBranch={deleteBranch}
                      onAddBranch={addBranch}
                      onAddOtherwiseBranch={addOtherwiseBranch}
                      nodeErrors={errorsForRule}
                    />
                  );
                }
              })()}
            </CardContent>
          )}
        </Card>
      );
      })}

      {/* Parameters Dialog */}
      <ParameterManagementDialog
        open={parametersDialogOpen}
        onOpenChange={setParametersDialogOpen}
        parameters={parametersForDialog}
        onSave={saveParameters}
        isParameterInUse={isParameterInUse}
        title="Manage Parameters"
        description="Create and manage reusable parameters for your decision tree rules"
      />

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteConfirmation} onOpenChange={(open) => !open && setDeleteConfirmation(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              Delete Rule?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete <strong>{deleteConfirmation?.treeName}</strong>?
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDeleteTree}
              className="bg-red-600 hover:bg-red-700 focus:ring-red-600"
            >
              Delete Rule
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};