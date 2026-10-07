import React, { useCallback, useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from "@/components/ui/alert-dialog";
import {
  Trash2, Plus, ChevronDown, ChevronRight,
  GripVertical, Copy, Clipboard, Eye, EyeOff
} from "lucide-react";
import { FlatRuleLeafNodeEditor } from './FlatRuleLeafNodeEditor';
import { ConditionEditor } from './ConditionEditor';
import { ParameterizedNumericInput } from '../shared-tree/ParameterizedNumericInput';
import { ParameterizedCategoricalInput } from '../shared-tree/ParameterizedCategoricalInput';
import { ParameterizedStringInput } from '../shared-tree/ParameterizedStringInput';
import { UnaryConditionForm } from '../shared-tree/UnaryConditionForm';
import { UnaryOperatorSelector } from '../shared-tree/UnaryOperatorSelector';
import { CompositeLogicSelector } from '../shared-tree/CompositeLogicSelector';
import { ComputedFeatureInput } from '../shared-tree/ComputedFeatureInput';
import { CompositeConditionEditor } from './CompositeConditionEditor';
import { hasErrorAtPath, getErrorAtPath } from '@/utils/flat-rule-validation-parser';
import type { RuleTableRow, RuleBranch, RangeCondition, RuleCondition, LeafRuleNode, CompositeRuleNode } from './types';
import { ParameterOption } from '../shared-tree/ParameterizedNumericInput';
import {
  useFlatRuleCollapsedNodes,
  useFlatRuleFocusedNode,
  useFlatRuleAvailableFields,
  useFlatRuleCopiedNode,
  useFlatRuleSetCollapsedNodes,
  useFlatRuleSetFocusedNode,
  useFlatRuleSetTrees
} from './useFlatRuleStore';

// Configuration constants
const MAX_DEPTH_BEFORE_FOCUS = 10;

function createDefaultLeafNode(): RuleTableRow {
  return {
    id: crypto.randomUUID(),
    type: 'leaf',
    result_idx: -1,
    editMode: false
  };
}

interface NodeBlockProps {
  node: RuleTableRow;
  treeId: string;
  depth?: number;
  absoluteDepth?: number;
  availableParameters: ParameterOption[];
  onManageParameters?: () => void;
  onDragStart: (e: React.DragEvent, node: RuleTableRow, treeId: string) => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent, targetBranchId: string, treeId: string) => void;
  onCopyNode: (node: RuleTableRow) => void;
  onDeleteBranch: (treeId: string, branchId: string) => void;
  onAddBranch: (treeId: string, nodeId: string) => void;
  onAddOtherwiseBranch: (treeId: string, nodeId: string) => void;
  nodeErrors?: any[]; // Validation errors for this node
  basePath?: (string | number)[]; // Path to this node in the rule tree
}

export const NodeBlock: React.FC<NodeBlockProps> = ({
  node,
  treeId,
  depth = 0,
  absoluteDepth = 0,
  availableParameters,
  onManageParameters,
  onDragStart,
  onDragOver,
  onDrop,
  onCopyNode,
  onDeleteBranch,
  onAddBranch,
  onAddOtherwiseBranch,
  nodeErrors = [],
  basePath = []
}) => {
  const collapsedNodes = useFlatRuleCollapsedNodes();
  const focusedNode = useFlatRuleFocusedNode();
  const availableFeatures = useFlatRuleAvailableFields();
  const copiedNode = useFlatRuleCopiedNode();
  const setCollapsedNodes = useFlatRuleSetCollapsedNodes();
  const setFocusedNode = useFlatRuleSetFocusedNode();
  const setTrees = useFlatRuleSetTrees();
  const [strictModeConfirmOpen, setStrictModeConfirmOpen] = useState(false);

  const updateNode = useCallback((treeId: string, nodeId: string, updates: any) => {
    setTrees(prev => prev.map(tree => {
      if (tree.id !== treeId) return tree;

      const updateNodeRecursively = (node: RuleTableRow): RuleTableRow => {
        if (node.id === nodeId) {
          // For type changes, replace the entire node
          if (updates.type && updates.type !== node.type) {
            return updates as RuleTableRow;
          }
          // For field updates, merge with existing node
          return { ...node, ...updates } as RuleTableRow;
        }

        // Unary and Composite nodes have then/otherwise fields
        if (node.type === 'unary' || node.type === 'composite') {
          return {
            ...node,
            then: node.then ? updateNodeRecursively(node.then) : node.then,
            otherwise: node.otherwise ? updateNodeRecursively(node.otherwise) : node.otherwise
          } as RuleTableRow;
        }

        // Cases nodes have branches array (direct RuleTableRow children) and otherwise
        if (node.type === 'cases') {
          return {
            ...node,
            branches: node.branches?.map(branch => updateNodeRecursively(branch)),
            otherwise: typeof node.otherwise === 'number' ? node.otherwise : undefined
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

  // Check if enabling strict mode would modify conditions
  const wouldStrictModeModify = useCallback((conditions: any[]): boolean => {
    if (conditions.length === 0) return false;

    // Check if any lower bounds are set (not null) in when clause
    const hasLowerBounds = conditions.some((cond: any) => cond.when?.min != null);

    // Check if conditions are already sorted by max
    const isSorted = conditions.every((cond: any, idx: number) => {
      if (idx === 0) return true;
      const prevMax = conditions[idx - 1].when?.max ?? Infinity;
      const currMax = cond.when?.max ?? Infinity;
      return prevMax <= currMax;
    });

    return hasLowerBounds || !isSorted;
  }, []);

  // Apply strict mode changes
  const applyStrictMode = useCallback(() => {
    const conditions = node.type === 'cases' && node.op === 'ranges' ? node.conditions : [];
    const updatedConditions = conditions
      .map((cond: any) => ({
        ...cond,
        when: { ...cond.when, min: null }
      }))
      .sort((a: any, b: any) => {
        const aMax = a.when?.max ?? Infinity;
        const bMax = b.when?.max ?? Infinity;
        return aMax - bMax;
      });

    updateNode(treeId, node.id, {
      strict: true,
      conditions: updatedConditions
    } as any);
  }, [node, treeId, updateNode]);

  // Check for errors at this node level
  const nodeHasError = hasErrorAtPath(nodeErrors, basePath);

  const bgColor = nodeHasError ? 'bg-red-50 border-red-500' :
                 node.type === 'leaf' ? 'bg-green-50 border-green-200' :
                 node.type === 'unary' ? 'bg-blue-50 border-blue-200' :
                 node.type === 'cases' && node.op === 'isin' ? 'bg-purple-50 border-purple-200' :
                 node.type === 'cases' && node.op === 'string_match' ? 'bg-amber-50 border-amber-200' :
                 node.type === 'cases' && node.op === 'ranges' ? 'bg-orange-50 border-orange-200' :
                 node.type === 'composite' ? 'bg-indigo-50 border-indigo-200' :
                 'bg-gray-50 border-gray-200';

  const textColor = node.type === 'leaf' ? 'text-green-700' :
                   node.type === 'unary' ? 'text-blue-700' :
                   node.type === 'cases' && node.op === 'isin' ? 'text-purple-700' :
                   node.type === 'cases' && node.op === 'string_match' ? 'text-amber-700' :
                   node.type === 'cases' && node.op === 'ranges' ? 'text-orange-700' :
                   node.type === 'composite' ? 'text-indigo-700' :
                   'text-gray-700';

  const isCollapsed = collapsedNodes.has(node.id);
  const hasChildren = 'branches' in node && node.branches && node.branches.length > 0;
  
  // Focus logic for the tree
  const isFocused = focusedNode?.nodeId === node.id && focusedNode?.treeId === treeId;
  
  const maxDepthBeforeFocus = MAX_DEPTH_BEFORE_FOCUS;
  const isInFocusedSubtree = focusedNode && focusedNode.treeId === treeId;
  const needsFocus = absoluteDepth >= maxDepthBeforeFocus && !isInFocusedSubtree;
  const canExpand = !needsFocus || isCollapsed;
  const showFocusButton = absoluteDepth >= 5; // Show focus button at depth 5+
  
  // No indentation - all nodes align flush left
  const indentSize = 0;

  const toggleCollapse = () => {
    setCollapsedNodes(prev => {
      const updated = new Set(prev);
      if (updated.has(node.id)) {
        updated.delete(node.id);
      } else {
        updated.add(node.id);
      }
      return updated;
    });
  };

  // Helper to create default nodes for unary/composite
  const createDefaultLeaf = () => ({
    id: crypto.randomUUID(),
    type: 'leaf' as const,
    result_idx: -1
  });

  const createDefaultUnaryCondition = () => ({
    type: 'unary' as const,
    op: '<=' as const,
    feature: node.type !== 'leaf' && 'feature' in node ? (node as any).feature || '' : '',
    threshold: 0
  });

  const createDefaultComposite = () => ({
    type: 'composite' as const,
    op: 'and' as const,
    conditions: []
  });

  return (
    <div
      key={node.id}
      className={`border rounded-md p-2 mb-1 ${bgColor} ${textColor} ${
        isFocused ? 'ring-2 ring-blue-400 shadow-md' : ''
      } ${nodeHasError ? 'ring-2 ring-red-500' : ''}`}
      style={{ marginLeft: `${indentSize}px` }}
    >
      {/* Error indicator badge */}
      {nodeHasError && (
        <div className="absolute -top-2 -left-2 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
          !
        </div>
      )}

      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-1.5">
          <div 
            className="w-4 h-4 text-gray-500 cursor-move"
            draggable
            onDragStart={(e) => onDragStart(e, node, treeId)}
          >
            <GripVertical className="w-4 h-4" />
          </div>
          
          <Select
            value={node.type.startsWith('cases_') ? 'cases' : node.type}
            onValueChange={(value) => {
              if (value === 'leaf') {
                // Clean conversion - only keep necessary fields
                updateNode(treeId, node.id, {
                  id: node.id,
                  type: 'leaf',
                  result_idx: -1
                } as any);
              } else if (value === 'unary') {
                updateNode(treeId, node.id, {
                  id: node.id,
                  type: 'unary',
                  condition: {
                    type: 'unary',
                    op: '<=',
                    feature: '',
                    threshold: 0
                  },
                  then: undefined,
                  otherwise: undefined
                } as any);
              } else if (value === 'cases') {
                // Default to ranges when selecting "cases"
                updateNode(treeId, node.id, {
                  id: node.id,
                  type: 'cases',
                  op: 'ranges',
                  feature: '',
                  conditions: [],
                  branches: [],
                  otherwise: 0
                } as any);
              } else if (value === 'composite') {
                updateNode(treeId, node.id, {
                  id: node.id,
                  type: 'composite',
                  op: 'and',
                  conditions: [],
                  then: undefined,
                  otherwise: undefined
                } as any);
              }
            }}
          >
            <SelectTrigger className="w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="leaf">Leaf</SelectItem>
              <SelectItem value="unary">Unary</SelectItem>
              <SelectItem value="cases">Cases</SelectItem>
              <SelectItem value="composite">Composite</SelectItem>
            </SelectContent>
          </Select>

          {/* Cases operator selector */}
          {node.type === 'cases' && (
            <Select
              value={node.op}
              onValueChange={(value) => {
                const newOp = value as 'ranges' | 'isin' | 'string_match';
                const currentFeature = node.feature || '';
                if (newOp === 'ranges') {
                  updateNode(treeId, node.id, {
                    type: 'cases',
                    op: 'ranges',
                    feature: currentFeature,
                    conditions: [],
                    branches: [],
                    otherwise: 0
                  } as any);
                } else if (newOp === 'isin') {
                  updateNode(treeId, node.id, {
                    type: 'cases',
                    op: 'isin',
                    feature: currentFeature,
                    conditions: [],
                    branches: [],
                    otherwise: 0
                  } as any);
                } else if (newOp === 'string_match') {
                  updateNode(treeId, node.id, {
                    type: 'cases',
                    op: 'string_match',
                    feature: currentFeature,
                    conditions: [],
                    branches: [],
                    otherwise: 0
                  } as any);
                }
              }}
            >
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ranges">Ranges</SelectItem>
                <SelectItem value="isin">IsIn</SelectItem>
                <SelectItem value="string_match">String Match</SelectItem>
              </SelectContent>
            </Select>
          )}

          {/* Unary operator selector - show inline editor for condition */}
          {node.type === 'unary' && (
            <div className="flex-1 flex items-center gap-2">
              <UnaryOperatorSelector
                value={(node.condition as any).op || '<='}
                onChange={(value) => {
                  const newOp = value as any;
                  const currentFeature = (node.condition as any).feature || '';
                  let newCondition: any = { type: 'unary', op: newOp };

                  // Clear operator-specific fields when changing operators
                  if (['<=', '<', '==', '>', '>=', '!='].includes(newOp)) {
                    newCondition = { type: 'unary', op: newOp, feature: currentFeature, threshold: 0 };
                  } else if (newOp === 'between') {
                    newCondition = { type: 'unary', op: newOp, feature: currentFeature, min: null, max: null };
                  } else if (newOp === 'isin') {
                    newCondition = { type: 'unary', op: newOp, feature: currentFeature, values: [] };
                  } else if (newOp === 'string_match') {
                    newCondition = { type: 'unary', op: newOp, feature: currentFeature, patterns: [''], match_type: 'exact', case_sensitive: true, trim_whitespace: false };
                  } else if (newOp === 'is_null' || newOp === 'is_not_null' || newOp === 'is_true' || newOp === 'is_false') {
                    newCondition = { type: 'unary', op: newOp, feature: currentFeature };
                  }

                  updateNode(treeId, node.id, { condition: newCondition } as any);
                }}
              />
            </div>
          )}

          {/* Composite logic operator selector */}
          {node.type === 'composite' && (
            <CompositeLogicSelector
              value={node.op || 'and'}
              conditionCount={(node.conditions || []).length}
              onChange={(value) => {
                updateNode(treeId, node.id, {
                  op: value
                } as any);
              }}
            />
          )}

          {node.type === 'leaf' && (
            <div className="w-full">
              <FlatRuleLeafNodeEditor node={node} treeId={treeId} />
            </div>
          )}
        </div>

        {/* Right-side action buttons */}
        <div className="flex items-center gap-0.5">
          {showFocusButton && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                if (isFocused) {
                  setFocusedNode(null); // Exit focus mode
                } else {
                  setFocusedNode({ treeId, nodeId: node.id, depth: absoluteDepth });
                }
              }}
              className="p-1 h-6 w-6"
              title={isFocused ? 'Exit focus mode' : 'Focus on this subtree'}
            >
              {isFocused ? (
                <EyeOff className="w-3 h-3 text-blue-600" />
              ) : (
                <Eye className="w-3 h-3" />
              )}
            </Button>
          )}
          
          {hasChildren && canExpand && (
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleCollapse}
              className="p-1 h-6 w-6"
              title={isCollapsed ? 'Expand' : 'Collapse'}
            >
              {isCollapsed ? (
                <ChevronRight className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
            </Button>
          )}
          
          {hasChildren && needsFocus && !isCollapsed && (
            <div className="p-1 h-6 w-6 flex items-center justify-center">
              <span className="text-xs text-orange-600 font-bold" title="Too deep - use focus mode to expand">⚠</span>
            </div>
          )}
          
          {node.type !== 'leaf' && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => onAddBranch(treeId, node.id)}
                className="h-6 px-2 text-xs"
              >
                <Plus className="w-3 h-3 mr-1" />
                Branch
              </Button>
            </>
          )}
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onCopyNode(node)}
            title="Copy node"
            className="p-1 h-6 w-6"
          >
            <Copy className="w-3 h-3" />
          </Button>
          {copiedNode && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                // Use the in-memory copied node
                const nodeToUse = copiedNode;

                if (nodeToUse) {
                  // Deep copy the node with fresh IDs to avoid conflicts
                  const deepCopyWithNewIds = (originalNode: RuleTableRow): RuleTableRow => {
                    const base = {
                      ...originalNode,
                      id: crypto.randomUUID(),
                    };

                    // Handle different node types
                    if (originalNode.type === 'unary' || originalNode.type === 'composite') {
                      return {
                        ...base,
                        then: originalNode.then ? deepCopyWithNewIds(originalNode.then) : originalNode.then,
                        otherwise: originalNode.otherwise ? deepCopyWithNewIds(originalNode.otherwise) : originalNode.otherwise,
                      } as RuleTableRow;
                    }

                    if (originalNode.type === 'cases') {
                      return {
                        ...base,
                        branches: originalNode.branches?.map(branch => deepCopyWithNewIds(branch)),
                        otherwise: originalNode.otherwise ? deepCopyWithNewIds(originalNode.otherwise) : originalNode.otherwise,
                      } as RuleTableRow;
                    }

                    return base as RuleTableRow;
                  };

                  const nodeWithNewIds = deepCopyWithNewIds(nodeToUse);

                  // Paste the node by replacing this node
                  updateNode(treeId, node.id, nodeWithNewIds);
                }
              }}
              title="Paste here"
              className="p-1 h-6 w-6 bg-blue-50 border-blue-200 text-blue-700"
            >
              <Clipboard className="w-3 h-3" />
            </Button>
          )}
        </div>
      </div>

      {/* Inline condition editor for unary nodes (always visible, not hidden when collapsed) */}
      {node.type === 'unary' && (
        <div className="mt-2 p-2 rounded bg-white bg-opacity-60">
          <UnaryConditionForm
            condition={node.condition}
            onChange={(newCondition) => updateNode(treeId, node.id, { condition: newCondition } as any)}
            availableFeatures={availableFeatures}
            availableParameters={availableParameters}
            onManageParameters={onManageParameters}
          />
        </div>
      )}

      {/* Render content based on node type */}
      {!isCollapsed && canExpand && (() => {
        // UNARY NODES - binary branching with then/otherwise
        if (node.type === 'unary') {
          return (
            <>
              <div className="mt-2">
                {/* THEN branch */}
                <div className="mt-1">
                  {node.then ? (
                    <>
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-xs font-medium text-gray-600">THEN:</div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-600 p-1 h-5 w-5"
                          onClick={() => updateNode(treeId, node.id, { then: undefined } as any)}
                          title="Delete THEN branch"
                        >
                          <Trash2 className="w-2.5 h-2.5" />
                        </Button>
                      </div>
                      <NodeBlock
                        node={node.then}
                        treeId={treeId}
                        depth={depth + 1}
                        absoluteDepth={absoluteDepth + 1}
                        availableParameters={availableParameters}
                        onManageParameters={onManageParameters}
                        onDragStart={onDragStart}
                        onDragOver={onDragOver}
                        onDrop={onDrop}
                        onCopyNode={onCopyNode}
                        onDeleteBranch={onDeleteBranch}
                        onAddBranch={onAddBranch}
                        onAddOtherwiseBranch={onAddOtherwiseBranch}
                      />
                    </>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs border-dashed"
                      onClick={() => updateNode(treeId, node.id, { then: createDefaultLeaf() } as any)}
                    >
                      <Plus className="w-3 h-3 mr-1" />
                      Add THEN
                    </Button>
                  )}
                </div>

                {/* OTHERWISE branch */}
                <div className="mt-2">
                  {node.otherwise ? (
                    <>
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-xs font-medium text-gray-600">OTHERWISE:</div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-600 p-1 h-5 w-5"
                          onClick={() => updateNode(treeId, node.id, { otherwise: undefined } as any)}
                          title="Delete OTHERWISE branch"
                        >
                          <Trash2 className="w-2.5 h-2.5" />
                        </Button>
                      </div>
                      <NodeBlock
                        node={node.otherwise}
                        treeId={treeId}
                        depth={depth + 1}
                        absoluteDepth={absoluteDepth + 1}
                        availableParameters={availableParameters}
                        onManageParameters={onManageParameters}
                        onDragStart={onDragStart}
                        onDragOver={onDragOver}
                        onDrop={onDrop}
                        onCopyNode={onCopyNode}
                        onDeleteBranch={onDeleteBranch}
                        onAddBranch={onAddBranch}
                        onAddOtherwiseBranch={onAddOtherwiseBranch}
                      />
                    </>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs border-dashed"
                      onClick={() => updateNode(treeId, node.id, { otherwise: createDefaultLeaf() } as any)}
                    >
                      <Plus className="w-3 h-3 mr-1" />
                      Add OTHERWISE
                    </Button>
                  )}
                </div>
              </div>
            </>
          );
        }

        // CASES NODES - multi-way branching with conditions
        if (node.type === 'cases') {
          return (
            <>
              {/* Feature selector and string match settings at top */}
              <div className="mt-2 p-2 rounded bg-white bg-opacity-60 space-y-2">
                <div>
                  <label className="text-sm font-medium mb-2 block">Feature</label>
                  <ComputedFeatureInput
                    value={node.feature || ''}
                    onChange={(newFeature: string | { type: 'computed'; expression: string }) => updateNode(treeId, node.id, { feature: newFeature } as any)}
                    availableFeatures={availableFeatures}
                    availableParameters={availableParameters}
                  />
                </div>

                {/* String match settings (shared across all conditions) */}
                {node.type === 'cases' && node.op === 'string_match' && (
                  <>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-600 text-sm">Match Type:</span>
                      <Select
                        value={(node as any).match_type || 'exact'}
                        onValueChange={(value) => updateNode(treeId, node.id, { match_type: value } as any)}
                      >
                        <SelectTrigger className="w-32 h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="exact">Exact</SelectItem>
                          <SelectItem value="starts_with">Starts With</SelectItem>
                          <SelectItem value="contains">Contains</SelectItem>
                          <SelectItem value="ends_with">Ends With</SelectItem>
                          <SelectItem value="regex">Regex</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <Checkbox
                          checked={(node as any).case_sensitive ?? true}
                          onCheckedChange={(checked) => updateNode(treeId, node.id, { case_sensitive: !!checked } as any)}
                        />
                        <span className="text-sm text-gray-700">Case Sensitive</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <Checkbox
                          checked={(node as any).trim_whitespace ?? false}
                          onCheckedChange={(checked) => updateNode(treeId, node.id, { trim_whitespace: !!checked } as any)}
                        />
                        <span className="text-sm text-gray-700">Trim Whitespace</span>
                      </label>
                    </div>
                  </>
                )}
              </div>

              {/* Ranges-specific strict mode toggle */}
              {node.type === 'cases' && node.op === 'ranges' && (
                <div className="mt-2 p-2 rounded bg-white bg-opacity-60">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <Checkbox
                      checked={(node as any).strict ?? false}
                      onCheckedChange={(checked) => {
                        const isEnablingStrictMode = !!checked;

                        if (isEnablingStrictMode && node.conditions.length > 0) {
                          // Check if enabling would modify conditions
                          if (wouldStrictModeModify(node.conditions)) {
                            // Show confirmation dialog
                            setStrictModeConfirmOpen(true);
                          } else {
                            // No changes needed, just enable
                            applyStrictMode();
                          }
                        } else {
                          // Just toggle off strict mode, keep conditions as-is
                          updateNode(treeId, node.id, { strict: isEnablingStrictMode } as any);
                        }
                      }}
                    />
                    <span className="text-sm text-gray-700">Strict Mode (auto-infer lower bounds)</span>
                  </label>
                  <p className="text-xs text-gray-500 mt-1 ml-6">
                    {(node as any).strict
                      ? 'Only upper bounds needed. Lower bounds auto-filled from previous condition.'
                      : 'Manual mode: specify both min and max for each range.'}
                  </p>
                </div>
              )}

              {/* Conditions section */}
              <div className="mt-2 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-medium text-gray-600">CONDITIONS</div>
                </div>

                {node.type === 'cases' && node.op === 'ranges' && (
                  <>
                    {/* Table with integrated header */}
                    <div className="border border-gray-200 rounded-md divide-y divide-gray-200 overflow-hidden">
                      {/* Table header */}
                      <div className="grid grid-cols-[1fr_1fr_1fr_auto] gap-2 px-2 py-1.5 bg-white border-b border-gray-200">
                        <div className="text-xs font-medium text-gray-500">Lower</div>
                        <div className="text-xs font-medium text-gray-500">Upper</div>
                        <div className="text-xs font-medium text-gray-500">Then →</div>
                        <div className="w-6"></div>
                      </div>

                      {/* Table rows */}
                      <div className="divide-y divide-gray-200">
                      {node.conditions.map((condition, idx) => {
                        const isStrictMode = (node as any).strict ?? false;
                        const prevMax = idx > 0 ? (node.conditions[idx - 1].when as any).max : null;
                        const displayMin = isStrictMode ? prevMax : (condition.when as any).min;

                        return (
                          <div key={idx} className="grid grid-cols-[1fr_1fr_1fr_auto] gap-2 p-2 items-center bg-white hover:bg-gray-50">
                            {/* Lower bound */}
                            <div>
                              {isStrictMode ? (
                                <div className="text-xs text-gray-500 px-2 py-1 bg-gray-100 rounded h-8 flex items-center">
                                  {idx === 0 ? '-∞' : displayMin ?? '-∞'}
                                </div>
                              ) : (
                                <ParameterizedNumericInput
                                  value={(condition.when as any).min ?? null}
                                  onChange={(newValue) => {
                                    const newConditions = [...node.conditions];
                                    newConditions[idx] = { ...condition, when: { ...condition.when, min: newValue } };
                                    updateNode(treeId, node.id, { conditions: newConditions } as any);
                                  }}
                                  availableParameters={availableParameters}
                                  label=""
                                  placeholder="-∞"
                                  allowInfinity={true}
                                  allowNaN={true}
                                  onManageParameters={onManageParameters}
                                  showManageParameters={true}
                                  showLabel={false}
                                />
                              )}
                            </div>

                            {/* Upper bound */}
                            <div>
                              <ParameterizedNumericInput
                                value={(condition.when as any).max ?? null}
                                onChange={(newValue) => {
                                  const newConditions = [...node.conditions];
                                  newConditions[idx] = { ...condition, when: { ...condition.when, max: newValue } };
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                                availableParameters={availableParameters}
                                label=""
                                placeholder="∞"
                                allowInfinity={true}
                                allowNaN={true}
                                onManageParameters={onManageParameters}
                                showManageParameters={true}
                                showLabel={false}
                              />
                            </div>

                            {/* Then dropdown */}
                            <div>
                              <Select
                                value={String(condition.then)}
                                onValueChange={(value) => {
                                  const newConditions = [...node.conditions];
                                  newConditions[idx] = { ...condition, then: parseInt(value) };
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                              >
                                <SelectTrigger className="h-8">
                                  <SelectValue placeholder="Branch" />
                                </SelectTrigger>
                                <SelectContent>
                                  {node.branches.map((_, branchIdx) => (
                                    <SelectItem key={branchIdx} value={String(branchIdx)}>
                                      Branch {branchIdx}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>

                            {/* Remove button */}
                            <div>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-red-600 p-1 h-6 w-6"
                                onClick={() => {
                                  const newConditions = node.conditions.filter((_, i) => i !== idx);
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                                title="Delete condition"
                              >
                                <Trash2 className="w-3 h-3" />
                              </Button>
                            </div>
                          </div>
                        );
                      })}

                        {/* Add condition row */}
                        <div className="p-2 bg-gray-50">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-8 text-xs border-dashed w-full"
                            onClick={() => {
                              const isStrictMode = (node as any).strict ?? false;
                              const prevMax = node.conditions.length > 0
                                ? (node.conditions[node.conditions.length - 1].when as any).max
                                : null;

                              const newCondition = {
                                when: {
                                  min: isStrictMode ? prevMax : null,
                                  max: null
                                },
                                then: 0
                              };
                              const newConditions = [...node.conditions, newCondition];
                              updateNode(treeId, node.id, { conditions: newConditions } as any);
                            }}
                          >
                            <Plus className="w-3 h-3 mr-1" />
                            Add Condition
                          </Button>
                        </div>
                      </div>
                    </div>
                  </>
                )}

                {/* IsIn table */}
                {node.type === 'cases' && node.op === 'isin' && (
                  <>
                    <div className="border border-gray-200 rounded-md divide-y divide-gray-200 overflow-hidden">
                      {/* Table header */}
                      <div className="grid grid-cols-[2fr_1fr_auto] gap-2 px-2 py-1.5 bg-white border-b border-gray-200">
                        <div className="text-xs font-medium text-gray-500">Values</div>
                        <div className="text-xs font-medium text-gray-500">Then →</div>
                        <div className="w-6"></div>
                      </div>

                      {/* Table rows */}
                      <div className="divide-y divide-gray-200">
                        {node.conditions.map((condition, idx) => (
                          <div key={idx} className="grid grid-cols-[2fr_1fr_auto] gap-2 p-2 items-center bg-white hover:bg-gray-50">
                            {/* Values */}
                            <div>
                              <ParameterizedCategoricalInput
                                value={(condition.when as any).values || []}
                                onChange={(newValues) => {
                                  const newConditions = [...node.conditions];
                                  newConditions[idx] = { ...condition, when: { ...condition.when, values: newValues } };
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                                availableParameters={availableParameters}
                                label=""
                                placeholder="Type integers and press Enter"
                                onManageParameters={onManageParameters}
                                showManageParameters={true}
                                showLabel={false}
                              />
                            </div>

                            {/* Then dropdown */}
                            <div>
                              <Select
                                value={String(condition.then)}
                                onValueChange={(value) => {
                                  const newConditions = [...node.conditions];
                                  newConditions[idx] = { ...condition, then: parseInt(value) };
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                              >
                                <SelectTrigger className="h-8">
                                  <SelectValue placeholder="Branch" />
                                </SelectTrigger>
                                <SelectContent>
                                  {node.branches.map((_, branchIdx) => (
                                    <SelectItem key={branchIdx} value={String(branchIdx)}>
                                      Branch {branchIdx}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>

                            {/* Remove button */}
                            <div>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-red-600 p-1 h-6 w-6"
                                onClick={() => {
                                  const newConditions = node.conditions.filter((_, i) => i !== idx);
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                                title="Delete condition"
                              >
                                <Trash2 className="w-3 h-3" />
                              </Button>
                            </div>
                          </div>
                        ))}

                        {/* Add condition row */}
                        <div className="p-2 bg-gray-50">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-8 text-xs border-dashed w-full"
                            onClick={() => {
                              const newCondition = { when: { values: [] }, then: 0 };
                              const newConditions = [...node.conditions, newCondition];
                              updateNode(treeId, node.id, { conditions: newConditions } as any);
                            }}
                          >
                            <Plus className="w-3 h-3 mr-1" />
                            Add Condition
                          </Button>
                        </div>
                      </div>
                    </div>
                  </>
                )}

                {/* StringMatch table */}
                {node.type === 'cases' && node.op === 'string_match' && (
                  <>
                    <div className="border border-gray-200 rounded-md divide-y divide-gray-200 overflow-hidden">
                      {/* Table header */}
                      <div className="grid grid-cols-[2fr_1fr_auto] gap-2 px-2 py-1.5 bg-white border-b border-gray-200">
                        <div className="text-xs font-medium text-gray-500">Patterns (OR logic)</div>
                        <div className="text-xs font-medium text-gray-500">Then →</div>
                        <div className="w-6"></div>
                      </div>

                      {/* Table rows */}
                      <div className="divide-y divide-gray-200">
                        {node.conditions.map((condition, idx) => (
                          <div key={idx} className="grid grid-cols-[2fr_1fr_auto] gap-2 p-2 items-start bg-white hover:bg-gray-50">
                            {/* Patterns cell */}
                            <div className="space-y-1">
                              {((condition.when as any).patterns || ['']).map((pattern: string | any, patternIdx: number) => (
                                <div key={patternIdx} className="flex items-center gap-1">
                                  <ParameterizedStringInput
                                    value={pattern}
                                    onChange={(newValue) => {
                                      const newConditions = [...node.conditions];
                                      const currentPatterns = [...((condition.when as any).patterns || [])];
                                      currentPatterns[patternIdx] = newValue;
                                      newConditions[idx] = { ...condition, when: { ...condition.when, patterns: currentPatterns } };
                                      updateNode(treeId, node.id, { conditions: newConditions } as any);
                                    }}
                                    availableParameters={availableParameters}
                                    placeholder={`Pattern ${patternIdx + 1}`}
                                    onManageParameters={onManageParameters}
                                    showManageParameters={true}
                                    showLabel={false}
                                    className="flex-1"
                                  />

                                  {/* Remove pattern button */}
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="text-red-600 p-1 h-6 w-6 flex-shrink-0"
                                    onClick={() => {
                                      const newConditions = [...node.conditions];
                                      const currentPatterns = ((condition.when as any).patterns || []).filter((_: any, i: number) => i !== patternIdx);
                                      // Keep at least one pattern
                                      newConditions[idx] = {
                                        ...condition,
                                        when: {
                                          ...condition.when,
                                          patterns: currentPatterns.length > 0 ? currentPatterns : ['']
                                        }
                                      };
                                      updateNode(treeId, node.id, { conditions: newConditions } as any);
                                    }}
                                    title="Remove pattern"
                                  >
                                    <Trash2 className="w-3 h-3" />
                                  </Button>
                                </div>
                              ))}

                              {/* Add pattern button */}
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 text-xs text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                                onClick={() => {
                                  const currentPatterns = (condition.when as any).patterns || [];
                                  const newConditions = [...node.conditions];
                                  newConditions[idx] = { ...condition, when: { ...condition.when, patterns: [...currentPatterns, ''] } };
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                              >
                                <Plus className="w-3 h-3 mr-1" />
                                Add Pattern
                              </Button>
                            </div>

                            {/* Then dropdown */}
                            <div className="pt-1">
                              <Select
                                value={String(condition.then)}
                                onValueChange={(value) => {
                                  const newConditions = [...node.conditions];
                                  newConditions[idx] = { ...condition, then: parseInt(value) };
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                              >
                                <SelectTrigger className="h-8">
                                  <SelectValue placeholder="Branch" />
                                </SelectTrigger>
                                <SelectContent>
                                  {node.branches.map((_, branchIdx) => (
                                    <SelectItem key={branchIdx} value={String(branchIdx)}>
                                      Branch {branchIdx}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>

                            {/* Remove condition button */}
                            <div className="pt-1">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="text-red-600 p-1 h-6 w-6"
                                onClick={() => {
                                  const newConditions = node.conditions.filter((_, i) => i !== idx);
                                  updateNode(treeId, node.id, { conditions: newConditions } as any);
                                }}
                                title="Delete condition"
                              >
                                <Trash2 className="w-3 h-3" />
                              </Button>
                            </div>
                          </div>
                        ))}

                        {/* Add condition row */}
                        <div className="p-2 bg-gray-50">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-8 text-xs border-dashed w-full"
                            onClick={() => {
                              const newCondition = { when: { patterns: [''] }, then: 0 };
                              const newConditions = [...node.conditions, newCondition];
                              updateNode(treeId, node.id, { conditions: newConditions } as any);
                            }}
                          >
                            <Plus className="w-3 h-3 mr-1" />
                            Add Condition
                          </Button>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Branches section */}
              <div className="mt-3 space-y-2">
                <div className="text-xs font-medium text-gray-600">BRANCHES (Outputs)</div>
                {node.branches.map((branch, branchIdx) => (
                  <div key={branchIdx} className="border-l-2 border-gray-300 pl-2">
                    <div className="flex items-center justify-between mb-1">
                      <div className="text-xs font-medium text-gray-600">Branch {branchIdx}</div>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-red-600 p-1 h-5 w-5"
                        onClick={() => {
                          // Remove branch and update condition references
                          const newBranches = node.branches.filter((_, i) => i !== branchIdx);
                          const newConditions = node.conditions.map(cond => ({
                            ...cond,
                            then: cond.then > branchIdx ? cond.then - 1 : cond.then === branchIdx ? 0 : cond.then
                          }));
                          // Note: otherwise is a direct child node, not an index - no need to update it
                          updateNode(treeId, node.id, {
                            branches: newBranches,
                            conditions: newConditions
                          } as any);
                        }}
                        title="Delete branch"
                      >
                        <Trash2 className="w-2.5 h-2.5" />
                      </Button>
                    </div>
                    <NodeBlock
                      node={branch}
                      treeId={treeId}
                      depth={depth + 1}
                      absoluteDepth={absoluteDepth + 1}
                      availableParameters={availableParameters}
                      onManageParameters={onManageParameters}
                      onDragStart={onDragStart}
                      onDragOver={onDragOver}
                      onDrop={onDrop}
                      onCopyNode={onCopyNode}
                      onDeleteBranch={onDeleteBranch}
                      onAddBranch={onAddBranch}
                      onAddOtherwiseBranch={onAddOtherwiseBranch}
                      nodeErrors={nodeErrors}
                      basePath={[...basePath, 'branches', branchIdx]}
                    />
                  </div>
                ))}

                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs border-dashed w-full"
                  onClick={() => {
                    const newBranches = [...node.branches, createDefaultLeaf()];
                    updateNode(treeId, node.id, { branches: newBranches } as any);
                  }}
                >
                  <Plus className="w-3 h-3 mr-1" />
                  Add Branch
                </Button>
              </div>

              {/* OTHERWISE branch - handled by onAddOtherwiseBranch callback, rendered below */}
            </>
          );
        }

        // COMPOSITE NODES - AND/OR logic
        if (node.type === 'composite') {
          const conditions = node.conditions || [];

          return (
            <>
              <div className="mt-2">
                {/* Conditions list */}
                <div className="space-y-2">
                  <div className="text-xs font-medium text-gray-600">
                    CONDITIONS ({node.op?.toUpperCase()})
                  </div>

                  {conditions.length === 0 && (
                    <div className="text-xs text-red-600 p-2 bg-red-50 rounded">
                      A composite needs at least one condition. Add one below.
                    </div>
                  )}

                  <CompositeConditionEditor
                    conditions={conditions}
                    onUpdate={(newConditions) => {
                      updateNode(treeId, node.id, {
                        conditions: newConditions
                      } as any);
                    }}
                    availableFeatures={availableFeatures}
                    availableParameters={availableParameters}
                    onManageParameters={onManageParameters}
                    depth={0}
                    conditionErrors={nodeErrors}
                    basePath={[...basePath, 'conditions']}
                  />

                  {/* Add condition button - NOT can only have 1 condition */}
                  {(node.op !== 'not' || conditions.length === 0) && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs border-dashed w-full"
                      onClick={() => {
                        const newCondition = {
                          type: 'unary',
                          op: '<=',
                          feature: '',
                          threshold: 0
                        };
                        const newConditions = [...conditions, newCondition];
                        updateNode(treeId, node.id, {
                          conditions: newConditions
                        } as any);
                      }}
                    >
                      <Plus className="w-3 h-3 mr-1" />
                      Add Condition
                    </Button>
                  )}
                </div>

                {/* THEN branch */}
                <div className="mt-1">
                  {node.then ? (
                    <>
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-xs font-medium text-gray-600">THEN:</div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-600 p-1 h-5 w-5"
                          onClick={() => updateNode(treeId, node.id, { then: undefined } as any)}
                          title="Delete THEN branch"
                        >
                          <Trash2 className="w-2.5 h-2.5" />
                        </Button>
                      </div>
                      <NodeBlock
                        node={node.then}
                        treeId={treeId}
                        depth={depth + 1}
                        absoluteDepth={absoluteDepth + 1}
                        availableParameters={availableParameters}
                        onManageParameters={onManageParameters}
                        onDragStart={onDragStart}
                        onDragOver={onDragOver}
                        onDrop={onDrop}
                        onCopyNode={onCopyNode}
                        onDeleteBranch={onDeleteBranch}
                        onAddBranch={onAddBranch}
                        onAddOtherwiseBranch={onAddOtherwiseBranch}
                      />
                    </>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs border-dashed"
                      onClick={() => updateNode(treeId, node.id, { then: createDefaultLeaf() } as any)}
                    >
                      <Plus className="w-3 h-3 mr-1" />
                      Add THEN
                    </Button>
                  )}
                </div>

                {/* OTHERWISE branch */}
                <div className="mt-2">
                  {node.otherwise ? (
                    <>
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-xs font-medium text-gray-600">OTHERWISE:</div>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-600 p-1 h-5 w-5"
                          onClick={() => updateNode(treeId, node.id, { otherwise: undefined } as any)}
                          title="Delete OTHERWISE branch"
                        >
                          <Trash2 className="w-2.5 h-2.5" />
                        </Button>
                      </div>
                      <NodeBlock
                        node={node.otherwise}
                        treeId={treeId}
                        depth={depth + 1}
                        absoluteDepth={absoluteDepth + 1}
                        availableParameters={availableParameters}
                        onManageParameters={onManageParameters}
                        onDragStart={onDragStart}
                        onDragOver={onDragOver}
                        onDrop={onDrop}
                        onCopyNode={onCopyNode}
                        onDeleteBranch={onDeleteBranch}
                        onAddBranch={onAddBranch}
                        onAddOtherwiseBranch={onAddOtherwiseBranch}
                      />
                    </>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-8 text-xs border-dashed"
                      onClick={() => updateNode(treeId, node.id, { otherwise: createDefaultLeaf() } as any)}
                    >
                      <Plus className="w-3 h-3 mr-1" />
                      Add OTHERWISE
                    </Button>
                  )}
                </div>
              </div>
            </>
          );
        }

        // Unknown node type - render nothing
        return null;
      })()}

      {/* Strict Mode Confirmation Dialog */}
      <AlertDialog open={strictModeConfirmOpen} onOpenChange={setStrictModeConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Enable Strict Mode?</AlertDialogTitle>
            <AlertDialogDescription>
              This operation will modify your conditions:
              <ul className="list-disc list-inside mt-2 space-y-1">
                <li>All lower bounds will be cleared</li>
                <li>Conditions will be sorted by upper bound</li>
                <li>Lower bounds will be auto-inferred from previous conditions</li>
              </ul>
              <p className="mt-2 font-semibold">This action cannot be undone.</p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                applyStrictMode();
                setStrictModeConfirmOpen(false);
              }}
            >
              Enable Strict Mode
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};