import React from 'react';
import { Button } from "@/components/ui/button";
import { Plus, X } from "lucide-react";
import { SingleConditionEditor } from './SingleConditionEditor';
import type { RuleCondition } from './types';
import { ParameterOption } from '../shared-tree/ParameterizedNumericInput';
import {
  useFlatRuleSetTrees
} from './useFlatRuleStore';

interface ConditionEditorProps {
  conditions: RuleCondition | RuleCondition[] | undefined;
  branchId: string;
  treeId: string;
  nodeType: string;
  availableParameters: ParameterOption[];
  onManageParameters?: () => void;
  strictMode?: boolean; // For range nodes - disables OR conditions
  previousMaxValue?: number | any | null; // For strict mode - the previous range's max value
  rangeConfig?: 'lower_inclusive' | 'upper_inclusive'; // For strict mode - operator configuration
  isFirstRange?: boolean; // For strict mode - is this the first range?
}

function createDefaultCondition(type: string): RuleCondition {
  switch (type) {
    case 'numerical':
      return { op: '<=', threshold: 0 };
    case 'categorical':
      return { categories: [] };
    case 'string':
      return { pattern: '', matchType: 'exact', caseSensitive: true };
    case 'ranges':
      return { min: null, max: null };
    default:
      return { op: '<=', threshold: 0 };
  }
}

export const ConditionEditor: React.FC<ConditionEditorProps> = ({
  conditions,
  branchId,
  treeId,
  nodeType,
  availableParameters,
  onManageParameters,
  strictMode = false,
  previousMaxValue,
  rangeConfig,
  isFirstRange
}) => {
  const setTrees = useFlatRuleSetTrees();

  const addConditionToGroup = (treeId: string, branchId: string, nodeType: string) => {
    setTrees(prev => prev.map(tree => {
      if (tree.id !== treeId) return tree;
      
      const updateNodeRecursively = (node: any): any => {
        if (node.branches) {
          const updatedBranches = node.branches.map((branch: any) => {
            if (branch.id === branchId && branch.when) {
              const currentCondition = branch.when;
              const newCondition = createDefaultCondition(nodeType);
              
              // Convert single condition to array or add to existing array
              const newWhen = Array.isArray(currentCondition)
                ? [...currentCondition, newCondition]
                : [currentCondition, newCondition];
              
              return { ...branch, when: newWhen };
            }
            return {
              ...branch,
              then: updateNodeRecursively(branch.then)
            };
          });
          
          return {
            ...node,
            branches: updatedBranches,
            otherwise: node.otherwise ? updateNodeRecursively(node.otherwise) : node.otherwise
          };
        }
        
        if (node.otherwise) {
          return {
            ...node,
            otherwise: updateNodeRecursively(node.otherwise)
          };
        }
        
        return node;
      };
      
      return {
        ...tree,
        rootNode: updateNodeRecursively(tree.rootNode)
      };
    }));
  };

  const removeConditionFromGroup = (treeId: string, branchId: string, conditionIndex: number) => {
    setTrees(prev => prev.map(tree => {
      if (tree.id !== treeId) return tree;
      
      const updateNodeRecursively = (node: any): any => {
        if (node.branches) {
          const updatedBranches = node.branches.map((branch: any) => {
            if (branch.id === branchId && branch.when && Array.isArray(branch.when)) {
              const updatedConditions = branch.when.filter((_: any, index: number) => index !== conditionIndex);
              
              // If only one condition left, unwrap from array
              const newWhen = updatedConditions.length === 1 ? updatedConditions[0] : updatedConditions;
              
              return { ...branch, when: newWhen };
            }
            return {
              ...branch,
              then: updateNodeRecursively(branch.then)
            };
          });
          
          return {
            ...node,
            branches: updatedBranches,
            otherwise: node.otherwise ? updateNodeRecursively(node.otherwise) : node.otherwise
          };
        }
        
        if (node.otherwise) {
          return {
            ...node,
            otherwise: updateNodeRecursively(node.otherwise)
          };
        }
        
        return node;
      };
      
      return {
        ...tree,
        rootNode: updateNodeRecursively(tree.rootNode)
      };
    }));
  };

  if (!conditions) return null;
  
  // Handle both single condition and array of conditions
  const conditionArray = Array.isArray(conditions) ? conditions : [conditions];
  
  return (
    <div className="space-y-1">
      {conditionArray.map((condition, index) => (
        <div key={`${branchId}-${index}`} className="flex items-center gap-2">
          {index > 0 && <span className="text-xs text-blue-600 font-medium">OR</span>}
          <SingleConditionEditor
            condition={condition}
            nodeType={nodeType as 'numerical' | 'categorical' | 'string' | 'ranges'}
            branchId={branchId}
            conditionIndex={index}
            treeId={treeId}
            availableParameters={availableParameters}
            onManageParameters={onManageParameters}
            strictMode={strictMode}
            previousMaxValue={previousMaxValue}
            rangeConfig={rangeConfig}
            isFirstRange={isFirstRange}
          />
          {conditionArray.length > 1 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => removeConditionFromGroup(treeId, branchId, index)}
              className="p-1 h-4 w-4 text-red-600"
              title="Remove this condition"
            >
              <X className="w-2 h-2" />
            </Button>
          )}
        </div>
      ))}
      
      {/* Add condition button - not for numerical nodes or strict mode range nodes */}
      {nodeType !== 'numerical' && !strictMode && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => addConditionToGroup(treeId, branchId, nodeType)}
          className="p-1 h-4 text-xs text-blue-600"
          title="Add OR condition"
        >
          <Plus className="w-2 h-2 mr-1" />
          Add OR
        </Button>
      )}
    </div>
  );
};