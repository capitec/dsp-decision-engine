import React, {useCallback} from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ComparisonOperatorSelector, ComparisonOperator } from '../shared-tree/ComparisonOperatorSelector';
import { ParameterizedNumericInput } from '../shared-tree/ParameterizedNumericInput';
import { ParameterizedCategoricalInput } from '../shared-tree/ParameterizedCategoricalInput';
import { ParameterOption } from '../shared-tree/ParameterizedNumericInput';
import { RangeInput } from './RangeInput';
import { StringPatternInput } from './StringPatternInput';
import type {
  RuleCondition,
  NumericalCondition,
  CategoricalCondition,
  StringCondition,
  RangeCondition
} from './types';
import { useFlatRuleUpdateConditionInGroup } from './useFlatRuleStore';

interface SingleConditionEditorProps {
  condition: RuleCondition;
  nodeType: 'numerical' | 'categorical' | 'string' | 'ranges'; // The parent node's type determines condition type
  branchId: string;
  conditionIndex: number;
  treeId: string;
  availableParameters?: ParameterOption[];
  onManageParameters?: () => void;
  strictMode?: boolean; // For range conditions in strict mode
  previousMaxValue?: number | any | null; // For strict mode - the previous range's max value
  rangeConfig?: 'lower_inclusive' | 'upper_inclusive'; // For strict mode - operator configuration
  isFirstRange?: boolean; // For strict mode - is this the first range?
}

export const SingleConditionEditor: React.FC<SingleConditionEditorProps> = ({
  condition,
  nodeType,
  branchId,
  conditionIndex,
  treeId,
  availableParameters = [],
  onManageParameters,
  strictMode = false,
  previousMaxValue,
  rangeConfig,
  isFirstRange
}) => {
  const updateConditionInGroup = useFlatRuleUpdateConditionInGroup();

  const changeThresholdCallback = useCallback((threshold: number | any) => {
      console.log("Updating threshold to", threshold);
      updateConditionInGroup(treeId, branchId, conditionIndex, { threshold });
    }, [treeId, branchId, conditionIndex, updateConditionInGroup]);

  const changeOpCallback = useCallback((op: ComparisonOperator) => {
    updateConditionInGroup(treeId, branchId, conditionIndex, { op });
  }, [treeId, branchId, conditionIndex, updateConditionInGroup]);

  switch (nodeType) {
    case 'numerical': {
      const numericalCondition = condition as NumericalCondition;
      return (
        <div className="flex items-center gap-2">
          <ComparisonOperatorSelector
            value={numericalCondition.op || "<="}
            onChange={changeOpCallback}
            showLabel={false}
            compact={true}
            className="w-auto"
          />
          <ParameterizedNumericInput
            key={`threshold-${treeId}-${branchId}-${conditionIndex}`}
            value={typeof numericalCondition.threshold === 'string' ? parseFloat(numericalCondition.threshold) || 0 : numericalCondition.threshold ?? 0}
            onChange={changeThresholdCallback}
            availableParameters={availableParameters}
            onManageParameters={onManageParameters}
            showManageParameters={true}
            placeholder="0"
            showLabel={false}
            className="flex-1"
          />
        </div>
      );
    }

    case 'categorical': {
      const categoricalCondition = condition as CategoricalCondition;
      return (
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">is in</span>
          <ParameterizedCategoricalInput
            value={categoricalCondition.categories ?? []}
            onChange={(categories) => updateConditionInGroup(treeId, branchId, conditionIndex, { categories })}
            availableParameters={availableParameters}
            onManageParameters={onManageParameters}
            showManageParameters={true}
            placeholder="Add integers..."
            showLabel={false}
            className="flex-1"
          />
        </div>
      );
    }

    case 'string': {
      const stringCondition = condition as StringCondition;
      return (
        <div className="flex items-center gap-2">
          <Select value={stringCondition.matchType} onValueChange={(value) => updateConditionInGroup(treeId, branchId, conditionIndex, { matchType: value as any })}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="exact">exact</SelectItem>
              <SelectItem value="contains">contains</SelectItem>
              <SelectItem value="starts_with">starts</SelectItem>
              <SelectItem value="ends_with">ends</SelectItem>
              <SelectItem value="regex">regex</SelectItem>
            </SelectContent>
          </Select>
          <StringPatternInput
            value={stringCondition.pattern as any}
            onChange={(pattern) => updateConditionInGroup(treeId, branchId, conditionIndex, { pattern })}
            availableParameters={availableParameters}
            onManageParameters={onManageParameters}
            placeholder="pattern"
          />
        </div>
      );
    }

    case 'ranges': {
      const rangeCondition = condition as RangeCondition;
      return (
        <RangeInput
          minValue={rangeCondition.min as any}
          maxValue={rangeCondition.max as any}
          onMinChange={(min) => updateConditionInGroup(treeId, branchId, conditionIndex, { min })}
          onMaxChange={(max) => updateConditionInGroup(treeId, branchId, conditionIndex, { max })}
          availableParameters={availableParameters}
          onManageParameters={onManageParameters}
          strictMode={strictMode}
          previousMaxValue={previousMaxValue}
          rangeConfig={rangeConfig}
          isFirstRange={isFirstRange}
        />
      );
    }
    
    default:
      return <span className="text-gray-500">Unknown condition</span>;
  }
};