import React from 'react';
import { Button } from "@/components/ui/button";
import { Plus, Trash2 } from "lucide-react";
import { ParameterizedNumericInput } from './ParameterizedNumericInput';
import { ParameterizedCategoricalInput } from './ParameterizedCategoricalInput';
import { ParameterizedStringInput } from './ParameterizedStringInput';
import { ParameterOption } from './ParameterizedNumericInput';

interface CasesConditionEditorProps {
  caseType: 'cases_ranges' | 'cases_isin' | 'cases_string_match';
  conditions: any[];
  onUpdate: (newConditions: any[]) => void;
  availableParameters: ParameterOption[];
  onManageParameters?: () => void;
  strictMode?: boolean; // Only applicable to cases_ranges
}

/**
 * Editor for cases node conditions.
 * Each condition is a "when" - the "then" branches are handled by edges.
 */
export const CasesConditionEditor: React.FC<CasesConditionEditorProps> = ({
  caseType,
  conditions,
  onUpdate,
  availableParameters,
  onManageParameters,
  strictMode = false
}) => {
  const addCondition = () => {
    let newCondition: any;

    if (caseType === 'cases_ranges') {
      // In strict mode, auto-fill min from previous condition's max
      const prevMax = strictMode && conditions.length > 0
        ? conditions[conditions.length - 1].max
        : null;
      newCondition = { min: strictMode ? prevMax : null, max: null };
    } else if (caseType === 'cases_isin') {
      newCondition = { values: [] };
    } else if (caseType === 'cases_string_match') {
      newCondition = { patterns: [''] };
    }

    onUpdate([...conditions, newCondition]);
  };

  const removeCondition = (index: number) => {
    onUpdate(conditions.filter((_, i) => i !== index));
  };

  const updateCondition = (index: number, updates: any) => {
    const newConditions = [...conditions];
    newConditions[index] = { ...newConditions[index], ...updates };
    onUpdate(newConditions);
  };

  return (
    <div className="space-y-2">
      <div className="text-xs font-medium text-gray-600 mb-2">
        CONDITIONS (Branches via edges)
      </div>

      {conditions.length === 0 && (
        <div className="text-xs text-gray-500 italic p-2 bg-white border border-gray-200 rounded">
          No conditions added. Add a condition below.
        </div>
      )}

      {conditions.map((condition, index) => (
        <div key={index} className="p-2 rounded bg-white bg-opacity-60 space-y-2 border border-gray-200">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-medium text-gray-600">WHEN #{index}</span>
            <Button
              variant="ghost"
              size="sm"
              className="text-red-600 p-1 h-5 w-5"
              onClick={() => removeCondition(index)}
              title="Remove condition"
            >
              <Trash2 className="w-3 h-3" />
            </Button>
          </div>

          {/* Ranges editor */}
          {caseType === 'cases_ranges' && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="font-medium text-gray-600 text-sm w-16">Min:</span>
                {strictMode ? (
                  <div className="text-xs text-gray-500 px-2 py-1 bg-gray-100 rounded h-8 flex items-center flex-1">
                    {index === 0 ? '-∞' : (conditions[index - 1]?.max ?? '-∞')}
                  </div>
                ) : (
                  <ParameterizedNumericInput
                    value={condition.min ?? null}
                    onChange={(newValue) => updateCondition(index, { min: newValue })}
                    availableParameters={availableParameters}
                    label=""
                    placeholder="-∞"
                    allowInfinity={true}
                    allowNaN={true}
                    onManageParameters={onManageParameters}
                    showManageParameters={true}
                  />
                )}
              </div>
              <div className="flex items-center gap-2">
                <span className="font-medium text-gray-600 text-sm w-16">Max:</span>
                <ParameterizedNumericInput
                  value={condition.max ?? null}
                  onChange={(newValue) => updateCondition(index, { max: newValue })}
                  availableParameters={availableParameters}
                  label=""
                  placeholder="∞"
                  allowInfinity={true}
                  allowNaN={true}
                  onManageParameters={onManageParameters}
                  showManageParameters={true}
                />
              </div>
            </div>
          )}

          {/* IsIn editor */}
          {caseType === 'cases_isin' && (
            <div className="flex items-center gap-2">
              <span className="font-medium text-gray-600 text-sm w-16">Values:</span>
              <ParameterizedCategoricalInput
                value={condition.values || []}
                onChange={(newValues) => updateCondition(index, { values: newValues })}
                availableParameters={availableParameters}
                label=""
                placeholder="Type values and press Enter"
                onManageParameters={onManageParameters}
                showManageParameters={true}
              />
            </div>
          )}

          {/* String match editor */}
          {caseType === 'cases_string_match' && (
            <div className="space-y-1">
              <span className="font-medium text-gray-600 text-sm">Patterns:</span>
              {(condition.patterns || ['']).map((pattern: string | any, patIdx: number) => (
                <div key={patIdx} className="flex items-center gap-2">
                  <ParameterizedStringInput
                    value={pattern}
                    onChange={(newValue) => {
                      const currentPatterns = [...(condition.patterns || [])];
                      currentPatterns[patIdx] = newValue;
                      updateCondition(index, { patterns: currentPatterns });
                    }}
                    availableParameters={availableParameters}
                    placeholder={`Pattern ${patIdx + 1}`}
                    onManageParameters={onManageParameters}
                    showManageParameters={true}
                    showLabel={false}
                    className="flex-1"
                  />

                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-red-600 p-1 h-6 w-6"
                    onClick={() => {
                      const currentPatterns = (condition.patterns || []).filter((_: any, i: number) => i !== patIdx);
                      updateCondition(index, { patterns: currentPatterns.length > 0 ? currentPatterns : [''] });
                    }}
                    title="Remove pattern"
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              ))}

              <Button
                variant="outline"
                size="sm"
                className="h-6 text-xs border-dashed w-full"
                onClick={() => {
                  const currentPatterns = condition.patterns || [];
                  updateCondition(index, { patterns: [...currentPatterns, ''] });
                }}
              >
                <Plus className="w-3 h-3 mr-1" />
                Add Pattern
              </Button>
            </div>
          )}
        </div>
      ))}

      <Button
        variant="outline"
        size="sm"
        className="h-8 text-xs border-dashed w-full"
        onClick={addCondition}
      >
        <Plus className="w-3 h-3 mr-1" />
        Add Condition
      </Button>
    </div>
  );
};
