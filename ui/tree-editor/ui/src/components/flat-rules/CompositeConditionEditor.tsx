import React from 'react';
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Trash2, Plus } from "lucide-react";
import { ParameterizedNumericInput } from '../shared-tree/ParameterizedNumericInput';
import { ParameterizedCategoricalInput } from '../shared-tree/ParameterizedCategoricalInput';
import { ParameterizedStringInput } from '../shared-tree/ParameterizedStringInput';
import { ComputedFeatureInput } from '../shared-tree/ComputedFeatureInput';
import { ParameterOption } from '../shared-tree/ParameterizedNumericInput';

interface CompositeConditionEditorProps {
  conditions: any[];
  onUpdate: (newConditions: any[]) => void;
  availableFeatures: string[];
  availableParameters: ParameterOption[];
  onManageParameters?: () => void;
  depth?: number;
  conditionErrors?: any[]; // Parsed validation errors for this condition group
  basePath?: (string | number)[]; // Path to this condition group
}

import { hasErrorAtPath, getErrorAtPath } from '@/utils/flat-rule-validation-parser';
import { AlertCircle } from "lucide-react";

export const CompositeConditionEditor: React.FC<CompositeConditionEditorProps> = ({
  conditions,
  onUpdate,
  availableFeatures,
  availableParameters,
  onManageParameters,
  depth = 0,
  conditionErrors = [],
  basePath = []
}) => {
  const borderColors = ['border-gray-300', 'border-indigo-300', 'border-purple-300', 'border-pink-300', 'border-blue-300'];
  const bgColors = ['bg-white', 'bg-gray-50', 'bg-indigo-50', 'bg-purple-50', 'bg-pink-50'];
  const borderColor = borderColors[Math.min(depth, borderColors.length - 1)];
  const bgColor = bgColors[Math.min(depth, bgColors.length - 1)];

  // Helper to check if condition is composite
  const isCompositeCondition = (cond: any) => {
    // Primary check: explicit type field
    if (cond.type === 'composite') return true;
    if (cond.type === 'unary') return false;

    // Fallback for legacy data: check if has conditions array with AND/OR/NOT op
    return cond.conditions && ['and', 'or', 'not'].includes(cond.op);
  };

  return (
    <>
      {conditions.map((cond: any, idx: number) => {
        const isComposite = isCompositeCondition(cond);

        // Build path to this condition
        const currentPath = [...basePath, idx];

        // Check for errors at this condition
        const hasError = hasErrorAtPath(conditionErrors, currentPath);
        const errorMessage = getErrorAtPath(conditionErrors, currentPath);

        return (
        <div key={idx} data-testid="composite-condition" className={`p-2 rounded ${bgColor} space-y-2 ${hasError ? 'ring-2 ring-red-500 bg-red-50' : ''}`}>
          {/* Error indicator */}
          {hasError && (
            <div className="flex items-center gap-1.5 text-red-600 text-xs bg-red-100 px-2 py-1 rounded">
              <AlertCircle className="w-3 h-3" />
              <span>{errorMessage || 'Validation error'}</span>
            </div>
          )}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-medium text-gray-600 text-sm">
                {depth > 0 ? `Nested ${idx + 1}` : `Condition ${idx + 1}`}
              </span>

              {/* Unified operator/logic selector */}
              <Select
                value={cond.op}
                onValueChange={(value) => {
                  const newConditions = [...conditions];
                  let updatedCond: any;

                  if (value === 'and' || value === 'or' || value === 'not') {
                    updatedCond = { type: 'composite', op: value, conditions: cond.conditions ?? [] };
                  } else {
                    const feature = cond.feature || '';
                    if (['<=', '<', '==', '>', '>=', '!='].includes(value)) {
                      updatedCond = { type: 'unary', op: value, feature, threshold: cond.threshold ?? 0 };
                    } else if (value === 'between') {
                      updatedCond = { type: 'unary', op: value, feature, min: null, max: null };
                    } else if (value === 'isin') {
                      updatedCond = { type: 'unary', op: value, feature, values: [] };
                    } else if (value === 'string_match') {
                      updatedCond = { type: 'unary', op: value, feature, patterns: [''], match_type: 'exact', case_sensitive: true, trim_whitespace: false };
                    } else if (value === 'is_true' || value === 'is_false') {
                      updatedCond = { type: 'unary', op: value, feature };
                    }
                  }

                  newConditions[idx] = updatedCond;
                  onUpdate(newConditions);
                }}
              >
                <SelectTrigger className="w-32 h-6 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="<=">≤</SelectItem>
                  <SelectItem value="<">&lt;</SelectItem>
                  <SelectItem value="==">==</SelectItem>
                  <SelectItem value=">">&gt;</SelectItem>
                  <SelectItem value=">=">≥</SelectItem>
                  <SelectItem value="!=">!=</SelectItem>
                  <SelectItem value="between">Between</SelectItem>
                  <SelectItem value="isin">IsIn</SelectItem>
                  <SelectItem value="string_match">String Match</SelectItem>
                  <SelectItem value="is_true">Is True</SelectItem>
                  <SelectItem value="is_false">Is False</SelectItem>
                  <SelectItem value="and">AND</SelectItem>
                  <SelectItem value="or">OR</SelectItem>
                  <SelectItem value="not">NOT</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              variant="ghost"
              size="sm"
              className="text-red-600 p-1 h-5 w-5"
              onClick={() => {
                const newConditions = conditions.filter((_, i) => i !== idx);
                onUpdate(newConditions);
              }}
              title="Delete condition"
            >
              <Trash2 className="w-2.5 h-2.5" />
            </Button>
          </div>

          {/* Unary condition editor */}
          {!isComposite && (
            <div className={`space-y-2 pl-2 border-l-2 ${borderColor}`}>
              <div className="flex items-center gap-2">
                <span className="font-medium text-gray-600 text-sm w-20">Feature:</span>
                <div className="flex-1">
                  <ComputedFeatureInput
                    value={cond.feature || ''}
                    onChange={(newFeature: string | { type: 'computed'; expression: string }) => {
                      const newConditions = [...conditions];
                      newConditions[idx] = { ...cond, feature: newFeature };
                      onUpdate(newConditions);
                    }}
                    availableFeatures={availableFeatures}
                  />
                </div>
              </div>

              {/* Threshold for comparison operators */}
              {['<=', '<', '==', '>', '>=', '!='].includes(cond.op) && (
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-600 text-sm">Threshold:</span>
                  <ParameterizedNumericInput
                    value={cond.threshold ?? 0}
                    onChange={(newValue) => {
                      const newConditions = [...conditions];
                      newConditions[idx] = { ...cond, threshold: newValue };
                      onUpdate(newConditions);
                    }}
                    availableParameters={availableParameters}
                    label=""
                    placeholder="0"
                    allowInfinity={true}
                    allowNaN={true}
                    step="0.01"
                    onManageParameters={onManageParameters}
                    showManageParameters={true}
                    showLabel={false}
                  />
                </div>
              )}

              {/* Between operator */}
              {cond.op === 'between' && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-600 text-sm w-20">Min:</span>
                    <ParameterizedNumericInput
                      value={cond.min ?? null}
                      onChange={(newValue) => {
                        const newConditions = [...conditions];
                        newConditions[idx] = { ...cond, min: newValue };
                        onUpdate(newConditions);
                      }}
                      availableParameters={availableParameters}
                      label=""
                      placeholder="null"
                      allowInfinity={true}
                      allowNaN={true}
                      onManageParameters={onManageParameters}
                      showManageParameters={true}
                      showLabel={false}
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-600 text-sm w-20">Max:</span>
                    <ParameterizedNumericInput
                      value={cond.max ?? null}
                      onChange={(newValue) => {
                        const newConditions = [...conditions];
                        newConditions[idx] = { ...cond, max: newValue };
                        onUpdate(newConditions);
                      }}
                      availableParameters={availableParameters}
                      label=""
                      placeholder="null"
                      allowInfinity={true}
                      allowNaN={true}
                      onManageParameters={onManageParameters}
                      showManageParameters={true}
                      showLabel={false}
                    />
                  </div>
                </div>
              )}

              {/* IsIn operator */}
              {cond.op === 'isin' && (
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-600 text-sm">Values:</span>
                  <ParameterizedCategoricalInput
                    value={cond.values || []}
                    onChange={(newValues) => {
                      const newConditions = [...conditions];
                      newConditions[idx] = { ...cond, values: newValues };
                      onUpdate(newConditions);
                    }}
                    availableParameters={availableParameters}
                    label=""
                    placeholder="Type integers and press Enter"
                    onManageParameters={onManageParameters}
                    showManageParameters={true}
                    showLabel={false}
                  />
                </div>
              )}

              {/* String match operator */}
              {cond.op === 'string_match' && (
                <div className="space-y-2">
                  <div className="space-y-1">
                    <span className="font-medium text-gray-600 text-sm">Patterns (OR logic)</span>
                    <div className="space-y-1">
                      {(cond.patterns || ['']).map((pattern: string | any, patternIdx: number) => (
                        <div key={patternIdx} className="flex items-center gap-2">
                          <ParameterizedStringInput
                            value={pattern}
                            onChange={(newValue) => {
                              const newConditions = [...conditions];
                              const currentPatterns = [...(cond.patterns || [])];
                              currentPatterns[patternIdx] = newValue;
                              newConditions[idx] = { ...cond, patterns: currentPatterns };
                              onUpdate(newConditions);
                            }}
                            availableParameters={availableParameters}
                            placeholder={`Pattern ${patternIdx + 1}`}
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
                              const newConditions = [...conditions];
                              const currentPatterns = (cond.patterns || []).filter((_: any, i: number) => i !== patternIdx);
                              newConditions[idx] = { ...cond, patterns: currentPatterns.length > 0 ? currentPatterns : [''] };
                              onUpdate(newConditions);
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
                        className="h-8 text-xs border-dashed w-full"
                        onClick={() => {
                          const newConditions = [...conditions];
                          const currentPatterns = cond.patterns || [];
                          newConditions[idx] = { ...cond, patterns: [...currentPatterns, ''] };
                          onUpdate(newConditions);
                        }}
                      >
                        <Plus className="w-3 h-3 mr-1" />
                        Add Pattern
                      </Button>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-600 text-sm">Match Type:</span>
                    <Select
                      value={cond.match_type || 'exact'}
                      onValueChange={(value) => {
                        const newConditions = [...conditions];
                        newConditions[idx] = { ...cond, match_type: value };
                        onUpdate(newConditions);
                      }}
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
                        checked={cond.case_sensitive ?? true}
                        onCheckedChange={(checked) => {
                          const newConditions = [...conditions];
                          newConditions[idx] = { ...cond, case_sensitive: !!checked };
                          onUpdate(newConditions);
                        }}
                      />
                      <span className="text-sm text-gray-700">Case Sensitive</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <Checkbox
                        checked={cond.trim_whitespace ?? false}
                        onCheckedChange={(checked) => {
                          const newConditions = [...conditions];
                          newConditions[idx] = { ...cond, trim_whitespace: !!checked };
                          onUpdate(newConditions);
                        }}
                      />
                      <span className="text-sm text-gray-700">Trim Whitespace</span>
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Composite condition - recursive! */}
          {isComposite && (
            <div className={`pl-2 border-l-2 ${borderColor} space-y-2`}>
              {(cond.conditions || []).length === 0 && (
                <div className="text-xs text-red-600 p-2 bg-red-50 rounded">
                  A composite needs at least one condition. Add one below.
                </div>
              )}

              {/* Recursive call! */}
              <CompositeConditionEditor
                conditions={cond.conditions || []}
                onUpdate={(nestedConditions) => {
                  const newConditions = [...conditions];
                  newConditions[idx] = { ...cond, conditions: nestedConditions };
                  onUpdate(newConditions);
                }}
                availableFeatures={availableFeatures}
                availableParameters={availableParameters}
                onManageParameters={onManageParameters}
                depth={depth + 1}
                conditionErrors={conditionErrors}
                basePath={[...currentPath, 'conditions']}
              />

              {/* Add nested condition button - NOT can only have 1 condition */}
              {(cond.op !== 'not' || (cond.conditions || []).length === 0) && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 text-xs border-dashed w-full"
                  onClick={() => {
                    const newConditions = [...conditions];
                    const nestedConditions = [...(cond.conditions || [])];
                    nestedConditions.push({
                      type: 'unary',
                      op: '<=',
                      feature: '',
                      threshold: 0
                    });
                    newConditions[idx] = { ...cond, conditions: nestedConditions };
                    onUpdate(newConditions);
                  }}
                >
                  <Plus className="w-3 h-3 mr-1" />
                  Add Nested Condition
                </Button>
              )}
            </div>
          )}
        </div>
        );
      })}
    </>
  );
};
