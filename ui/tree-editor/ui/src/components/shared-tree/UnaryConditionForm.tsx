import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Plus, Trash2 } from "lucide-react";
import { ParameterizedNumericInput } from './ParameterizedNumericInput';
import { ParameterizedCategoricalInput } from './ParameterizedCategoricalInput';
import { ParameterizedStringInput } from './ParameterizedStringInput';
import { ComputedFeatureInput } from './ComputedFeatureInput';
import { ParameterOption } from './ParameterizedNumericInput';
import type { UnaryCondition } from '@/components/flat-rules/types';

interface UnaryConditionFormProps {
  condition: UnaryCondition;
  onChange: (newCondition: UnaryCondition) => void;
  availableFeatures: string[];
  availableParameters: ParameterOption[];
  onManageParameters?: () => void;
}

/**
 * Shared form component for editing unary conditions.
 * Used by both flat rules editor and tree sidebar editor.
 *
 * Handles all 13 unary operators:
 * - Comparison: <=, <, ==, >, >=, !=
 * - Range: between
 * - Membership: isin
 * - String: string_match
 * - Boolean checks: is_true, is_false
 */
export const UnaryConditionForm: React.FC<UnaryConditionFormProps> = ({
  condition,
  onChange,
  availableFeatures,
  availableParameters,
  onManageParameters
}) => {
  // Extract feature value (handle both string and computed object format from backend)
  const featureValue = typeof condition.feature === 'string'
    ? condition.feature
    : (condition.feature as any);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="font-medium text-gray-600 text-sm w-20">Feature:</span>
        <div className="flex-1">
          <ComputedFeatureInput
            value={featureValue}
            onChange={(value) => onChange({ ...condition, feature: value as any })}
            availableFeatures={availableFeatures}
            placeholder="feature or expression (e.g., feature1 + feature2)"
          />
        </div>
      </div>

      {/* Threshold for comparison operators */}
      {['<=', '<', '==', '>', '>=', '!='].includes(condition.op) && (
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-600 text-sm">Threshold:</span>
          <ParameterizedNumericInput
            value={(condition as any).threshold ?? 0}
            onChange={(newValue) => onChange({ ...condition, threshold: newValue } as any)}
            availableParameters={availableParameters}
            label=""
            placeholder="0"
            allowInfinity={true}
            allowNaN={true}
            step="0.01"
            onManageParameters={onManageParameters}
            showManageParameters={true}
          />
        </div>
      )}

      {/* Between operator */}
      {condition.op === 'between' && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-600 text-sm w-20">Min:</span>
            <ParameterizedNumericInput
              value={(condition as any).min ?? null}
              allowEmpty
              onChange={(newValue) => onChange({ ...condition, min: newValue } as any)}
              availableParameters={availableParameters}
              label=""
              placeholder="null"
              allowInfinity={true}
              allowNaN={true}
              onManageParameters={onManageParameters}
              showManageParameters={true}
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium text-gray-600 text-sm w-20">Max:</span>
            <ParameterizedNumericInput
              value={(condition as any).max ?? null}
              allowEmpty
              onChange={(newValue) => onChange({ ...condition, max: newValue } as any)}
              availableParameters={availableParameters}
              label=""
              placeholder="null"
              allowInfinity={true}
              allowNaN={true}
              onManageParameters={onManageParameters}
              showManageParameters={true}
            />
          </div>
        </div>
      )}

      {/* IsIn operator */}
      {condition.op === 'isin' && (
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-600 text-sm">Values:</span>
          <ParameterizedCategoricalInput
            value={(condition as any).values || []}
            onChange={(newValues) => onChange({ ...condition, values: newValues } as any)}
            availableParameters={availableParameters}
            label=""
            placeholder="Type numbers and press Enter"
            onManageParameters={onManageParameters}
            showManageParameters={true}
          />
        </div>
      )}

      {/* String match operator */}
      {condition.op === 'string_match' && (
        <div className="space-y-2">
          {/* Patterns list with individual parameter support */}
          <div className="space-y-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-medium text-gray-600 text-sm">Patterns (OR logic)</span>
            </div>

            <div className="space-y-1">
              {((condition as any).patterns || ['']).map((pattern: string | any, idx: number) => (
                <div key={idx} className="flex items-center gap-2">
                  <ParameterizedStringInput
                    value={pattern}
                    onChange={(newValue) => {
                      const currentPatterns = [...((condition as any).patterns || [])];
                      currentPatterns[idx] = newValue;
                      onChange({ ...condition, patterns: currentPatterns } as any);
                    }}
                    availableParameters={availableParameters}
                    placeholder={`Pattern ${idx + 1}`}
                    onManageParameters={onManageParameters}
                    showManageParameters={true}
                    showLabel={false}
                    className="flex-1"
                  />

                  {/* Remove button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-red-600 p-1 h-6 w-6"
                    onClick={() => {
                      const currentPatterns = ((condition as any).patterns || []).filter((_: any, i: number) => i !== idx);
                      // Keep at least one pattern
                      onChange({ ...condition, patterns: currentPatterns.length > 0 ? currentPatterns : [''] } as any);
                    }}
                    title="Remove pattern"
                  >
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              ))}

              {/* Add Pattern button inline */}
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs border-dashed w-full"
                onClick={() => {
                  const currentPatterns = (condition as any).patterns || [];
                  onChange({ ...condition, patterns: [...currentPatterns, ''] } as any);
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
              value={(condition as any).match_type || 'exact'}
              onValueChange={(value) => onChange({ ...condition, match_type: value } as any)}
            >
              <SelectTrigger className="w-32 h-8" data-testid="match-type">
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
                checked={(condition as any).case_sensitive ?? true}
                onCheckedChange={(checked) => onChange({ ...condition, case_sensitive: !!checked } as any)}
              />
              <span className="text-sm text-gray-700">Case Sensitive</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={(condition as any).trim_whitespace ?? false}
                onCheckedChange={(checked) => onChange({ ...condition, trim_whitespace: !!checked } as any)}
              />
              <span className="text-sm text-gray-700">Trim Whitespace</span>
            </label>
          </div>
        </div>
      )}
    </div>
  );
};
