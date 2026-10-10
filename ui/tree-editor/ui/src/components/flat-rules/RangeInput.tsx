import React from 'react';
import { ParameterizedNumericInput, ParameterOption } from '../shared-tree/ParameterizedNumericInput';
import type { ParameterReference } from '@/types/tree-types';
import { boundLabel } from '@/utils/param-refs';

interface RangeInputProps {
  minValue: number | ParameterReference | null | undefined;
  maxValue: number | ParameterReference | null | undefined;
  onMinChange: (value: number | ParameterReference | null) => void;
  onMaxChange: (value: number | ParameterReference | null) => void;
  availableParameters?: ParameterOption[];
  onManageParameters?: () => void;
  disabled?: boolean;
  strictMode?: boolean; // When true, force min to null and only allow editing max
  previousMaxValue?: number | ParameterReference | null; // For strict mode - show the previous range's max
  rangeConfig?: 'lower_inclusive' | 'upper_inclusive'; // For strict mode - determine operator display
  isFirstRange?: boolean; // For strict mode - is this the first range?
}

export const RangeInput: React.FC<RangeInputProps> = ({
  minValue,
  maxValue,
  onMinChange,
  onMaxChange,
  availableParameters = [],
  onManageParameters,
  disabled = false,
  strictMode = false,
  previousMaxValue,
  rangeConfig = 'lower_inclusive',
  isFirstRange = false
}) => {
  // In strict mode, show relationship to previous range
  if (strictMode) {
    // Determine operators based on rangeConfig
    const lowerOp = rangeConfig === 'lower_inclusive' ? '≥' : '>';
    const upperOp = rangeConfig === 'lower_inclusive' ? '<' : '≤';

    // Format previous value for display
    const formatValue = (val: number | ParameterReference | null | undefined): string =>
      val === Infinity ? '+∞' : val === -Infinity ? '-∞' : boundLabel(val);

    return (
      <div className="flex items-center gap-2 flex-wrap">
        {!isFirstRange && previousMaxValue !== null && previousMaxValue !== undefined && (
          <>
            <span className="text-sm text-gray-600 whitespace-nowrap">{lowerOp}</span>
            <span className="text-sm font-medium text-gray-700 px-1">{formatValue(previousMaxValue)}</span>
            <span className="text-sm text-gray-600 whitespace-nowrap">and</span>
          </>
        )}
        <span className="text-sm text-gray-600 whitespace-nowrap">{upperOp}</span>
        <div className="min-w-[100px] max-w-[200px]">
          <ParameterizedNumericInput
            value={maxValue ?? 0}
            onChange={(value) => {
              // Ensure min stays null in strict mode
              onMaxChange(value);
              if (minValue !== null && minValue !== undefined) {
                onMinChange(null);
              }
            }}
            availableParameters={availableParameters}
            onManageParameters={onManageParameters}
            showManageParameters={true}
            placeholder="value"
            allowInfinity={true}
            showLabel={false}
            disabled={disabled}
            className="w-full"
          />
        </div>
      </div>
    );
  }

  // Normal mode - show both min and max
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-sm text-gray-600 whitespace-nowrap">between</span>
      <div className="min-w-[100px] max-w-[200px]">
        <ParameterizedNumericInput
          value={minValue ?? 0}
          onChange={onMinChange}
          availableParameters={availableParameters}
          onManageParameters={onManageParameters}
          showManageParameters={true}
          placeholder="-∞"
          allowInfinity={true}
          showLabel={false}
          disabled={disabled}
          className="w-full"
        />
      </div>
      <span className="text-sm text-gray-600 whitespace-nowrap px-1">and</span>
      <div className="min-w-[100px] max-w-[200px]">
        <ParameterizedNumericInput
          value={maxValue ?? 0}
          onChange={onMaxChange}
          availableParameters={availableParameters}
          onManageParameters={onManageParameters}
          showManageParameters={true}
          placeholder="+∞"
          allowInfinity={true}
          showLabel={false}
          disabled={disabled}
          className="w-full"
        />
      </div>
    </div>
  );
};