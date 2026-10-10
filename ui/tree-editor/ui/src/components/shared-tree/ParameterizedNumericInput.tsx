import React, { useState, useMemo, useRef, useEffect } from 'react';
import { isParamRef as isParameterReference, makeParamRef, paramRefName } from '@/utils/param-refs';
import { NumericInput } from "@/components/ui/numeric-input";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Settings } from "lucide-react";
import type { ParameterReference } from '@/types/tree-types';

/**
 * Available parameter option for dropdown
 */
export interface ParameterOption {
  /** Parameter identifier */
  name: string;
  /** Display label (optional, defaults to name) */
  label?: string;
  /** Parameter type for display */
  type?: string;
  /** Current/default value for display */
  value?: any;
}

interface ParameterizedNumericInputProps {
  /** Current value: either a number or a parameter reference */
  value: number | ParameterReference;
  /** Callback when value changes */
  onChange: (value: number | ParameterReference) => void;
  /** Available parameters/variables for selection */
  availableParameters?: ParameterOption[];
  /** Label for the input */
  label?: string;
  /** Show label above input */
  showLabel?: boolean;
  /** Placeholder for numeric input */
  placeholder?: string;
  /** Allow infinity values */
  allowInfinity?: boolean;
  /** Allow NaN values */
  allowNaN?: boolean;
  /** Step value for numeric input */
  step?: string;
  /** Disable the input */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when "Manage Parameters" is clicked */
  onManageParameters?: () => void;
  /** Show "Manage Parameters" option in dropdown/autocomplete */
  showManageParameters?: boolean;
}


function formatParameterValue(param: ParameterOption): string {
  if (Array.isArray(param.value)) {
    return `[${param.value.join(', ')}]`;
  }
  if (typeof param.value === 'object' && param.value !== null) {
    return JSON.stringify(param.value);
  }
  if (param.value !== undefined && param.value !== null) {
    return String(param.value);
  }
  return '';
}

/**
 * Shared numeric input component with parameter/variable reference support.
 *
 * Features:
 * - Direct numeric value input with NumericInput
 * - # button to toggle between value and parameter mode
 * - Parameter input with autocomplete
 * - Support for infinity and NaN values
 *
 * Used in:
 * - Tree editor numerical test nodes (with variable references)
 * - Flat rules numeric thresholds (with parameter references)
 */
export const ParameterizedNumericInput: React.FC<ParameterizedNumericInputProps> = ({
  value,
  onChange,
  availableParameters = [],
  label,
  showLabel = true,
  placeholder = "0",
  allowInfinity = false,
  allowNaN = false,
  step = "0.01",
  disabled = false,
  className = "",
  onManageParameters,
  showManageParameters = true
}) => {
  const isParameter = isParameterReference(value);
  const parameterName = paramRefName(value) ?? '';
  const numericValue = isParameter ? 0 : (value as number);

  // Autocomplete state
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [autocompleteIndex, setAutocompleteIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const autocompleteRef = useRef<HTMLDivElement>(null);

  // Filter parameters for autocomplete
  const filteredParameters = useMemo(() => {
    if (!isParameter) return [];
    const query = parameterName.toLowerCase();
    return availableParameters.filter(p =>
      p.name.toLowerCase().includes(query)
    );
  }, [isParameter, parameterName, availableParameters]);

  // Click outside to close autocomplete
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        autocompleteRef.current &&
        !autocompleteRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowAutocomplete(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleToggleMode = () => {
    if (isParameter) {
      // Switch to numeric mode - use default value
      onChange(0);
      setShowAutocomplete(false);
    } else {
      // Switch to parameter mode
      onChange(makeParamRef(''));
      setShowAutocomplete(true);
    }
  };

  const handleSelectParameter = (paramName: string) => {
    onChange(makeParamRef(paramName));
    setShowAutocomplete(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showAutocomplete || filteredParameters.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setAutocompleteIndex((prev) =>
          prev < filteredParameters.length + (showManageParameters && onManageParameters ? 0 : -1)
            ? prev + 1
            : 0
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setAutocompleteIndex((prev) =>
          prev > 0
            ? prev - 1
            : filteredParameters.length + (showManageParameters && onManageParameters ? 0 : -1)
        );
        break;
      case 'Enter':
        e.preventDefault();
        if (autocompleteIndex < filteredParameters.length) {
          handleSelectParameter(filteredParameters[autocompleteIndex].name);
        } else if (showManageParameters && onManageParameters) {
          onManageParameters();
          setShowAutocomplete(false);
        }
        break;
      case 'Escape':
        e.preventDefault();
        setShowAutocomplete(false);
        break;
    }
  };

  return (
    <div className={`space-y-2 ${className}`}>
      {showLabel && label && (
        <div className="flex items-center justify-between">
          <Label>{label}</Label>
        </div>
      )}

      <div className="flex items-center gap-2" data-testid="numeric-input">
        {isParameter ? (
          // Parameter reference mode
          <>
            {/* Freetext input with autocomplete for parameter name */}
            <div className="flex-1 relative">
                <Input
                  ref={inputRef}
                  value={parameterName}
                  onChange={(e) => {
                    onChange(makeParamRef(e.target.value));
                    setAutocompleteIndex(0);
                  }}
                  onKeyDown={handleKeyDown}
                  onFocus={() => {
                    setShowAutocomplete(true);
                  }}
                  placeholder="parameter_name"
                  className="h-8 font-mono"
                  disabled={disabled}
                />
                {/* Autocomplete dropdown */}
                {showAutocomplete && (filteredParameters.length > 0 || (showManageParameters && onManageParameters)) && (
                  <div
                    ref={autocompleteRef}
                    className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-md shadow-lg max-h-60 overflow-auto"
                  >
                    {filteredParameters.map((param, index) => (
                      <div
                        key={param.name}
                        className={`px-3 py-2 cursor-pointer hover:bg-gray-100 ${
                          index === autocompleteIndex ? 'bg-gray-100' : ''
                        }`}
                        onMouseDown={(e) => {
                          e.preventDefault();
                          handleSelectParameter(param.name);
                          inputRef.current?.blur();
                        }}
                      >
                        <div className="flex flex-col">
                          <span className="font-medium text-sm">{param.label || param.name}</span>
                          {param.type && (
                            <span className="text-xs text-muted-foreground">
                              {param.type}
                              {formatParameterValue(param) && `: ${formatParameterValue(param)}`}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                    {showManageParameters && onManageParameters && (
                      <>
                        {filteredParameters.length > 0 && <div className="h-px bg-gray-200 my-1" />}
                        <div
                          className={`px-3 py-2 cursor-pointer hover:bg-gray-100 text-blue-600 ${
                            autocompleteIndex === filteredParameters.length ? 'bg-gray-100' : ''
                          }`}
                          onMouseDown={(e) => {
                            e.preventDefault();
                            onManageParameters();
                            setShowAutocomplete(false);
                            inputRef.current?.blur();
                          }}
                        >
                          <div className="flex items-center gap-2">
                            <Settings className="h-3 w-3" />
                            <span className="text-sm font-medium">Manage Parameters</span>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            <Button
              type="button"
              variant="default"
              size="sm"
              onClick={handleToggleMode}
              disabled={disabled}
              title="Switch to direct value input"
              className="h-8 w-8 p-0 text-xs flex-shrink-0 bg-blue-600 hover:bg-blue-700 text-white font-bold"
            >
              #
            </Button>
          </>
        ) : (
          // Direct numeric value mode
          <>
            <NumericInput
              value={numericValue}
              onChange={(val) => onChange(typeof val === 'number' ? val : 0)}
              step={step}
              placeholder={placeholder}
              className="h-8 flex-1"
              allowEmpty={false}
              allowInfinity={allowInfinity}
              allowNaN={allowNaN}
              disabled={disabled}
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleToggleMode}
              disabled={disabled}
              title="Use parameter (values passed at tree execution)"
              className="h-8 w-8 p-0 text-xs flex-shrink-0 border-gray-300 hover:border-blue-500 hover:text-blue-600"
            >
              #
            </Button>
          </>
        )}
      </div>
    </div>
  );
};
