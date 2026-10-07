import React, { useState, useRef, useEffect } from 'react';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { isParamRef, makeParamRef, paramRefName, type ParamRefValue } from "@/utils/param-refs";

/** Parameter reference stored in place of a string value */
export type InputRef = ParamRefValue;

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

interface ParameterizedStringInputProps {
  /** Current value: either a string or an InputRef */
  value: string | InputRef;
  /** Callback when value changes */
  onChange: (value: string | InputRef) => void;
  /** Available parameters for selection */
  availableParameters?: ParameterOption[];
  /** Label for the input */
  label?: string;
  /** Show label above input */
  showLabel?: boolean;
  /** Placeholder text */
  placeholder?: string;
  /** Disable the input */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Callback when "Manage Parameters" is clicked */
  onManageParameters?: () => void;
  /** Show "Manage Parameters" option in dropdown */
  showManageParameters?: boolean;
}

const isInputRef = (value: string | InputRef): value is InputRef => isParamRef(value);

/**
 * Shared string input component with parameter reference support.
 *
 * Features:
 * - Direct string value input
 * - # button to toggle between value and parameter mode
 * - Parameter input with autocomplete dropdown
 * - Smart filtering as user types
 *
 * Used in:
 * - Flat rules string_match patterns (with InputRef format)
 */
export const ParameterizedStringInput: React.FC<ParameterizedStringInputProps> = ({
  value,
  onChange,
  availableParameters = [],
  label,
  showLabel = true,
  placeholder = "",
  disabled = false,
  className = "",
  onManageParameters,
  showManageParameters = true
}) => {
  const isParameter = isInputRef(value);
  const parameterName = paramRefName(value) ?? '';
  const stringValue = isParameter ? '' : (value as string);

  const [isParameterMode, setIsParameterMode] = useState(isParameter);
  const [searchQuery, setSearchQuery] = useState(parameterName);
  const [showDropdown, setShowDropdown] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Filter parameters based on search query
  const filteredParameters = availableParameters.filter(param =>
    param.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    param.label?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const toggleParameterMode = () => {
    if (isParameterMode) {
      // Switch to direct string value
      onChange('');
      setIsParameterMode(false);
      setSearchQuery('');
    } else {
      // Switch to parameter mode
      onChange(makeParamRef(''));
      setIsParameterMode(true);
      setSearchQuery('');
      setShowDropdown(true);
    }
  };

  const handleParameterSelect = (paramName: string) => {
    onChange(makeParamRef(paramName));
    setSearchQuery(paramName);
    setShowDropdown(false);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (isParameterMode) {
      setSearchQuery(e.target.value);
      onChange(makeParamRef(e.target.value));
      setShowDropdown(true);
    } else {
      onChange(e.target.value);
    }
  };

  const handleInputFocus = () => {
    if (isParameterMode && availableParameters.length > 0) {
      setShowDropdown(true);
    }
  };

  return (
    <div className={`space-y-2 ${className}`}>
      {showLabel && label && (
        <Label className="text-sm font-medium text-gray-700">{label}</Label>
      )}

      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Input
            ref={inputRef}
            value={isParameterMode ? searchQuery : stringValue}
            onChange={handleInputChange}
            onFocus={handleInputFocus}
            placeholder={isParameterMode ? "parameter_name" : placeholder}
            disabled={disabled}
            className={`h-8 ${isParameterMode ? 'font-mono' : ''}`}
          />

          {/* Autocomplete dropdown */}
          {isParameterMode && showDropdown && (filteredParameters.length > 0 || showManageParameters) && (
            <div
              ref={dropdownRef}
              className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto"
            >
              {filteredParameters.length > 0 ? (
                <div className="py-1">
                  {filteredParameters.map((param) => (
                    <button
                      key={param.name}
                      type="button"
                      className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 focus:bg-gray-100 focus:outline-none"
                      onClick={() => handleParameterSelect(param.name)}
                    >
                      <div className="font-medium">{param.label || param.name}</div>
                      {param.type && (
                        <div className="text-xs text-gray-500">{param.type}</div>
                      )}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="px-3 py-2 text-sm text-gray-500 italic">
                  No parameters found
                </div>
              )}

              {/* Manage Parameters option */}
              {showManageParameters && onManageParameters && (
                <>
                  {filteredParameters.length > 0 && (
                    <div className="border-t border-gray-200" />
                  )}
                  <button
                    type="button"
                    className="w-full px-3 py-2 text-left text-sm text-blue-600 hover:bg-blue-50 focus:bg-blue-50 focus:outline-none font-medium"
                    onClick={() => {
                      onManageParameters();
                      setShowDropdown(false);
                    }}
                  >
                    + Manage Parameters
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {/* Parameter toggle button */}
        <Button
          type="button"
          variant={isParameterMode ? "default" : "outline"}
          size="sm"
          className={`h-8 w-8 p-0 text-xs flex-shrink-0 ${
            isParameterMode
              ? 'bg-blue-600 hover:bg-blue-700 text-white font-bold'
              : 'border-gray-300 hover:border-blue-500 hover:text-blue-600'
          }`}
          onClick={toggleParameterMode}
          disabled={disabled}
          title={isParameterMode ? 'Switch to direct value input' : 'Use parameter (values passed at tree execution)'}
        >
          #
        </Button>
      </div>
    </div>
  );
};
