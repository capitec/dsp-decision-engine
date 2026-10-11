import React, { useState, useRef, useEffect } from 'react';
import { isParamRef as isParameterReference, makeParamRef, paramRefName } from '@/utils/param-refs';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Settings } from "lucide-react";
import { ParameterOption } from '../shared-tree/ParameterizedNumericInput';
import type { ParameterReference } from '@/types/tree-types';

interface StringPatternInputProps {
  value: string | ParameterReference | null | undefined;
  onChange: (value: string | ParameterReference) => void;
  availableParameters?: ParameterOption[];
  onManageParameters?: () => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}


export const StringPatternInput: React.FC<StringPatternInputProps> = ({
  value,
  onChange,
  availableParameters = [],
  onManageParameters,
  placeholder = "pattern",
  disabled = false,
  className = "min-w-[120px] max-w-[250px]"
}) => {
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [autocompleteQuery, setAutocompleteQuery] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const autocompleteRef = useRef<HTMLDivElement>(null);

  const isParamMode = isParameterReference(value);
  const stringValue = isParamMode ? '' : (value || '');

  // For string patterns, allow all parameter types (they'll be coerced to strings)
  const stringParameters = availableParameters;

  const filteredParameters = stringParameters.filter(p =>
    p.name.toLowerCase().includes(autocompleteQuery.toLowerCase())
  );

  // Close autocomplete when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        autocompleteRef.current &&
        !autocompleteRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setShowAutocomplete(false);
      }
    };

    if (showAutocomplete) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showAutocomplete]);

  const handleParameterSelect = (paramName: string) => {
    onChange(makeParamRef(paramName));
    setShowAutocomplete(false);
    setAutocompleteQuery('');
  };

  const handleToggleMode = () => {
    if (isParamMode) {
      // Switch to direct value mode
      const param = stringParameters.find(p => p.name === paramRefName(value));
      onChange(param?.value || '');
    } else {
      // Switch to parameter mode
      if (stringParameters.length > 0) {
        setShowAutocomplete(true);
        setAutocompleteQuery('');
      }
    }
  };

  if (isParamMode) {
    return (
      <div className={`relative flex items-center gap-1 ${className}`}>
        <Input
          ref={inputRef}
          value={paramRefName(value) ?? ''}
          onChange={(e) => onChange(makeParamRef(e.target.value))}
          onFocus={() => {
            if (stringParameters.length > 0) {
              setShowAutocomplete(true);
            }
          }}
          placeholder="parameter_name"
          className="flex-1 h-8 font-mono"
          disabled={disabled}
        />
        <Button
          type="button"
          variant="default"
          size="sm"
          onClick={handleToggleMode}
          disabled={disabled}
          title="Switch to direct value"
          className="h-8 w-8 p-0 bg-blue-600 hover:bg-blue-700 text-white font-bold flex-shrink-0"
        >
          #
        </Button>

        {/* Autocomplete Dropdown for parameter mode */}
        {showAutocomplete && filteredParameters.length > 0 && (
          <div
            ref={autocompleteRef}
            className="absolute top-full left-0 mt-1 w-full bg-white border border-gray-300 rounded-md shadow-lg z-50 max-h-48 overflow-y-auto"
          >
            {filteredParameters.map((param) => (
              <button
                key={param.name}
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  handleParameterSelect(param.name);
                  inputRef.current?.blur();
                }}
                className="w-full px-3 py-2 text-left hover:bg-gray-100 focus:bg-gray-100 focus:outline-none"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm">{param.name}</span>
                  <span className="text-xs text-gray-500">{param.type}</span>
                </div>
                {param.value && (
                  <div className="text-xs text-gray-600 truncate mt-0.5">
                    {String(param.value)}
                  </div>
                )}
              </button>
            ))}
            {onManageParameters && (
              <button
                type="button"
                onMouseDown={(e) => {
                  e.preventDefault();
                  onManageParameters();
                  setShowAutocomplete(false);
                  inputRef.current?.blur();
                }}
                className="w-full px-3 py-2 text-left text-blue-600 hover:bg-blue-50 border-t flex items-center gap-2"
              >
                <Settings className="h-3 w-3" />
                <span className="text-sm">Manage Parameters</span>
              </button>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`relative flex items-center gap-1 ${className}`}>
      <Input
        ref={inputRef}
        value={stringValue}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className="flex-1"
        onFocus={() => {
          if (stringParameters.length > 0) {
            setShowAutocomplete(true);
          }
        }}
      />
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={handleToggleMode}
        disabled={disabled || stringParameters.length === 0}
        title="Use parameter reference"
        className="h-8 w-8 p-0 flex-shrink-0 border-gray-300 hover:border-blue-500 hover:text-blue-600"
      >
        #
      </Button>

      {/* Autocomplete Dropdown */}
      {showAutocomplete && filteredParameters.length > 0 && (
        <div
          ref={autocompleteRef}
          className="absolute top-full left-0 mt-1 w-full bg-white border border-gray-300 rounded-md shadow-lg z-50 max-h-48 overflow-y-auto"
        >
          {filteredParameters.map((param) => (
            <button
              key={param.name}
              type="button"
              onMouseDown={(e) => {
                e.preventDefault();
                handleParameterSelect(param.name);
                inputRef.current?.blur();
              }}
              className="w-full px-3 py-2 text-left hover:bg-gray-100 focus:bg-gray-100 focus:outline-none"
            >
              <div className="flex items-center justify-between">
                <span className="font-medium text-sm">{param.name}</span>
                <span className="text-xs text-gray-500">{param.type}</span>
              </div>
              {param.value && (
                <div className="text-xs text-gray-600 truncate mt-0.5">
                  {String(param.value)}
                </div>
              )}
            </button>
          ))}
          {onManageParameters && (
            <button
              type="button"
              onMouseDown={(e) => {
                e.preventDefault();
                onManageParameters();
                setShowAutocomplete(false);
                inputRef.current?.blur();
              }}
              className="w-full px-3 py-2 text-left text-blue-600 hover:bg-blue-50 border-t flex items-center gap-2"
            >
              <Settings className="h-3 w-3" />
              <span className="text-sm">Manage Parameters</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
};