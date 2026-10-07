import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FunctionSquare } from "lucide-react";
import { FeatureSelector } from './FeatureSelector';
import { ParameterOption } from './ParameterizedNumericInput';

interface ComputedFeatureInputProps {
  /** Current value: either a string feature name or computed feature object */
  value: string | { type: 'computed'; expression: string };
  /** Callback when value changes */
  onChange: (value: string | { type: 'computed'; expression: string }) => void;
  /** Available feature names for autocomplete */
  availableFeatures: string[];
  /** Available parameters for autocomplete */
  availableParameters?: ParameterOption[];
  /** Placeholder text */
  placeholder?: string;
  /** Additional CSS classes */
  className?: string;
}

// Available Polars functions for autocomplete
const POLARS_FUNCTIONS = [
  'duration',
  'datetime',
  'date',
  'time',
  'lit',
];

interface AutocompleteItem {
  type: 'feature' | 'parameter' | 'function';
  value: string;
  label: string;
  description?: string;
}

/**
 * Input for computed features with IDE-like autocomplete.
 *
 * Features:
 * - f(x) toggle button to switch between simple and computed mode
 * - Autocomplete for:
 *   - p.parameter_name (parameters)
 *   - feature names
 *   - Polars functions
 * - Syntax: "feature1 + feature2", "p.threshold * 2", "lit(5)"
 */
export const ComputedFeatureInput: React.FC<ComputedFeatureInputProps> = ({
  value,
  onChange,
  availableFeatures,
  availableParameters = [],
  placeholder = "Expression (e.g., feature1 + feature2)",
  className = ""
}) => {
  const isComputed = typeof value === 'object' && value.type === 'computed';
  const expression = isComputed ? value.expression : '';

  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [autocompleteIndex, setAutocompleteIndex] = useState(0);
  const [cursorPosition, setCursorPosition] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const autocompleteRef = useRef<HTMLDivElement>(null);

  // Get the current word being typed at cursor position
  const getCurrentWord = (text: string, position: number): { word: string; start: number; end: number } => {
    // Find word boundaries (alphanumeric + underscore + dot for parameters)
    let start = position;
    while (start > 0 && /[a-zA-Z0-9_.]/.test(text[start - 1])) {
      start--;
    }
    let end = position;
    while (end < text.length && /[a-zA-Z0-9_.]/.test(text[end])) {
      end++;
    }
    return { word: text.slice(start, end), start, end };
  };

  // Build autocomplete suggestions based on current word
  const suggestions = useMemo(() => {
    if (!showAutocomplete || !isComputed) return [];

    const { word } = getCurrentWord(expression, cursorPosition);
    const lowerWord = word.toLowerCase();
    const items: AutocompleteItem[] = [];

    // Check if typing parameter (p. prefix)
    if (word.startsWith('p.')) {
      const paramSearch = word.slice(2).toLowerCase();
      availableParameters.forEach(param => {
        if (param.name.toLowerCase().includes(paramSearch)) {
          items.push({
            type: 'parameter',
            value: `p.${param.name}`,
            label: `p.${param.name}`,
            description: `Parameter: ${param.type || 'any'}`
          });
        }
      });
    } else if (lowerWord === 'p') {
      // Show all parameters when typing "p"
      availableParameters.forEach(param => {
        items.push({
          type: 'parameter',
          value: `p.${param.name}`,
          label: `p.${param.name}`,
          description: `Parameter: ${param.type || 'any'}`
        });
      });
    } else {
      // Match features
      availableFeatures.forEach(feature => {
        if (feature.toLowerCase().includes(lowerWord)) {
          items.push({
            type: 'feature',
            value: feature,
            label: feature,
            description: 'Feature'
          });
        }
      });

      // Match functions
      POLARS_FUNCTIONS.forEach(func => {
        if (func.toLowerCase().includes(lowerWord)) {
          items.push({
            type: 'function',
            value: `${func}()`,
            label: func,
            description: 'Polars function'
          });
        }
      });
    }

    return items.slice(0, 10); // Limit to 10 suggestions
  }, [expression, cursorPosition, showAutocomplete, isComputed, availableFeatures, availableParameters]);

  // Handle autocomplete selection
  const selectSuggestion = (item: AutocompleteItem) => {
    const { word, start, end } = getCurrentWord(expression, cursorPosition);
    const before = expression.slice(0, start);
    const after = expression.slice(end);
    const newExpression = before + item.value + after;
    const newCursorPos = start + item.value.length;

    onChange({ type: 'computed', expression: newExpression });
    setShowAutocomplete(false);

    // Set cursor position after update
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus();
        inputRef.current.setSelectionRange(newCursorPos, newCursorPos);
      }
    }, 0);
  };

  // Keyboard navigation for autocomplete
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showAutocomplete || suggestions.length === 0) {
      // Show autocomplete on certain keys
      if (e.key === '.' || (e.key === ' ' && e.ctrlKey)) {
        setShowAutocomplete(true);
        setAutocompleteIndex(0);
        e.preventDefault();
        return;
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setAutocompleteIndex((prev) => Math.min(prev + 1, suggestions.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setAutocompleteIndex((prev) => Math.max(prev - 1, 0));
        break;
      case 'Enter':
      case 'Tab':
        e.preventDefault();
        if (suggestions[autocompleteIndex]) {
          selectSuggestion(suggestions[autocompleteIndex]);
        }
        break;
      case 'Escape':
        e.preventDefault();
        setShowAutocomplete(false);
        break;
    }
  };

  // Update cursor position on input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newExpression = e.target.value;
    const newCursorPos = e.target.selectionStart || 0;

    onChange({ type: 'computed', expression: newExpression });
    setCursorPosition(newCursorPos);

    // Show autocomplete if typing
    if (newExpression.length > 0) {
      setShowAutocomplete(true);
      setAutocompleteIndex(0);
    }
  };

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

  return (
    <div className={`relative flex items-center gap-1 ${className}`}>
      {isComputed ? (
        <>
          <Input
            ref={inputRef}
            type="text"
            value={expression}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onClick={(e) => {
              setCursorPosition(e.currentTarget.selectionStart || 0);
              setShowAutocomplete(true);
            }}
            placeholder={placeholder}
            className="flex-1 font-mono text-xs"
          />

          {/* Autocomplete dropdown */}
          {showAutocomplete && suggestions.length > 0 && (
            <div
              ref={autocompleteRef}
              className="absolute top-full left-0 mt-1 w-full bg-white border border-gray-300 rounded shadow-lg z-50 max-h-60 overflow-y-auto"
            >
              {suggestions.map((item, index) => (
                <div
                  key={`${item.type}-${item.value}`}
                  className={`px-3 py-2 cursor-pointer text-xs ${
                    index === autocompleteIndex ? 'bg-blue-100' : 'hover:bg-gray-100'
                  }`}
                  onClick={() => selectSuggestion(item)}
                  onMouseEnter={() => setAutocompleteIndex(index)}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono">{item.label}</span>
                    <span className="text-gray-500 text-[10px] ml-2">{item.description}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="flex-1">
          <FeatureSelector
            value={typeof value === 'string' ? value : ''}
            onChange={(featureName) => onChange(featureName)}
            availableFeatures={availableFeatures}
            label=""
            placeholder="Select or search feature..."
            allowCreate={true}
          />
        </div>
      )}

      {/* Toggle button */}
      <Button
        type="button"
        variant={isComputed ? "default" : "outline"}
        size="sm"
        className="h-8 px-2 shrink-0"
        onClick={() => {
          if (isComputed) {
            // Switch back to simple mode - use empty string
            onChange('');
          } else {
            // Switch to computed mode
            onChange({ type: 'computed', expression: '' });
          }
          setShowAutocomplete(false);
        }}
        title={isComputed ? "Switch to simple feature" : "Switch to computed feature"}
      >
        <FunctionSquare className="w-4 h-4" />
      </Button>
    </div>
  );
};
