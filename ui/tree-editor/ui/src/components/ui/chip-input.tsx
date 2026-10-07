import React, { useState, useCallback, KeyboardEvent, useRef, useEffect } from 'react';
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChipInputProps {
  values: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  validator?: (value: string) => boolean;
  formatter?: (value: string) => string;
  suggestions?: string[]; // Autocomplete suggestions
  maxItems?: number; // Maximum number of items allowed
  allowDuplicates?: boolean; // If true, don't filter out selected values from suggestions
}

export const ChipInput: React.FC<ChipInputProps> = ({
  values = [],
  onChange,
  placeholder = "Type and press Enter to add...",
  className,
  disabled = false,
  validator = () => true,
  formatter = (v) => v,
  suggestions = [],
  maxItems,
  allowDuplicates = false,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [focused, setFocused] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter suggestions based on input and optionally already selected values
  const filteredSuggestions = suggestions.filter(
    s => {
      const matchesSearch = s.toLowerCase().includes(inputValue.toLowerCase());
      const notAlreadySelected = allowDuplicates || !values.includes(s);
      return matchesSearch && notAlreadySelected;
    }
  );

  // Check if max items reached
  const isMaxReached = maxItems !== undefined && values.length >= maxItems;

  const addValue = useCallback((value: string) => {
    const trimmedValue = value.trim();
    if (!trimmedValue) return;

    // Check max items
    if (isMaxReached) return;

    const formattedValue = formatter(trimmedValue);

    // Check if value already exists (only if duplicates not allowed)
    if (!allowDuplicates && values.includes(formattedValue)) return;

    // Validate the value
    if (!validator(formattedValue)) return;

    onChange([...values, formattedValue]);
    setInputValue('');
    setShowSuggestions(false);
    setSelectedSuggestionIndex(-1);
  }, [values, onChange, validator, formatter, isMaxReached, allowDuplicates]);

  const removeValue = useCallback((index: number) => {
    const newValues = values.filter((_, i) => i !== index);
    onChange(newValues);
  }, [values, onChange]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLInputElement>) => {
    // Navigation in suggestions
    if (showSuggestions && filteredSuggestions.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedSuggestionIndex((prev) =>
          prev < filteredSuggestions.length - 1 ? prev + 1 : prev
        );
        return;
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedSuggestionIndex((prev) => (prev > 0 ? prev - 1 : -1));
        return;
      } else if (e.key === 'Enter' && selectedSuggestionIndex >= 0) {
        e.preventDefault();
        addValue(filteredSuggestions[selectedSuggestionIndex]);
        return;
      } else if (e.key === 'Escape') {
        setShowSuggestions(false);
        setSelectedSuggestionIndex(-1);
        return;
      }
    }

    if (e.key === 'Enter') {
      e.preventDefault();
      addValue(inputValue);
    } else if (e.key === 'Backspace' && !inputValue && values.length > 0) {
      // Remove last chip if input is empty and backspace is pressed
      removeValue(values.length - 1);
    }
  }, [inputValue, addValue, removeValue, values.length, showSuggestions, filteredSuggestions, selectedSuggestionIndex]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setInputValue(newValue);

    // Show suggestions if there are any and input is not empty
    if (suggestions.length > 0 && newValue.trim()) {
      setShowSuggestions(true);
      setSelectedSuggestionIndex(-1);
    } else {
      setShowSuggestions(false);
    }
  }, [suggestions]);

  const handleContainerClick = useCallback(() => {
    if (!disabled && !isMaxReached) {
      inputRef.current?.focus();
    }
  }, [disabled, isMaxReached]);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    addValue(suggestion);
    inputRef.current?.focus();
  }, [addValue]);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div ref={containerRef} className="relative w-full">
      <div
        className={cn(
          "flex min-h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm cursor-text",
          "ring-offset-background transition-colors",
          focused && "ring-2 ring-ring ring-offset-2",
          disabled && "cursor-not-allowed opacity-50",
          isMaxReached && "opacity-70",
          className
        )}
        onClick={handleContainerClick}
      >
        <div className="flex flex-wrap gap-1 flex-1 items-center">
          {values.map((value, index) => (
            <Badge
              key={index}
              variant="secondary"
              className="text-xs flex items-center gap-1 px-2 py-1"
            >
              <span>{value}</span>
              {!disabled && (
                <X
                  className="h-3 w-3 cursor-pointer hover:text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeValue(index);
                  }}
                />
              )}
            </Badge>
          ))}
          {!isMaxReached && (
            <Input
              ref={inputRef}
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              onFocus={() => {
                setFocused(true);
                if (inputValue.trim() && suggestions.length > 0) {
                  setShowSuggestions(true);
                }
              }}
              onBlur={() => {
                setFocused(false);
                // Delay hiding suggestions to allow click events
                setTimeout(() => setShowSuggestions(false), 200);
              }}
              placeholder={values.length === 0 ? placeholder : ""}
              disabled={disabled}
              className="border-0 bg-transparent p-0 h-auto focus-visible:ring-0 focus-visible:ring-offset-0 flex-1 min-w-24"
            />
          )}
          {isMaxReached && maxItems && (
            <span className="text-xs text-muted-foreground italic">Max {maxItems} items</span>
          )}
        </div>
      </div>

      {/* Suggestions dropdown */}
      {showSuggestions && filteredSuggestions.length > 0 && (
        <div className="absolute z-50 w-full mt-1 max-h-60 overflow-auto rounded-md border bg-popover shadow-md">
          {filteredSuggestions.map((suggestion, index) => (
            <div
              key={suggestion}
              className={cn(
                "px-3 py-2 text-sm cursor-pointer hover:bg-accent",
                index === selectedSuggestionIndex && "bg-accent"
              )}
              onMouseDown={(e) => {
                e.preventDefault(); // Prevent input blur
                handleSuggestionClick(suggestion);
              }}
            >
              {suggestion}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};