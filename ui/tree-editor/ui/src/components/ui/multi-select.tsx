import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { X, Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface MultiSelectProps {
  options: string[];
  values: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  maxItems?: number;
  emptyMessage?: string;
}

/**
 * Multi-select dropdown with search functionality.
 * Shows all options in a dropdown with checkboxes and search filter.
 * Used for selecting multiple values from a fixed set (e.g., categorical types).
 */
export const MultiSelect: React.FC<MultiSelectProps> = ({
  options,
  values = [],
  onChange,
  placeholder = "Select items...",
  className,
  disabled = false,
  maxItems,
  emptyMessage = "No options available",
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const isMaxReached = maxItems !== undefined && values.length >= maxItems;

  // Filter options based on search and exclude already selected
  const filteredOptions = options.filter(
    option => option.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleOption = useCallback((option: string) => {
    if (values.includes(option)) {
      // Remove
      onChange(values.filter(v => v !== option));
    } else {
      // Add (if not at max)
      if (!isMaxReached) {
        onChange([...values, option]);
      }
    }
  }, [values, onChange, isMaxReached]);

  const removeValue = useCallback((option: string) => {
    onChange(values.filter(v => v !== option));
  }, [values, onChange]);

  const handleToggleDropdown = useCallback(() => {
    if (!disabled) {
      setIsOpen(prev => !prev);
      if (!isOpen) {
        // Focus search when opening
        setTimeout(() => searchInputRef.current?.focus(), 100);
      }
    }
  }, [disabled, isOpen]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setSearchQuery('');
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      {/* Selected values display */}
      <div
        className={cn(
          "flex min-h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm cursor-pointer",
          "ring-offset-background transition-colors",
          isOpen && "ring-2 ring-ring ring-offset-2",
          disabled && "cursor-not-allowed opacity-50"
        )}
        onClick={handleToggleDropdown}
      >
        <div className="flex flex-wrap gap-1 flex-1 items-center">
          {values.length === 0 ? (
            <span className="text-muted-foreground">{placeholder}</span>
          ) : (
            values.map((value) => (
              <Badge
                key={value}
                variant="secondary"
                className="text-xs flex items-center gap-1 px-2 py-1"
              >
                <span>{value}</span>
                {!disabled && (
                  <X
                    className="h-3 w-3 cursor-pointer hover:text-destructive"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeValue(value);
                    }}
                  />
                )}
              </Badge>
            ))
          )}
          {isMaxReached && maxItems && (
            <span className="text-xs text-muted-foreground italic ml-auto">Max {maxItems}</span>
          )}
        </div>
        <ChevronDown className={cn(
          "h-4 w-4 text-muted-foreground ml-2 transition-transform",
          isOpen && "transform rotate-180"
        )} />
      </div>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 w-full mt-1 rounded-md border bg-popover shadow-lg">
          {/* Search input */}
          <div className="p-2 border-b">
            <Input
              ref={searchInputRef}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search..."
              className="h-8 text-sm"
              onClick={(e) => e.stopPropagation()}
            />
          </div>

          {/* Options list */}
          <div className="max-h-60 overflow-auto p-1">
            {filteredOptions.length === 0 ? (
              <div className="px-3 py-2 text-sm text-muted-foreground text-center">
                {searchQuery ? "No results found" : emptyMessage}
              </div>
            ) : (
              filteredOptions.map((option) => {
                const isSelected = values.includes(option);
                const canSelect = !isMaxReached || isSelected;

                return (
                  <div
                    key={option}
                    className={cn(
                      "flex items-center gap-2 px-3 py-2 text-sm rounded cursor-pointer",
                      "hover:bg-accent",
                      !canSelect && "opacity-50 cursor-not-allowed"
                    )}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (canSelect) {
                        toggleOption(option);
                      }
                    }}
                  >
                    <div className={cn(
                      "h-4 w-4 border rounded flex items-center justify-center",
                      isSelected && "bg-primary border-primary"
                    )}>
                      {isSelected && <Check className="h-3 w-3 text-primary-foreground" />}
                    </div>
                    <span>{option}</span>
                  </div>
                );
              })
            )}
          </div>

          {/* Footer with selected count */}
          {values.length > 0 && (
            <div className="p-2 border-t text-xs text-muted-foreground flex items-center justify-between">
              <span>{values.length} selected</span>
              {!disabled && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-xs"
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange([]);
                  }}
                >
                  Clear all
                </Button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
