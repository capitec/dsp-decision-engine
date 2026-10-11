import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export interface NumericInputProps {
  value?: number | string | null;
  onChange?: (value: number | string | null) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
  step?: number | string;
  min?: number;
  max?: number;
  allowEmpty?: boolean;
  allowInfinity?: boolean;
  allowNaN?: boolean;
  allowNull?: boolean;
}

export const NumericInput: React.FC<NumericInputProps> = ({
  value,
  onChange,
  placeholder = "Enter number",
  className,
  disabled = false,
  step,
  min,
  max,
  allowEmpty = false,
  allowInfinity = true,
  allowNaN = false,
  allowNull = false
}) => {
  const [inputValue, setInputValue] = useState<string>('');
  const [hasError, setHasError] = useState(false);
  
  // Convert numeric value to string representation
  const valueToString = useCallback((val: number | string | null | undefined): string => {
    if (val === undefined) {
      return '';
    }

    if (val === null) {
      return 'null';
    }

    if (typeof val === 'string') {
      return val;
    }

    if (typeof val === 'number') {
      if (isNaN(val)) {
        return 'nan';
      }
      if (val === Infinity) {
        return 'inf';
      }
      if (val === -Infinity) {
        return '-inf';
      }
      return val.toString();
    }

    return '';
  }, []);

  // Parse string input to numeric value
  const parseStringValue = useCallback((str: string): number | string | null => {
    const trimmed = str.trim().toLowerCase();

    // Handle empty input
    if (trimmed === '') {
      return allowEmpty ? '' : null;
    }

    // Handle null value
    if (allowNull && trimmed === 'null') {
      return null;
    }

    // Handle special string values - return as strings for JSON compatibility
    if (allowInfinity) {
      if (trimmed === 'inf' || trimmed === 'infinity' || trimmed === '+inf' || trimmed === '+infinity') {
        return 'inf';
      }
      if (trimmed === '-inf' || trimmed === '-infinity') {
        return '-inf';
      }
    }

    if (allowNaN && trimmed === 'nan') {
      return 'nan';
    }

    // Try to parse as regular number
    const parsed = parseFloat(str);
    if (isNaN(parsed)) {
      return null; // Invalid input
    }

    // Apply min/max constraints
    if (min !== undefined && parsed < min) {
      return null;
    }
    if (max !== undefined && parsed > max) {
      return null;
    }

    return parsed;
  }, [allowEmpty, allowInfinity, allowNaN, allowNull, min, max]);

  // Update input value when prop changes
  useEffect(() => {
    setInputValue(valueToString(value));
    setHasError(false);
  }, [value, valueToString]);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const newInputValue = e.target.value;
    setInputValue(newInputValue);

    const parsedValue = parseStringValue(newInputValue);

    if (parsedValue === null && newInputValue.trim() !== '' && newInputValue.trim().toLowerCase() !== 'null') {
      // Invalid input - show error but don't call onChange
      setHasError(true);
    } else {
      // Valid input or empty
      setHasError(false);
      if (parsedValue === '' && allowEmpty) {
        onChange?.('');
      } else if (typeof parsedValue === 'number') {
        onChange?.(parsedValue);
      } else if (typeof parsedValue === 'string' && parsedValue !== '') {
        // Handle special string values like "inf", "-inf"
        onChange?.(parsedValue);
      } else if (parsedValue === null) {
        // Handle null value explicitly
        onChange?.(null);
      }
    }
  }, [parseStringValue, onChange, allowEmpty]);

  const handleBlur = useCallback(() => {
    // On blur, if there's an error, reset to the original value
    if (hasError) {
      setInputValue(valueToString(value));
      setHasError(false);
      return;
    }

    // If field is empty and allowEmpty is false, revert to old value or default to 0
    if (inputValue.trim() === '' && !allowEmpty) {
      if (value === undefined || value === null || value === '') {
        // No existing value, set default to 0
        onChange?.(0);
        setInputValue('0');
      } else {
        // Revert to existing value
        setInputValue(valueToString(value));
      }
    }
  }, [hasError, value, valueToString, inputValue, allowEmpty, onChange]);

  // Build tooltip message for special values
  const tooltipMessage = useMemo(() => {
    const specialValues: string[] = [];
    if (allowInfinity) {
      specialValues.push('"inf" for ∞', '"-inf" for -∞');
    }
    if (allowNaN) {
      specialValues.push('"nan" for NaN');
    }
    if (allowNull) {
      specialValues.push('"null" for null');
    }
    return specialValues.length > 0 ? `Special values: ${specialValues.join(', ')}` : null;
  }, [allowInfinity, allowNaN, allowNull]);

  const input = (
    <Input
      value={inputValue}
      onChange={handleInputChange}
      onBlur={handleBlur}
      placeholder={placeholder}
      className={cn(
        hasError && "border-red-500 focus-visible:ring-red-500",
        className
      )}
      disabled={disabled}
      step={step}
    />
  );

  // If there are special values allowed, wrap in tooltip
  if (tooltipMessage) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            {input}
          </TooltipTrigger>
          <TooltipContent>
            <p className="text-xs">{tooltipMessage}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return input;
};