import React, { useState, useCallback, useEffect } from 'react';
import { Input } from "@/components/ui/input";
import { NumericInput } from "@/components/ui/numeric-input";
import { 
  VariableInfo,
  isStringVariable,
  isNumericVariable,
  validateVariableValue
} from '../editor/variable-types';

interface VariableEditorProps {
  variable: VariableInfo;
  onChange: (newValue: any) => void;
  className?: string;
}

export const VariableEditor: React.FC<VariableEditorProps> = ({
  variable,
  onChange,
  className = ""
}) => {
  const [localValue, setLocalValue] = useState<any>(variable.value);
  const [error, setError] = useState<string | null>(null);

  // Update local value when variable changes
  useEffect(() => {
    setLocalValue(variable.value);
    setError(null);
  }, [variable.value, variable.var_type]);

  const validateAndCommit = useCallback((value: any) => {
    const validationError = validateVariableValue(variable.var_type, value);
    if (validationError) {
      setError(validationError);
      return false;
    }
    
    setError(null);
    onChange(value);
    return true;
  }, [variable.var_type, onChange]);

  const handleStringChange = useCallback((value: string) => {
    setLocalValue(value);
    validateAndCommit(value);
  }, [validateAndCommit]);

  const handleNumericChange = useCallback((value: number | string | null) => {
    setLocalValue(value);
    validateAndCommit(value);
  }, [validateAndCommit]);


  if (isStringVariable(variable)) {
    return (
      <div className={`space-y-2 ${className}`}>
        <Input
          value={localValue}
          onChange={(e) => handleStringChange(e.target.value)}
          placeholder="Enter text value"
          className="text-sm"
        />
        {error && <div className="text-xs text-red-600">{error}</div>}
      </div>
    );
  }

  if (isNumericVariable(variable)) {
    return (
      <div className={`space-y-2 ${className}`}>
        <NumericInput
          value={localValue}
          onChange={handleNumericChange}
          placeholder="Enter number (supports inf, nan)"
          className="text-sm"
          allowEmpty={false}
          allowInfinity={true}
          allowNaN={true}
        />
        {error && <div className="text-xs text-red-600">{error}</div>}
      </div>
    );
  }


  return (
    <div className={`text-sm text-gray-500 ${className}`}>
      Unsupported variable type: {(variable as any).type}
    </div>
  );
};