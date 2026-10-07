import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";

export type ComparisonOperator = "<=" | "<" | "==" | ">" | ">=" | "!=";

interface ComparisonOperatorSelectorProps {
  value: ComparisonOperator;
  onChange: (value: ComparisonOperator) => void;
  label?: string;
  showLabel?: boolean;
  disabled?: boolean;
  className?: string;
  /** Show only symbol in trigger (e.g., "<" instead of "< (Less than)") */
  compact?: boolean;
}

/**
 * Shared comparison operator selector with proper Unicode symbols (≤, ≥)
 * Used in both tree editor numerical nodes and flat-rules numeric conditions
 */
export const ComparisonOperatorSelector: React.FC<ComparisonOperatorSelectorProps> = ({
  value,
  onChange,
  label = "Comparison Operator",
  showLabel = true,
  disabled = false,
  className = "",
  compact = false
}) => {
  return (
    <div className={`space-y-2 ${className}`}>
      {showLabel && <Label>{label}</Label>}
      <Select
        value={value}
        onValueChange={(v) => onChange(v as ComparisonOperator)}
        disabled={disabled}
      >
        <SelectTrigger>
          {compact ? (
            <span className="text-base">{getOperatorSymbol(value)}</span>
          ) : (
            <SelectValue />
          )}
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="<=">≤ (Less than or equal)</SelectItem>
          <SelectItem value="<">{'<'} (Less than)</SelectItem>
          <SelectItem value="==">=  (Equal to)</SelectItem>
          <SelectItem value="!=">≠ (Not equal to)</SelectItem>
          <SelectItem value=">">{'>'} (Greater than)</SelectItem>
          <SelectItem value=">=">≥ (Greater than or equal)</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};

/**
 * Get Unicode symbol for comparison operator
 */
export function getOperatorSymbol(operator: ComparisonOperator): string {
  const symbolMap: Record<ComparisonOperator, string> = {
    "<=": "≤",
    "<": "<",
    "==": "=",
    "!=": "≠",
    ">": ">",
    ">=": "≥",
  };
  return symbolMap[operator] || operator;
}
