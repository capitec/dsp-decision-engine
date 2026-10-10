import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface CompositeLogicSelectorProps {
  value: 'and' | 'or' | 'not';
  onChange: (newLogic: 'and' | 'or' | 'not') => void;
  /** NOT is not offered once there is more than one condition to negate */
  conditionCount?: number;
  className?: string;
}

/**
 * Shared selector for composite logic operators (AND/OR/NOT).
 * Used by both flat rules editor and tree sidebar editor.
 */
export const CompositeLogicSelector: React.FC<CompositeLogicSelectorProps> = ({
  value,
  onChange,
  conditionCount = 0,
  className = "w-24"
}) => {
  return (
    <Select value={value} onValueChange={(v) => onChange(v as 'and' | 'or' | 'not')}>
      <SelectTrigger className={className} data-testid="composite-op">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="and">AND</SelectItem>
        <SelectItem value="or">OR</SelectItem>
        <SelectItem value="not" disabled={conditionCount > 1}>NOT</SelectItem>
      </SelectContent>
    </Select>
  );
};
