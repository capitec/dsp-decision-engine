import React from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface UnaryOperatorSelectorProps {
  value: string;
  onChange: (newOp: string) => void;
  className?: string;
}

/**
 * Shared selector for all 13 unary operators.
 * Used by both flat rules editor and tree sidebar editor.
 */
export const UnaryOperatorSelector: React.FC<UnaryOperatorSelectorProps> = ({
  value,
  onChange,
  className = "w-32"
}) => {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className={className}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="<=">≤</SelectItem>
        <SelectItem value="<">&lt;</SelectItem>
        <SelectItem value="==">==</SelectItem>
        <SelectItem value=">">&gt;</SelectItem>
        <SelectItem value=">=">≥</SelectItem>
        <SelectItem value="!=">!=</SelectItem>
        <SelectItem value="between">Between</SelectItem>
        <SelectItem value="isin">IsIn</SelectItem>
        <SelectItem value="string_match">String Match</SelectItem>
        <SelectItem value="is_true">Is True</SelectItem>
        <SelectItem value="is_false">Is False</SelectItem>
      </SelectContent>
    </Select>
  );
};
