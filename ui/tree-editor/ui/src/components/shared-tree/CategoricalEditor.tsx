import React from 'react';
import { Label } from '@/components/ui/label';
import { ChipInput } from '@/components/ui/chip-input';

interface CategoricalEditorProps {
  categories: string[];
  onCategoriesChange: (categories: string[]) => void;
}

/**
 * Simple categorical type editor.
 * Manages list of category values with add/remove.
 */
export function CategoricalEditor({
  categories,
  onCategoriesChange,
}: CategoricalEditorProps) {
  return (
    <div className="space-y-2">
      <Label>Categories ({categories.length})</Label>
      <ChipInput
        values={categories}
        onChange={onCategoriesChange}
        placeholder="Add category and press Enter..."
        className="mt-2"
      />
      <p className="text-xs text-muted-foreground">
        Press Enter or comma to add categories. Click the × to remove.
      </p>
    </div>
  );
}
