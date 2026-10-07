import React, { useState, useEffect, useCallback } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';
import type { UITypeDef, UITypeDefStruct, UITypeDefCategorical } from '@/types/tree-types';
import { UnifiedOutputTable } from './UnifiedOutputTable';
import type { OutputConfiguration } from './output-types';
import { CategoricalEditor } from './CategoricalEditor';

interface CustomTypeEditorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: 'create' | 'edit';
  typeKind: 'struct' | 'categorical';
  existingType?: UITypeDef;
  existingTypeId?: string;
  onSave: (typeId: string, typeDef: UITypeDef) => void;
  usedTypeIds: string[];
}

/**
 * Dialog for creating or editing custom type definitions.
 * Delegates to StructEditor or CategoricalEditor based on type kind.
 */
export function CustomTypeEditor({
  open,
  onOpenChange,
  mode,
  typeKind,
  existingType,
  existingTypeId,
  onSave,
  usedTypeIds,
}: CustomTypeEditorProps) {
  const [name, setName] = useState('');
  const [structOutputConfig, setStructOutputConfig] = useState<OutputConfiguration>({
    dtypes: [],
    type_defs: {},
    output: [],
    default: {}
  });
  const [categories, setCategories] = useState<string[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  // Initialize state when dialog opens or type changes
  useEffect(() => {
    if (open) {
      if (existingType) {
        setName(existingType.name);
        if (existingType.type === 'struct') {
          // Convert to OutputConfiguration format
          setStructOutputConfig({
            dtypes: existingType.definition.fields,
            type_defs: {},
            output: existingType.definition.data,
            default: existingType.definition.data[0] || {}
          });
        } else {
          setCategories(existingType.definition.categories);
        }
      } else {
        // Reset for new type
        setName('');
        setStructOutputConfig({
          dtypes: [],
          type_defs: {},
          output: [],
          default: {}
        });
        setCategories([]);
      }
      setErrors([]);
    }
  }, [open, existingType]);

  const validateAndSave = () => {
    const newErrors: string[] = [];

    // Validate name
    if (!name.trim()) {
      newErrors.push('Name is required');
    }

    // Generate type ID (kebab-case from name)
    const typeId = existingTypeId || name.trim().toLowerCase().replace(/\s+/g, '-');

    // Check for duplicate type ID (only in create mode)
    if (mode === 'create' && usedTypeIds.includes(typeId)) {
      newErrors.push(`Type ID "${typeId}" already exists. Choose a different name.`);
    }

    if (typeKind === 'struct') {
      // Validate struct
      if (structOutputConfig.dtypes.length === 0) {
        newErrors.push('At least one field is required');
      }
      const displayField = structOutputConfig.dtypes[0]?.[0];
      // Check display field values are unique
      if (displayField && structOutputConfig.output.length > 0) {
        const displayValues = structOutputConfig.output.map((r: any) => r[displayField]);
        const uniqueValues = new Set(displayValues.filter((v: any) => v !== undefined && v !== '' && v !== null));
        if (uniqueValues.size !== displayValues.filter((v: any) => v !== undefined && v !== '' && v !== null).length) {
          newErrors.push('Display field values must be unique across all records');
        }
        if (displayValues.some((v: any) => v === undefined || v === '' || v === null)) {
          newErrors.push('Display field values cannot be empty');
        }
      }
    } else {
      // Validate categorical
      if (categories.length === 0) {
        newErrors.push('At least one category is required');
      }
    }

    if (newErrors.length > 0) {
      setErrors(newErrors);
      return;
    }

    // Build type definition
    let typeDef: UITypeDef;
    if (typeKind === 'struct') {
      const displayField = structOutputConfig.dtypes[0]?.[0] || '';
      typeDef = {
        name: name.trim(),
        type: 'struct',
        definition: {
          fields: structOutputConfig.dtypes,
          data: structOutputConfig.output,
          display_field: displayField,
        },
      } as UITypeDefStruct;
    } else {
      typeDef = {
        name: name.trim(),
        type: 'categorical',
        definition: {
          categories,
        },
      } as UITypeDefCategorical;
    }

    onSave(typeId, typeDef);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {mode === 'create' ? 'Create' : 'Edit'} {typeKind === 'struct' ? 'Struct' : 'Categorical'} Type
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {errors.length > 0 && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                <ul className="list-disc list-inside">
                  {errors.map((error, i) => (
                    <li key={i}>{error}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          )}

          <div>
            <Label htmlFor="type-name">Type Name</Label>
            <Input
              id="type-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Customer, Product Category"
            />
          </div>

          {typeKind === 'struct' ? (
            <UnifiedOutputTable
              outputConfig={structOutputConfig}
              onConfigChange={(updates) => setStructOutputConfig({ ...structOutputConfig, ...updates })}
              mode="struct"
              displayFieldName={structOutputConfig.dtypes[0]?.[0]}
              title="Struct Fields & Data"
              addRecordLabel="Add Record"
              addFieldLabel="Add Field"
            />
          ) : (
            <CategoricalEditor
              categories={categories}
              onCategoriesChange={setCategories}
            />
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={validateAndSave}>
            {mode === 'create' ? 'Create' : 'Save'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
