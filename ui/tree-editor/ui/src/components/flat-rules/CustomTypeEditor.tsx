import React, { useState, useCallback, useMemo } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Save, AlertCircle, Settings, Trash2, Copy, FileText, Database } from "lucide-react";
import { ChipInput } from "@/components/ui/chip-input";
import { useToast } from "@/hooks/use-toast";
import {
  UITypeDefs,
  UITypeDef,
  UITypeDefStruct,
  UITypeDefCategorical,
  validateUITypeDef,
  createDefaultStructTypeDef,
  createDefaultCategoricalTypeDef
} from './output-types';
import { UnifiedOutputTable } from '@/components/shared-tree';
import type { OutputConfiguration } from '@/components/shared-tree';
import { updateStructReferencesWithMapping } from './updateStructReferences';
import { structFieldProblems, isOfferedInStructField } from './struct-field-types';

interface CustomTypeEditorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  uiTypeDefs: UITypeDefs;
  onTypeDefsChange: (typeDefs: UITypeDefs) => void;
  outputConfig?: OutputConfiguration;
  onOutputConfigChange?: (updates: Partial<OutputConfiguration>) => void;
}

export const CustomTypeEditor: React.FC<CustomTypeEditorProps> = ({
  open,
  onOpenChange,
  uiTypeDefs,
  onTypeDefsChange,
  outputConfig,
  onOutputConfigChange,
}) => {
  const [selectedTypeId, setSelectedTypeId] = useState<string | null>(null);
  const [editingType, setEditingType] = useState<UITypeDef | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const { toast } = useToast();

  const typeDefEntries = Object.entries(uiTypeDefs);
  const selectedType = selectedTypeId ? uiTypeDefs[selectedTypeId] : null;

  const handleCreateNew = useCallback((type: 'struct' | 'categorical') => {
    const newType = type === 'struct' ? createDefaultStructTypeDef() : createDefaultCategoricalTypeDef();
    setEditingType(newType);
    setSelectedTypeId(null);
    setErrors([]);
  }, []);

  const handleEdit = useCallback((typeId: string) => {
    const typeDef = uiTypeDefs[typeId];
    if (typeDef) {
      setEditingType({ ...typeDef });
      setSelectedTypeId(null);
      setErrors([]);
    }
  }, [uiTypeDefs]);

  const handleSave = useCallback(() => {
    if (!editingType) return;

    const validationErrors = validateUITypeDef(editingType, uiTypeDefs);
    setErrors(validationErrors);

    if (validationErrors.length > 0) {
      return;
    }

    // Find existing type ID by name match or create new one
    const existingEntry = typeDefEntries.find(([_, def]) => def.name === editingType.name);
    const typeId = existingEntry ? existingEntry[0] : crypto.randomUUID();

    let finalTypeDefs = { ...uiTypeDefs, [typeId]: editingType };
    let finalOutputConfig = outputConfig;

    // If this is a struct type and we're updating an existing type, check for display field changes
    if (editingType.type === 'struct' && existingEntry) {
      const oldTypeDef = uiTypeDefs[typeId] as UITypeDefStruct;
      const newTypeDef = editingType as UITypeDefStruct;
      const displayFieldName = newTypeDef.definition.display_field;

      // Build complete mapping of all display value changes
      const mapping: Record<string, string> = {};

      // Check each record to see if display values changed
      if (oldTypeDef.definition.data.length === newTypeDef.definition.data.length) {
        for (let i = 0; i < oldTypeDef.definition.data.length; i++) {
          const oldDisplayValue = String(oldTypeDef.definition.data[i][displayFieldName]);
          const newDisplayValue = String(newTypeDef.definition.data[i][displayFieldName]);

          if (oldDisplayValue !== newDisplayValue) {
            mapping[oldDisplayValue] = newDisplayValue;
          }
        }
      }

      // Apply all changes at once if there are any
      if (Object.keys(mapping).length > 0) {
        const { updatedTypeDefs, updatedOutputConfig } = updateStructReferencesWithMapping(
          mapping,
          finalTypeDefs,
          finalOutputConfig
        );

        finalTypeDefs = updatedTypeDefs;
        finalOutputConfig = updatedOutputConfig;
      }
    }

    onTypeDefsChange(finalTypeDefs);

    if (finalOutputConfig && onOutputConfigChange) {
      onOutputConfigChange(finalOutputConfig);
    }

    setEditingType(null);
    setErrors([]);
    setSelectedTypeId(typeId); // Select the saved type

    toast({
      title: existingEntry ? "Type updated" : "Type created",
      description: `"${editingType.name}" has been ${existingEntry ? 'updated' : 'created'} successfully.`,
    });
  }, [editingType, uiTypeDefs, typeDefEntries, outputConfig, onTypeDefsChange, onOutputConfigChange, toast]);

  const handleCancel = useCallback(() => {
    setEditingType(null);
    setErrors([]);
  }, []);

  const handleDelete = useCallback((typeId: string) => {
    const typeDef = uiTypeDefs[typeId];
    if (!typeDef) return;

    if (!confirm(`Are you sure you want to delete "${typeDef.name}"?`)) {
      return;
    }

    const updatedTypeDefs = { ...uiTypeDefs };
    delete updatedTypeDefs[typeId];
    onTypeDefsChange(updatedTypeDefs);
    
    if (selectedTypeId === typeId) {
      setSelectedTypeId(null);
    }
    
    toast({
      title: "Type deleted",
      description: `"${typeDef.name}" has been deleted.`,
    });
  }, [uiTypeDefs, selectedTypeId, onTypeDefsChange, toast]);

  const handleDuplicate = useCallback((typeId: string) => {
    const typeDef = uiTypeDefs[typeId];
    if (!typeDef) return;

    const duplicatedType = {
      ...typeDef,
      name: `${typeDef.name} Copy`
    };

    const newTypeId = crypto.randomUUID();
    const updatedTypeDefs = { ...uiTypeDefs, [newTypeId]: duplicatedType };
    onTypeDefsChange(updatedTypeDefs);
    
    toast({
      title: "Type duplicated",
      description: `"${duplicatedType.name}" has been created.`,
    });
  }, [uiTypeDefs, onTypeDefsChange, toast]);

  if (!open) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-7xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Custom Type Definitions
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-hidden flex gap-6">
          {/* Left Panel - Type List */}
          <div className="w-80 flex flex-col gap-4">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">Type Definitions ({typeDefEntries.length})</CardTitle>
                  <div className="flex gap-1">
                    <Button size="sm" variant="outline" onClick={() => handleCreateNew('struct')}>
                      <Database className="h-3 w-3 mr-1" />
                      Struct
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => handleCreateNew('categorical')}>
                      <Settings className="h-3 w-3 mr-1" />
                      Categorical
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                {typeDefEntries.length === 0 ? (
                  <p className="text-sm text-gray-500 text-center py-6">
                    No custom types defined yet.
                  </p>
                ) : (
                  typeDefEntries.map(([typeId, typeDef]) => (
                    <div
                      key={typeId}
                      className={`p-3 border rounded cursor-pointer hover:bg-gray-50 ${
                        selectedTypeId === typeId ? 'border-blue-500 bg-blue-50' : ''
                      }`}
                      onClick={() => setSelectedTypeId(typeId)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <h4 className="font-medium text-sm">{typeDef.name}</h4>
                            <Badge variant="outline" className="text-xs">
                              {typeDef.type}
                            </Badge>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">
                            {typeDef.type === 'struct'
                              ? `${typeDef.definition.fields.length} fields, ${typeDef.definition.data.length} records`
                              : `${typeDef.definition.categories.length} categories`
                            }
                          </p>
                        </div>
                        <div className="flex gap-1 ml-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEdit(typeId);
                            }}
                            className="p-1 h-6 w-6"
                          >
                            <Settings className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDuplicate(typeId);
                            }}
                            className="p-1 h-6 w-6"
                          >
                            <Copy className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(typeId);
                            }}
                            className="p-1 h-6 w-6 text-red-600"
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Panel - Type Editor or Preview */}
          <div className="flex-1 overflow-y-auto">
            {editingType ? (
              <TypeEditor
                typeDef={editingType}
                errors={errors}
                onUpdateType={setEditingType}
                onSave={handleSave}
                onCancel={handleCancel}
                typeId={selectedTypeId}
                uiTypeDefs={uiTypeDefs}
                outputConfig={outputConfig}
                onOutputConfigChange={onOutputConfigChange}
              />
            ) : selectedType && selectedTypeId ? (
              <TypePreview
                typeDef={selectedType}
                uiTypeDefs={uiTypeDefs}
                onEdit={() => handleEdit(selectedTypeId)}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <FileText className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <p>Select a type to view details</p>
                  <p className="text-sm mt-1">or create a new one to get started</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

// Type Editor Component
interface TypeEditorProps {
  typeDef: UITypeDef;
  errors: string[];
  onUpdateType: (typeDef: UITypeDef) => void;
  onSave: () => void;
  onCancel: () => void;
  typeId: string | null;
  uiTypeDefs: UITypeDefs;
  outputConfig?: OutputConfiguration;
  onOutputConfigChange?: (updates: Partial<OutputConfiguration>) => void;
}

const TypeEditor: React.FC<TypeEditorProps> = ({
  typeDef,
  errors,
  onUpdateType,
  onSave,
  onCancel,
  typeId,
  uiTypeDefs,
  outputConfig,
  onOutputConfigChange,
}) => {
  const updateBasicInfo = useCallback((updates: Partial<Pick<UITypeDef, 'name'>>) => {
    onUpdateType({ ...typeDef, ...updates });
  }, [typeDef, onUpdateType]);

  return (
    <div className="space-y-6">
      {/* Validation errors */}
      {errors.length > 0 && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-4">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-4 w-4 text-red-600 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-red-800">Please fix the following errors:</p>
                <ul className="list-disc list-inside text-sm text-red-700 mt-1">
                  {errors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Basic info */}
      <Card>
        <CardHeader>
          <CardTitle>Type Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Type Name *</Label>
              <Input
                id="name"
                value={typeDef.name}
                onChange={(e) => updateBasicInfo({ name: e.target.value })}
                placeholder="e.g., ActionType, OutcomeCategories"
              />
            </div>
            <div className="space-y-2">
              <Label>Type Kind</Label>
              <Badge variant="outline" className="px-3 py-1">
                {typeDef.type === 'struct' ? 'Struct (Object)' : 'Categorical (Enum)'}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Actions at the top */}
      <div className="flex justify-end gap-3 pb-4 border-b">
        <Button variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button onClick={onSave}>
          <Save className="h-4 w-4 mr-2" />
          Save Type
        </Button>
      </div>

      {/* Type-specific editors */}
      {typeDef.type === 'struct' ? (
        <StructTypeEditor
          typeDef={typeDef as UITypeDefStruct}
          onUpdateType={onUpdateType as (t: UITypeDefStruct) => void}
          typeId={typeId}
          uiTypeDefs={uiTypeDefs}
          outputConfig={outputConfig}
          onOutputConfigChange={onOutputConfigChange}
        />
      ) : (
        <CategoricalTypeEditor
          typeDef={typeDef as UITypeDefCategorical}
          onUpdateType={onUpdateType as (t: UITypeDefCategorical) => void}
        />
      )}
    </div>
  );
};

// Struct Type Editor
export interface StructTypeEditorProps {
  typeDef: UITypeDefStruct;
  onUpdateType: (typeDef: UITypeDefStruct) => void;
  typeId?: string | null;
  uiTypeDefs?: UITypeDefs;
  outputConfig?: OutputConfiguration; // Parent output config for checking references
  onOutputConfigChange?: (updates: Partial<OutputConfiguration>) => void;
}

export const StructTypeEditor: React.FC<StructTypeEditorProps> = ({
  typeDef,
  onUpdateType,
  uiTypeDefs,
}) => {
  // A struct can't refer to itself, so leave it out of the pickers
  const availableTypeDefs = useMemo(
    () => Object.fromEntries(Object.entries(uiTypeDefs ?? {}).filter(([, def]) => def.name !== typeDef.name)),
    [uiTypeDefs, typeDef.name]
  );
  const fieldProblems = useMemo(() => structFieldProblems(typeDef, availableTypeDefs), [typeDef, availableTypeDefs]);
  const customTypeFilter = useCallback(
    (def: UITypeDef, depth: number) => isOfferedInStructField(def, depth, availableTypeDefs),
    [availableTypeDefs]
  );

  const displayFieldName = typeDef.definition.fields.length > 0 ? typeDef.definition.fields[0][0] : undefined;

  // Convert struct typeDef to OutputConfiguration format
  const structOutputConfig: OutputConfiguration = {
    dtypes: typeDef.definition.fields,
    type_defs: availableTypeDefs,
    output: typeDef.definition.data,
    default: typeDef.definition.data.length > 0 ? typeDef.definition.data[0] : {}
  };

  const handleConfigChange = useCallback((updates: Partial<OutputConfiguration>) => {
    const newFields = updates.dtypes || typeDef.definition.fields;
    const newData = updates.output || typeDef.definition.data;

    // Display field is always the first field
    const newDisplayField = newFields.length > 0 ? newFields[0][0] : '';

    // Just update the local editing state
    // Reference updates will be handled at save time
    onUpdateType({
      ...typeDef,
      definition: {
        ...typeDef.definition,
        fields: newFields,
        data: newData,
        display_field: newDisplayField
      }
    });
  }, [typeDef, onUpdateType]);

  const handleValidateCell = useCallback((value: any, fieldName: string, recordIndex: number) => {
    // Validate display field (first field) uniqueness and non-emptiness
    if (fieldName === displayFieldName) {
      // Check if empty
      if (!value || value.toString().trim() === '') {
        return {
          valid: false,
          error: 'Display field value cannot be empty'
        };
      }

      // Check for duplicates
      const duplicateIndex = typeDef.definition.data.findIndex((record: any, i: number) =>
        i !== recordIndex && record[fieldName] === value
      );

      if (duplicateIndex !== -1) {
        return {
          valid: false,
          error: `This value already exists in record ${duplicateIndex + 1}. Display field values must be unique.`
        };
      }
    }

    return { valid: true };
  }, [typeDef.definition.data, displayFieldName]);

  return (
    <>
      {Object.keys(fieldProblems).length > 0 && (
        <Card className="border-amber-200 bg-amber-50">
          <CardContent className="pt-4">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-4 w-4 text-amber-600 mt-0.5" />
              <ul className="list-disc list-inside text-sm text-amber-800">
                {Object.entries(fieldProblems).map(([name, problem]) => (
                  <li key={name}><strong>{name}</strong>: {problem}</li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Display Field Info */}
      {displayFieldName && (
        <Card className="border-blue-200 bg-blue-50">
          <CardContent className="pt-4">
            <div className="flex items-start gap-2">
              <div className="flex-1">
                <p className="text-sm font-medium text-blue-800">Display Field</p>
                <p className="text-xs text-blue-600 mt-1">
                  The first field "<strong>{displayFieldName}</strong>" is automatically used as the display value in dropdowns. Display field values must be unique and cannot be empty.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <UnifiedOutputTable
        outputConfig={structOutputConfig}
        onConfigChange={handleConfigChange}
        mode="struct"
        displayFieldName={displayFieldName}
        customTypeFilter={customTypeFilter}
        columnWarnings={fieldProblems}
        onValidateCell={handleValidateCell}
        title="Fields & Data"
        addRecordLabel="Add Record"
        addFieldLabel="Add Field"
      />
    </>
  );
};

// Categorical Type Editor
export interface CategoricalTypeEditorProps {
  typeDef: UITypeDefCategorical;
  onUpdateType: (typeDef: UITypeDefCategorical) => void;
}

export const CategoricalTypeEditor: React.FC<CategoricalTypeEditorProps> = ({
  typeDef,
  onUpdateType,
}) => {
  const updateCategories = useCallback((categories: string[]) => {
    // Just update immediately, impact analysis will happen on save
    onUpdateType({
      ...typeDef,
      definition: { ...typeDef.definition, categories }
    });
  }, [typeDef, onUpdateType]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Categories ({typeDef.definition.categories.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <Label>Category Values</Label>
            <ChipInput
              values={typeDef.definition.categories}
              onChange={updateCategories}
              placeholder="Add category..."
              className="mt-2"
            />
          </div>
          <div className="text-sm text-gray-600">
            <p>Categories define the possible values for this categorical type.</p>
            <p>Press <strong>Enter</strong> to add a category.</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Type Preview Component
interface TypePreviewProps {
  typeDef: UITypeDef;
  uiTypeDefs: UITypeDefs;
  onEdit: () => void;
}

const TypePreview: React.FC<TypePreviewProps> = ({ typeDef, uiTypeDefs, onEdit }) => {
  const fieldProblems = typeDef.type === 'struct' ? structFieldProblems(typeDef, uiTypeDefs) : {};
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>{typeDef.name}</CardTitle>
            <Button variant="outline" size="sm" onClick={onEdit}>
              <Settings className="h-4 w-4 mr-2" />
              Edit
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <Label>Type</Label>
              <Badge variant="outline" className="ml-2">
                {typeDef.type === 'struct' ? 'Struct (Object)' : 'Categorical (Enum)'}
              </Badge>
            </div>

            {typeDef.type === 'struct' ? (
              <>
                <div>
                  <Label>Fields ({typeDef.definition.fields.length})</Label>
                  <div className="mt-2 space-y-1">
                    {typeDef.definition.fields.map(([fieldName, fieldType]) => (
                      <div key={fieldName} className="p-2 bg-gray-50 rounded">
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{fieldName}</span>
                          <Badge variant="outline">{typeof fieldType === 'string' ? fieldType : 'Struct'}</Badge>
                        </div>
                        {fieldProblems[fieldName] && (
                          <p className="text-xs text-amber-700 mt-1">{fieldProblems[fieldName]}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <Label>Data Records ({typeDef.definition.data.length})</Label>
                  <div className="mt-2 max-h-48 overflow-auto">
                    <pre className="bg-gray-100 p-3 rounded text-xs">
                      {JSON.stringify(typeDef.definition.data, null, 2)}
                    </pre>
                  </div>
                </div>
              </>
            ) : (
              <div>
                <Label>Categories ({typeDef.definition.categories.length})</Label>
                <div className="mt-2 flex flex-wrap gap-1">
                  {typeDef.definition.categories.map((category, index) => (
                    <Badge key={index} variant="secondary">{category}</Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

