import React, { useState, useCallback, useMemo } from 'react';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Search, Plus, ChevronLeft, Database, Settings, Trash2, Edit } from "lucide-react";
import {
  UITypeDefs,
  UITypeDef,
  UITypeDefStruct,
  UITypeDefCategorical,
  OutputConfiguration,
  validateUITypeDef,
  createDefaultStructTypeDef,
  createDefaultCategoricalTypeDef,
  updateDTypesForTypeDefChange
} from './output-types';
import { StructTypeEditor, CategoricalTypeEditor } from './CustomTypeEditor';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { updateStructReferencesWithMapping } from './updateStructReferences';
import { findCategoryReferences, removeCategoryReferences, findStructRecordReferences, removeStructRecordReferences } from './referenceAnalysis';
import { DeleteImpactDialog } from './DeleteImpactDialog';

interface CustomTypeManagerProps {
  uiTypeDefs: UITypeDefs;
  outputConfig: OutputConfiguration;
  onTypeDefsChange: (typeDefs: UITypeDefs) => void;
  onOutputChange: (output: Array<Record<string, any>>) => void;
  onDTypesChange: (dtypes: OutputConfiguration['dtypes']) => void;
  onClose: () => void;
}

type ViewMode = 'list' | 'edit';

export const CustomTypeManager: React.FC<CustomTypeManagerProps> = ({
  uiTypeDefs,
  outputConfig,
  onTypeDefsChange,
  onOutputChange,
  onDTypesChange,
  onClose
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [searchQuery, setSearchQuery] = useState('');
  const [editingType, setEditingType] = useState<UITypeDef | null>(null);
  const [editingTypeId, setEditingTypeId] = useState<string | null>(null);
  const [originalType, setOriginalType] = useState<UITypeDef | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [pendingSave, setPendingSave] = useState<{
    type: 'category-deletion' | 'struct-record-deletion';
    deletedItems: string[]; // Category values or struct display values
    swaps?: Array<{ from: string; to: string }>; // Display value swaps (for structs)
  } | null>(null);

  const typeDefEntries = Object.entries(uiTypeDefs);

  // Filter types based on search
  const filteredTypes = useMemo(() => {
    if (!searchQuery.trim()) return typeDefEntries;
    const query = searchQuery.toLowerCase();
    return typeDefEntries.filter(([_, typeDef]) =>
      typeDef.name.toLowerCase().includes(query) ||
      typeDef.type.toLowerCase().includes(query)
    );
  }, [typeDefEntries, searchQuery]);

  const handleCreateNew = useCallback((type: 'struct' | 'categorical') => {
    const newType = type === 'struct' ? createDefaultStructTypeDef() : createDefaultCategoricalTypeDef();
    setEditingType(newType);
    setOriginalType(null); // No original for new types
    setEditingTypeId(null);
    setErrors([]);
    setViewMode('edit');
  }, []);

  const handleEdit = useCallback((typeId: string) => {
    const typeDef = uiTypeDefs[typeId];
    if (typeDef) {
      const typeClone = JSON.parse(JSON.stringify(typeDef)); // Deep clone
      setEditingType(typeClone);
      setOriginalType(JSON.parse(JSON.stringify(typeDef))); // Store original
      setEditingTypeId(typeId);
      setErrors([]);
      setViewMode('edit');
    }
  }, [uiTypeDefs]);

  const handleSave = useCallback(() => {
    if (!editingType) return;

    const validationErrors = validateUITypeDef(editingType, uiTypeDefs);
    setErrors(validationErrors);

    if (validationErrors.length > 0) {
      return;
    }

    // Check for name conflicts with other types
    const conflictingEntry = typeDefEntries.find(([id, def]) =>
      id !== editingTypeId && def.name === editingType.name
    );

    if (conflictingEntry) {
      setErrors([`A type with the name "${editingType.name}" already exists`]);
      return;
    }

    // Find existing type ID by matching editingTypeId or create new one
    const typeId = editingTypeId || crypto.randomUUID();

    // Check for deletions in categorical types
    if (editingType.type === 'categorical' && originalType && originalType.type === 'categorical' && editingTypeId) {
      const deletedCategories = originalType.definition.categories.filter(
        cat => !editingType.definition.categories.includes(cat)
      );

      if (deletedCategories.length > 0) {
        // Check if any deleted categories have references
        const hasReferences = deletedCategories.some(cat => {
          const refs = findCategoryReferences(cat, editingTypeId, outputConfig);
          return refs.length > 0;
        });

        if (hasReferences) {
          setPendingSave({ type: 'category-deletion', deletedItems: deletedCategories });
          return; // Show dialog, don't save yet
        }
      }
    }

    // Check for deletions and swaps in struct types
    if (editingType.type === 'struct' && originalType && originalType.type === 'struct' && editingTypeId) {
      const displayFieldName = editingType.definition.display_field;
      const oldDisplayValues = new Set(
        originalType.definition.data.map(record => String(record[displayFieldName]))
      );
      const newDisplayValues = new Set(
        editingType.definition.data.map(record => String(record[displayFieldName]))
      );

      const deletedRecords = Array.from(oldDisplayValues).filter(val => !newDisplayValues.has(val));

      // Detect swaps/renames by matching records based on their content (excluding display field)
      // This works even when there are deletions
      const swaps: Array<{ from: string; to: string }> = [];

      // Build a map of "record content (minus display field)" -> display value for original records
      const originalRecordMap = new Map<string, string>();
      originalType.definition.data.forEach(record => {
        const { [displayFieldName]: displayVal, ...rest } = record;
        const contentKey = JSON.stringify(rest);
        originalRecordMap.set(contentKey, String(displayVal));
      });

      // For each new record, check if its content matches an original record with different display value
      editingType.definition.data.forEach(record => {
        const { [displayFieldName]: newDisplayVal, ...rest } = record;
        const contentKey = JSON.stringify(rest);
        const oldDisplayVal = originalRecordMap.get(contentKey);

        if (oldDisplayVal && oldDisplayVal !== String(newDisplayVal)) {
          // Same content, different display value = rename/swap
          swaps.push({ from: oldDisplayVal, to: String(newDisplayVal) });
        }
      });

      const hasImpact = deletedRecords.length > 0 || swaps.length > 0;

      if (hasImpact) {
        // Check if deletions have references
        const deletionHasReferences = deletedRecords.some(displayValue => {
          const refs = findStructRecordReferences(displayValue, editingTypeId, outputConfig);
          return refs.length > 0;
        });

        // Check if swaps affect any output data
        const swapHasReferences = swaps.some(swap => {
          const refs = findStructRecordReferences(swap.from, editingTypeId, outputConfig);
          return refs.length > 0;
        });

        if (deletionHasReferences || swapHasReferences) {
          setPendingSave({
            type: 'struct-record-deletion',
            deletedItems: deletedRecords,
            swaps: swaps.length > 0 ? swaps : undefined
          });
          return; // Show dialog, don't save yet
        }
      }
    }

    // No deletions with references, proceed with save
    performSave();
  }, [editingType, originalType, editingTypeId, uiTypeDefs, typeDefEntries, outputConfig]);

  const performSave = useCallback(() => {
    if (!editingType) return;

    // Find existing type ID by matching editingTypeId or create new one
    const typeId = editingTypeId || crypto.randomUUID();

    let finalTypeDefs = { ...uiTypeDefs, [typeId]: editingType };
    let finalOutputConfig = { ...outputConfig };

    // Clean up deleted categorical values if any
    if (pendingSave && pendingSave.type === 'category-deletion' && editingTypeId) {
      pendingSave.deletedItems.forEach(categoryValue => {
        finalOutputConfig = removeCategoryReferences(categoryValue, editingTypeId, finalOutputConfig);
      });
    }

    // Clean up deleted struct records if any
    if (pendingSave && pendingSave.type === 'struct-record-deletion' && editingTypeId) {
      pendingSave.deletedItems.forEach(displayValue => {
        finalOutputConfig = removeStructRecordReferences(displayValue, editingTypeId, finalOutputConfig);
      });
    }

    // If this is a struct type and we're updating an existing type, check for display field changes
    if (editingType.type === 'struct' && originalType && originalType.type === 'struct' && editingTypeId) {
      const displayFieldName = editingType.definition.display_field;

      // Build complete mapping of all display value changes
      const mapping: Record<string, string> = {};

      // Check each record to see if display values changed
      if (originalType.definition.data.length === editingType.definition.data.length) {
        for (let i = 0; i < originalType.definition.data.length; i++) {
          const oldDisplayValue = String(originalType.definition.data[i][displayFieldName]);
          const newDisplayValue = String(editingType.definition.data[i][displayFieldName]);

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
        if (updatedOutputConfig) {
          finalOutputConfig = updatedOutputConfig;
        }
      }
    }

    onTypeDefsChange(finalTypeDefs);

    // Update output if it changed
    if (JSON.stringify(finalOutputConfig.output) !== JSON.stringify(outputConfig.output)) {
      onOutputChange(finalOutputConfig.output);
    }

    // Update dtypes to sync struct field definitions
    if (editingType.type === 'struct' && editingTypeId) {
      const updatedDTypes = updateDTypesForTypeDefChange(
        finalOutputConfig.dtypes,
        typeId,
        editingType
      );

      // Trigger config update with new dtypes
      if (JSON.stringify(updatedDTypes) !== JSON.stringify(finalOutputConfig.dtypes)) {
        onDTypesChange(updatedDTypes);
      }
    }

    // Go back to list
    setEditingType(null);
    setOriginalType(null);
    setEditingTypeId(null);
    setErrors([]);
    setPendingSave(null);
    setViewMode('list');
  }, [editingType, originalType, editingTypeId, uiTypeDefs, outputConfig, pendingSave, onTypeDefsChange, onOutputChange, onDTypesChange]);

  const handleCancel = useCallback(() => {
    setEditingType(null);
    setOriginalType(null);
    setEditingTypeId(null);
    setErrors([]);
    setViewMode('list');
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
  }, [uiTypeDefs, onTypeDefsChange]);

  return (
    <div className="h-full flex flex-col">
      {viewMode === 'list' ? (
        /* List View */
        <div className="flex-1 overflow-auto p-6">
          <div className="space-y-4">
            {/* Search and Create Actions */}
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search types..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Button onClick={() => handleCreateNew('struct')} variant="outline">
                <Database className="h-4 w-4 mr-2" />
                New Struct
              </Button>
              <Button onClick={() => handleCreateNew('categorical')} variant="outline">
                <Settings className="h-4 w-4 mr-2" />
                New Categorical
              </Button>
            </div>

            {/* Type List */}
            {filteredTypes.length === 0 ? (
              <Card>
                <CardContent className="p-12 text-center">
                  <Settings className="h-12 w-12 mx-auto text-gray-300 mb-4" />
                  <p className="text-gray-500 mb-4">
                    {searchQuery ? 'No types found matching your search' : 'No custom types defined yet'}
                  </p>
                  <div className="flex gap-2 justify-center">
                    <Button onClick={() => handleCreateNew('struct')} variant="outline" size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Struct Type
                    </Button>
                    <Button onClick={() => handleCreateNew('categorical')} variant="outline" size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Categorical Type
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredTypes.map(([typeId, typeDef]) => (
                  <Card
                    key={typeId}
                    className="cursor-pointer hover:shadow-md transition-shadow"
                    onClick={() => handleEdit(typeId)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <h3 className="font-semibold text-lg">{typeDef.name}</h3>
                          <Badge variant="outline" className="mt-1">
                            {typeDef.type === 'struct' ? 'Struct' : 'Categorical'}
                          </Badge>
                        </div>
                        <div className="flex gap-1">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEdit(typeId);
                            }}
                            className="p-1 h-7 w-7"
                          >
                            <Edit className="h-3 w-3" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(typeId);
                            }}
                            className="p-1 h-7 w-7 text-red-600 hover:text-red-700"
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600">
                        {typeDef.type === 'struct'
                          ? `${typeDef.definition.fields.length} fields, ${typeDef.definition.data.length} records`
                          : `${typeDef.definition.categories.length} categories`
                        }
                      </p>
                    </CardContent>
                  </Card>
                ))}

                {/* Quick create actions when search has few results */}
                {filteredTypes.length > 0 && filteredTypes.length <= 3 && (
                  <>
                    <Card
                      className="cursor-pointer hover:shadow-md transition-shadow border-dashed border-2"
                      onClick={() => handleCreateNew('struct')}
                    >
                      <CardContent className="p-4 flex flex-col items-center justify-center h-full min-h-[120px]">
                        <Plus className="h-8 w-8 text-gray-400 mb-2" />
                        <p className="font-semibold text-gray-700">New Struct Type</p>
                        <p className="text-xs text-gray-500 mt-1">Create a structured object type</p>
                      </CardContent>
                    </Card>
                    <Card
                      className="cursor-pointer hover:shadow-md transition-shadow border-dashed border-2"
                      onClick={() => handleCreateNew('categorical')}
                    >
                      <CardContent className="p-4 flex flex-col items-center justify-center h-full min-h-[120px]">
                        <Plus className="h-8 w-8 text-gray-400 mb-2" />
                        <p className="font-semibold text-gray-700">New Categorical Type</p>
                        <p className="text-xs text-gray-500 mt-1">Create an enum type</p>
                      </CardContent>
                    </Card>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Edit View */
        <div className="flex-1 overflow-auto">
          <div className="p-6 space-y-6">
            {/* Back Button */}
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="sm" onClick={handleCancel}>
                <ChevronLeft className="h-4 w-4 mr-1" />
                Back to List
              </Button>
              <div className="flex-1">
                <h3 className="text-lg font-semibold">
                  {editingTypeId ? 'Edit Type' : `New ${editingType?.type === 'struct' ? 'Struct' : 'Categorical'} Type`}
                </h3>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" onClick={handleCancel}>
                  Cancel
                </Button>
                <Button onClick={handleSave}>
                  Save Type
                </Button>
              </div>
            </div>

            {/* Validation Errors */}
            {errors.length > 0 && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  <p className="font-medium">Please fix the following errors:</p>
                  <ul className="list-disc list-inside mt-1">
                    {errors.map((error, index) => (
                      <li key={index}>{error}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}

            {/* Type Basic Info */}
            {editingType && (
              <Card>
                <CardContent className="pt-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Type Name</label>
                      <Input
                        value={editingType.name}
                        onChange={(e) => setEditingType({ ...editingType, name: e.target.value })}
                        placeholder="e.g., ActionType, OutcomeCategories"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Type Kind</label>
                      <div className="h-10 flex items-center">
                        <Badge variant="outline" className="px-3 py-1">
                          {editingType.type === 'struct' ? 'Struct (Object)' : 'Categorical (Enum)'}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Type Editor */}
            {editingType && (
              editingType.type === 'struct' ? (
                <StructTypeEditor
                  typeDef={editingType as UITypeDefStruct}
                  onUpdateType={(updated: UITypeDefStruct) => setEditingType(updated)}
                  typeId={editingTypeId}
                  uiTypeDefs={uiTypeDefs}
                  outputConfig={outputConfig}
                />
              ) : (
                <CategoricalTypeEditor
                  typeDef={editingType as UITypeDefCategorical}
                  onUpdateType={(updated: UITypeDefCategorical) => setEditingType(updated)}
                />
              )
            )}
          </div>
        </div>
      )}

      {/* Impact Dialog for Save-time Deletions */}
      {pendingSave && editingType && editingTypeId && (
        <DeleteImpactDialog
          open={true}
          onOpenChange={(open: boolean) => {
            if (!open) setPendingSave(null);
          }}
          onConfirm={() => {
            performSave();
          }}
          itemType={pendingSave.type === 'category-deletion' ? 'category' : 'struct-record'}
          itemName={pendingSave.deletedItems.length === 1 ? pendingSave.deletedItems[0] : `${pendingSave.deletedItems.length} items`}
          typeName={editingType.name}
          categoryReferences={
            pendingSave.type === 'category-deletion' && editingTypeId
              ? pendingSave.deletedItems.flatMap(cat => findCategoryReferences(cat, editingTypeId, outputConfig))
              : []
          }
          structReferences={
            pendingSave.type === 'struct-record-deletion' && editingTypeId
              ? pendingSave.deletedItems.flatMap(val => findStructRecordReferences(val, editingTypeId, outputConfig))
              : []
          }
          swaps={pendingSave.swaps}
        />
      )}
    </div>
  );
};
