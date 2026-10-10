import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Plus, Trash2, GripVertical, Settings } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  OutputConfiguration,
  OutputSchema,
  PolarsDataType,
  COMMON_POLARS_TYPES,
  createDTypeEntry,
  parseDTypeEntry
} from './output-types';
export type { OutputConfiguration, OutputSchema, PolarsDataType } from './output-types';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors, DragEndEvent } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, horizontalListSortingStrategy, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { EnhancedFieldEditor, UnifiedTypeSelector } from '@/components/shared-tree';
import { validateDisplayFormat, formatOutputLabel } from '@/components/editor/util';
import type { UITypeDef } from '@/types/tree-types';

interface UnifiedOutputTableProps {
  outputConfig: OutputConfiguration;
  onConfigChange: (updates: Partial<OutputConfiguration>) => void;
  onManageCustomTypes?: () => void;
  /** Limits which custom types the column type pickers offer (see UnifiedTypeSelector) */
  customTypeFilter?: (typeDef: UITypeDef, depth: number) => boolean;
  /** Inline warning per column name, shown under the column type */
  columnWarnings?: Record<string, string>;
  /** Called after the output row at `index` is deleted, so leaves can be re-pointed */
  onRecordRemoved?: (index: number) => void;

  // Mode configuration
  mode?: 'output' | 'struct';  // Default: 'output'
  displayFieldName?: string;    // First field name (used in struct mode for locking)

  // Validation
  onValidateCell?: (value: any, fieldName: string, recordIndex: number) => {
    valid: boolean;
    error?: string;
  };

  // Labels
  title?: string;
  addRecordLabel?: string;
  addFieldLabel?: string;
}


export const UnifiedOutputTable: React.FC<UnifiedOutputTableProps> = ({
  outputConfig,
  onConfigChange,
  onManageCustomTypes,
  customTypeFilter,
  columnWarnings,
  onRecordRemoved,
  mode = 'output',
  displayFieldName,
  onValidateCell,
  title,
  addRecordLabel = 'Add Record',
  addFieldLabel = 'Add Field',
}) => {
  const [newColumnName, setNewColumnName] = useState('');
  const [newColumnType, setNewColumnType] = useState<PolarsDataType>('String');
  const [showAddColumn, setShowAddColumn] = useState(false);
  const [showDisplayFormatDialog, setShowDisplayFormatDialog] = useState(false);
  const [displayFormatInput, setDisplayFormatInput] = useState('');
  const [displayFormatError, setDisplayFormatError] = useState<string | undefined>();
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const newColumnInputRef = useRef<HTMLInputElement>(null);

  // Determine the actual display field (first field if not specified)
  const actualDisplayField = displayFieldName || outputConfig.dtypes[0]?.[0];
  const isStructMode = mode === 'struct';

  // Get field names for validation
  const fieldNames = useMemo(() => {
    return outputConfig.dtypes.map(([name]) => name);
  }, [outputConfig.dtypes]);

  // Initialize display format input when dialog opens
  useEffect(() => {
    if (showDisplayFormatDialog) {
      const currentFormat = (outputConfig as any).display_fmt || '';
      setDisplayFormatInput(currentFormat);
      setDisplayFormatError(undefined);
    }
  }, [showDisplayFormatDialog, outputConfig]);

  // Auto-focus input when add column is shown
  useEffect(() => {
    if (showAddColumn && newColumnInputRef.current) {
      newColumnInputRef.current.focus();
    }
  }, [showAddColumn]);

  // Handle type selection
  const handleTypeSelect = useCallback((value: PolarsDataType) => {
    setNewColumnType(value);
  }, []);


  // The output's default row is its own row (`default`), not data[0]; a struct type shows one empty record to start
  const records = useMemo(() => {
    if (isStructMode && outputConfig.output.length === 0) {
      // Create initial default record with all fields
      const defaultRecord: Record<string, any> = { ...outputConfig.default };
      return [defaultRecord];
    }
    return outputConfig.output;
  }, [outputConfig.output, outputConfig.default, isStructMode]);

  // Column order for drag and drop
  const columnOrder = useMemo(() =>
    outputConfig.dtypes.map((_, index) => `col-${index}`)
  , [outputConfig.dtypes]);

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  // Add new column
  const handleAddColumn = useCallback(() => {
    if (!newColumnName.trim()) return;

    // Check for duplicate names
    const exists = outputConfig.dtypes.some(([name]) => name === newColumnName);
    if (exists) {
      alert('Column with this name already exists');
      return;
    }

    const dtype: PolarsDataType = newColumnType;
    let defaultValue: any;

    // Determine default value based on type
    if (typeof dtype === 'string') {
      defaultValue = getDefaultValueForType(dtype);
    } else {
      // Complex type
      switch (dtype.type) {
        case 'Decimal':
          defaultValue = '0';
          break;
        case 'List':
          defaultValue = [];
          break;
        case 'Set':
          defaultValue = [];
          break;
        case 'Array':
          defaultValue = [];
          break;
        case 'Struct':
          defaultValue = {};
          break;
        case 'Categorical':
          defaultValue = '';
          break;
        default:
          defaultValue = '';
      }
    }

    // Add to dtypes (unified schema)
    const newDTypes: OutputSchema = [...outputConfig.dtypes, [newColumnName, dtype]];

    // Update default
    const newDefault = { ...outputConfig.default, [newColumnName]: defaultValue };

    // Add to all existing records
    const newRecords = records.map(record => ({
      ...record,
      [newColumnName]: defaultValue
    }));

    onConfigChange({
      dtypes: newDTypes,
      default: newDefault,
      output: newRecords
    });

    setNewColumnName('');
    setNewColumnType('String');
    setShowAddColumn(false);
  }, [newColumnName, newColumnType, outputConfig, records, onConfigChange]);

  // Rename column
  const handleRenameColumn = useCallback((oldName: string, newName: string) => {
    if (oldName === newName) return;
    if (!newName.trim()) return;

    // Check for duplicate names
    const exists = outputConfig.dtypes.some(([name]) => name === newName && name !== oldName);
    if (exists) {
      alert('Column with this name already exists');
      return;
    }

    // Update dtypes
    const newDTypes: OutputSchema = outputConfig.dtypes.map(([name, type]) =>
      name === oldName ? [newName, type] : [name, type]
    );

    // Update default
    const newDefault = { ...outputConfig.default };
    if (oldName in newDefault) {
      newDefault[newName] = newDefault[oldName];
      delete newDefault[oldName];
    }

    // Update all records
    const newRecords = records.map(record => {
      const newRecord = { ...record };
      if (oldName in newRecord) {
        newRecord[newName] = newRecord[oldName];
        delete newRecord[oldName];
      }
      return newRecord;
    });

    onConfigChange({
      dtypes: newDTypes,
      default: newDefault,
      output: newRecords
    });
  }, [outputConfig, records, onConfigChange]);

  // Change column type
  const handleChangeColumnType = useCallback((columnName: string, newType: PolarsDataType) => {
    const columnIndex = outputConfig.dtypes.findIndex(([name]) => name === columnName);
    if (columnIndex === -1) return;

    const oldType = outputConfig.dtypes[columnIndex][1];
    const oldTypeStr = typeof oldType === 'string' ? oldType : oldType.type;

    // Skip if same type
    if (JSON.stringify(oldType) === JSON.stringify(newType)) return;

    let defaultValue: any;

    // Determine default value based on new type
    if (typeof newType === 'string') {
      const oldValue = outputConfig.default[columnName];
      defaultValue = convertValue(oldValue, oldTypeStr, newType);
    } else {
      // Complex type
      switch (newType.type) {
        case 'Decimal':
          const oldValue = outputConfig.default[columnName];
          defaultValue = convertValue(oldValue, oldTypeStr, 'Decimal');
          break;
        case 'List':
          defaultValue = [];
          break;
        case 'Set':
          defaultValue = [];
          break;
        case 'Array':
          defaultValue = [];
          break;
        case 'Struct':
          defaultValue = {};
          break;
        case 'Categorical':
          defaultValue = '';
          break;
        default:
          defaultValue = '';
      }
    }

    // Update dtypes
    const newDTypes: OutputSchema = [...outputConfig.dtypes];
    newDTypes[columnIndex] = [columnName, newType];

    // Update default
    const newDefault = { ...outputConfig.default };
    newDefault[columnName] = defaultValue;

    // Records keep their values where they convert to the new type
    const newRecords = records.map(record => ({
      ...record,
      [columnName]: typeof newType === 'string' ? convertValue(record[columnName], oldTypeStr, newType) : defaultValue
    }));

    onConfigChange({
      dtypes: newDTypes,
      default: newDefault,
      output: newRecords
    });
  }, [outputConfig, records, onConfigChange]);

  // Remove column
  const handleRemoveColumn = useCallback((columnName: string) => {
    if (!confirm(`Remove column "${columnName}"?`)) return;

    const newDTypes = outputConfig.dtypes.filter(([name]) => name !== columnName);
    const newDefault = { ...outputConfig.default };
    delete newDefault[columnName];

    const newRecords = records.map(record => {
      const { [columnName]: _, ...rest } = record;
      return rest;
    });

    onConfigChange({
      dtypes: newDTypes,
      default: newDefault,
      output: newRecords
    });
  }, [outputConfig, records, onConfigChange]);

  // Reorder columns
  const handleDragEnd = useCallback((event: DragEndEvent) => {
    const { active, over } = event;

    if (!over || active.id === over.id) return;

    const oldIndex = columnOrder.indexOf(active.id as string);
    const newIndex = columnOrder.indexOf(over.id as string);

    // In struct mode, prevent moving the display field (index 0) or moving other fields to position 0
    if (isStructMode && (oldIndex === 0 || newIndex === 0)) {
      return; // Block the drag
    }

    const newDTypes = arrayMove(outputConfig.dtypes, oldIndex, newIndex);

    onConfigChange({
      dtypes: newDTypes
    });
  }, [columnOrder, outputConfig, onConfigChange, isStructMode]);

  // Update cell value
  const handleCellChange = useCallback((recordIndex: number, columnName: string, value: any) => {
    // Validate if validator provided
    if (onValidateCell) {
      const result = onValidateCell(value, columnName, recordIndex);
      const errorKey = `${recordIndex}-${columnName}`;

      if (!result.valid) {
        setValidationErrors(prev => ({
          ...prev,
          [errorKey]: result.error || 'Invalid value'
        }));
        // Still update the value but show error
      } else {
        setValidationErrors(prev => {
          const newErrors = { ...prev };
          delete newErrors[errorKey];
          return newErrors;
        });
      }
    }

    if (recordIndex === -1) {
      onConfigChange({ default: { ...outputConfig.default, [columnName]: value } });
      return;
    }
    const newRecords = [...records];
    newRecords[recordIndex] = {
      ...newRecords[recordIndex],
      [columnName]: value
    };
    onConfigChange({ output: newRecords });
  }, [records, outputConfig.default, onConfigChange, onValidateCell]);

  // Add new data record
  const handleAddRecord = useCallback(() => {
    // Create empty record with default values for each field type
    const newRecord: Record<string, any> = {};
    outputConfig.dtypes.forEach(([fieldName, fieldType]) => {
      if (typeof fieldType === 'string') {
        newRecord[fieldName] = getDefaultValueForType(fieldType);
      } else {
        // Complex type
        switch (fieldType.type) {
          case 'Decimal':
            newRecord[fieldName] = '0';
            break;
          case 'List':
          case 'Set':
          case 'Array':
            newRecord[fieldName] = [];
            break;
          case 'Struct':
          case 'Custom':
            newRecord[fieldName] = null;
            break;
          default:
            newRecord[fieldName] = '';
        }
      }
    });
    onConfigChange({ output: [...records, newRecord] });
  }, [outputConfig.dtypes, records, onConfigChange]);

  // Remove data record
  const handleRemoveRecord = useCallback((index: number) => {
    // Allow deletion immediately - impact analysis will happen at save time
    const newRecords = records.filter((_, i) => i !== index);
    onConfigChange({ output: newRecords });
    onRecordRemoved?.(index);
  }, [records, onConfigChange, onRecordRemoved]);

  // Handle display format change
  const handleDisplayFormatChange = useCallback((value: string) => {
    setDisplayFormatInput(value);
    const error = validateDisplayFormat(value, fieldNames);
    setDisplayFormatError(error);
  }, [fieldNames]);

  // Save display format
  const handleSaveDisplayFormat = useCallback(() => {
    const error = validateDisplayFormat(displayFormatInput, fieldNames);
    if (error) {
      setDisplayFormatError(error);
      return;
    }
    onConfigChange({ display_fmt: displayFormatInput });
    setShowDisplayFormatDialog(false);
  }, [displayFormatInput, fieldNames, onConfigChange]);

  return (
    <>
      <Card>
        <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            {title || (isStructMode ? 'Struct Data & Schema' : 'Output Data & Schema')}
            <Badge variant="outline">{outputConfig.dtypes.length} {isStructMode ? 'fields' : 'columns'}, {records.length} records</Badge>
          </CardTitle>
          <div className="flex gap-2">
            {!showAddColumn && (
              <>
                <Button size="sm" variant="outline" onClick={() => setShowAddColumn(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  {addFieldLabel}
                </Button>
                {!isStructMode && (
                  <Button size="sm" variant="outline" onClick={() => setShowDisplayFormatDialog(true)}>
                    <Settings className="h-4 w-4 mr-2" />
                    Display Format
                  </Button>
                )}
              </>
            )}
            <Button size="sm" variant="outline" onClick={handleAddRecord}>
              <Plus className="h-4 w-4 mr-2" />
              {addRecordLabel}
            </Button>
          </div>
        </div>
        {showAddColumn && (
          <div className="flex items-center gap-2 mt-4 p-3 bg-gray-50 rounded border flex-wrap">
            <Input
              ref={newColumnInputRef}
              placeholder="Column name"
              value={newColumnName}
              onChange={(e) => setNewColumnName(e.target.value)}
              className="w-48"
              onKeyDown={(e) => e.key === 'Enter' && handleAddColumn()}
            />
            <UnifiedTypeSelector
              value={newColumnType}
              onValueChange={handleTypeSelect}
              includeCustomTypes={true}
              customTypeDefs={outputConfig.type_defs}
              customTypeFilter={customTypeFilter}
              showSearch={true}
              showCategories={true}
              showManageButton={true}
              onManageCustomTypes={onManageCustomTypes}
              enableComplexTypes={true}
              className="w-auto min-w-[100px]"
            />
            <Button size="sm" onClick={handleAddColumn}>Add</Button>
            <Button size="sm" variant="ghost" onClick={() => {
              setShowAddColumn(false);
            }}>Cancel</Button>
          </div>
        )}
      </CardHeader>
      <CardContent>
        {outputConfig.dtypes.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-500 mb-4">No columns defined yet</p>
            <Button onClick={() => setShowAddColumn(true)}>Add First Column</Button>
          </div>
        ) : (
          <div className="border rounded-lg overflow-auto min-h-[400px] md:min-h-[500px] lg:min-h-[600px]">
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
              <table className="w-full">
                <thead className="bg-gray-50 border-b">
                  <SortableContext items={columnOrder} strategy={horizontalListSortingStrategy}>
                    <tr>
                      {outputConfig.dtypes.map(([columnName, columnType], index) => (
                        <SortableColumnHeader
                          key={`col-${index}`}
                          id={`col-${index}`}
                          columnName={columnName}
                          columnType={columnType}
                          outputConfig={outputConfig}
                          onRename={(newName) => handleRenameColumn(columnName, newName)}
                          onChangeType={(newType) => handleChangeColumnType(columnName, newType)}
                          onRemove={() => handleRemoveColumn(columnName)}
                          onManageCustomTypes={onManageCustomTypes}
                          customTypeFilter={customTypeFilter}
                          warning={columnWarnings?.[columnName]}
                          isStructMode={isStructMode}
                          isDisplayField={columnName === actualDisplayField}
                          isDraggable={!isStructMode || index !== 0}
                        />
                      ))}
                      <th className="text-center p-3 font-medium text-sm w-20">Actions</th>
                    </tr>
                  </SortableContext>
                </thead>
                <tbody>
                  {(isStructMode
                    ? records.map((record, recordIndex) => [record, recordIndex] as const)
                    : [[outputConfig.default ?? {}, -1] as const, ...records.map((record, recordIndex) => [record, recordIndex] as const)]
                  ).map(([record, recordIndex]) => (
                    <tr key={recordIndex} data-testid={recordIndex === -1 ? 'output-default-row' : undefined}
                        className={`border-b ${recordIndex === -1 ? 'bg-blue-50' : 'hover:bg-gray-50'}`}>
                      {outputConfig.dtypes.map(([columnName, columnType]) => {
                        const errorKey = `${recordIndex}-${columnName}`;
                        const validationError = validationErrors[errorKey];
                        return (
                          <td key={columnName} className="p-2 border-r last:border-r-0">
                            <CellEditor
                              value={(record as Record<string, any>)[columnName]}
                              columnName={columnName}
                              columnType={columnType}
                              outputConfig={outputConfig}
                              onChange={(value) => handleCellChange(recordIndex, columnName, value)}
                              validationError={validationError}
                            />
                          </td>
                        );
                      })}
                      <td className="p-2 text-center">
                        {recordIndex === -1 ? (
                          <Badge variant="outline" className="text-xs">Default</Badge>
                        ) : (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleRemoveRecord(recordIndex)}
                            className="text-red-600 hover:text-red-700 p-1 h-8 w-8"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </DndContext>
          </div>
        )}
      </CardContent>
    </Card>

    {/* Display Format Dialog */}
    <Dialog open={showDisplayFormatDialog} onOpenChange={setShowDisplayFormatDialog}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Configure Display Format</DialogTitle>
          <DialogDescription>
            Define how output records are displayed in leaf node dropdowns.
            Use <code className="text-xs bg-gray-100 px-1 rounded">${`{fieldName}`}</code> syntax to reference fields.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="display-format">Display Format Template</Label>
            <Input
              id="display-format"
              value={displayFormatInput}
              onChange={(e) => handleDisplayFormatChange(e.target.value)}
              placeholder={`e.g., \${${fieldNames[0] || 'field'}}: \${${fieldNames[1] || 'description'}}`}
              className={displayFormatError ? 'border-red-500' : ''}
            />
            {displayFormatError && (
              <p className="text-sm text-red-600">{displayFormatError}</p>
            )}
            <p className="text-xs text-muted-foreground">
              Leave empty to use the first column by default.
            </p>
          </div>

          {fieldNames.length > 0 && (
            <div className="space-y-2">
              <Label className="text-xs">Available Fields:</Label>
              <div className="flex flex-wrap gap-1">
                {fieldNames.map((field) => (
                  <Badge
                    key={field}
                    variant="secondary"
                    className="text-xs cursor-pointer"
                    onClick={() => {
                      const cursorPos = (document.getElementById('display-format') as HTMLInputElement)?.selectionStart || displayFormatInput.length;
                      const newValue = displayFormatInput.slice(0, cursorPos) + `\${${field}}` + displayFormatInput.slice(cursorPos);
                      handleDisplayFormatChange(newValue);
                    }}
                  >
                    ${`{${field}}`}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {outputConfig.output.length > 0 && (
            <div className="space-y-2">
              <Label className="text-xs">Preview:</Label>
              <div className="border rounded p-2 bg-gray-50 max-h-32 overflow-y-auto">
                <div className="space-y-1">
                  {outputConfig.output.slice(0, 3).map((record, idx) => {
                    const preview = formatOutputLabel(record, displayFormatInput || undefined, fieldNames);
                    return (
                      <div key={idx} className="text-xs font-mono bg-white px-2 py-1 rounded border">
                        {preview || <span className="text-muted-foreground italic">(empty)</span>}
                      </div>
                    );
                  })}
                  {outputConfig.output.length > 3 && (
                    <div className="text-xs text-muted-foreground italic">
                      +{outputConfig.output.length - 3} more...
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setShowDisplayFormatDialog(false)}>
            Cancel
          </Button>
          <Button onClick={handleSaveDisplayFormat} disabled={!!displayFormatError}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  </>
  );
};

// Sortable column header
interface SortableColumnHeaderProps {
  id: string;
  columnName: string;
  columnType: PolarsDataType;
  outputConfig: OutputConfiguration;
  onRename: (newName: string) => void;
  onChangeType: (newType: PolarsDataType) => void;
  onRemove: () => void;
  onManageCustomTypes?: () => void;
  customTypeFilter?: (typeDef: UITypeDef, depth: number) => boolean;
  warning?: string;
  isStructMode: boolean;
  isDisplayField: boolean;
  isDraggable: boolean;
}

const SortableColumnHeader: React.FC<SortableColumnHeaderProps> = ({
  id,
  columnName,
  columnType,
  outputConfig,
  onRename,
  onChangeType,
  onRemove,
  onManageCustomTypes,
  customTypeFilter,
  warning,
  isStructMode,
  isDisplayField,
  isDraggable
}) => {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  const [isEditingName, setIsEditingName] = useState(false);
  const [editedName, setEditedName] = useState(columnName);

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const handleNameBlur = useCallback(() => {
    setIsEditingName(false);
    if (editedName.trim() && editedName !== columnName) {
      onRename(editedName.trim());
    } else {
      setEditedName(columnName);
    }
  }, [editedName, columnName, onRename]);

  const handleNameKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleNameBlur();
    } else if (e.key === 'Escape') {
      setEditedName(columnName);
      setIsEditingName(false);
    }
  }, [handleNameBlur, columnName]);

  // Get display type name with custom type resolution
  const typeName = useMemo(() => {
    // Handle string types
    if (typeof columnType === 'string') {
      // Check if it's a custom type reference
      if (columnType.startsWith('custom:')) {
        const typeId = columnType.replace('custom:', '');
        const typeDef = outputConfig.type_defs[typeId];
        if (typeDef) {
          return `${typeDef.name} (${typeDef.type})`;
        }
        return columnType; // Fallback if type def not found
      }
      return columnType;
    }

    // Handle complex types
    if (typeof columnType === 'object' && 'type' in columnType) {
      // Check if it has a type_id (custom type in object format)
      if ('type_id' in columnType) {
        const typeId = (columnType as any).type_id;
        const typeDef = outputConfig.type_defs[typeId];
        if (typeDef) {
          return `${typeDef.name} (${typeDef.type})`;
        }
      }
      return columnType.type;
    }

    return 'Unknown';
  }, [columnType, outputConfig.type_defs]);

  return (
    <th
      ref={setNodeRef}
      style={style}
      className="text-left p-3 font-medium text-sm border-r last:border-r-0 group"
    >
      <div className="flex items-center gap-2">
        {isDraggable ? (
          <GripVertical
            className="h-4 w-4 text-gray-400 cursor-move flex-shrink-0"
            {...attributes}
            {...listeners}
          />
        ) : (
          <div className="h-4 w-4 flex-shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1">
            {isEditingName ? (
              <Input
                value={editedName}
                onChange={(e) => setEditedName(e.target.value)}
                onBlur={handleNameBlur}
                onKeyDown={handleNameKeyDown}
                className="h-7 text-sm font-medium"
                autoFocus
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <div
                className="font-medium cursor-pointer hover:text-blue-600"
                onClick={() => setIsEditingName(true)}
                title="Click to rename"
              >
                {columnName}
              </div>
            )}
            {isStructMode && isDisplayField && (
              <Badge variant="default" className="text-xs bg-blue-600 text-white">
                display
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-1 mt-1 flex-wrap">
            <UnifiedTypeSelector
              value={columnType}
              onValueChange={onChangeType}
              includeCustomTypes={true}
              customTypeDefs={outputConfig.type_defs}
              customTypeFilter={customTypeFilter}
              showSearch={true}
              showCategories={true}
              showManageButton={true}
              onManageCustomTypes={onManageCustomTypes}
              enableComplexTypes={true}
              triggerClassName="h-6 text-xs border-0 p-0 hover:bg-gray-100 w-auto"
            >
              <Badge variant="outline" className="text-xs cursor-pointer">
                {typeName}
              </Badge>
            </UnifiedTypeSelector>
          </div>
          {warning && (
            <p className="text-xs text-amber-700 mt-1 font-normal whitespace-normal">{warning}</p>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onRemove}
          disabled={isStructMode && isDisplayField}
          className={`p-1 h-6 w-6 text-red-600 flex-shrink-0 ${
            isStructMode && isDisplayField
              ? 'opacity-30 cursor-not-allowed'
              : 'opacity-0 group-hover:opacity-100'
          }`}
          title={isStructMode && isDisplayField ? 'Cannot delete display field' : 'Delete field'}
        >
          <Trash2 className="h-3 w-3" />
        </Button>
      </div>
    </th>
  );
};

// Cell editor component - delegates to EnhancedFieldEditor
interface CellEditorProps {
  value: any;
  columnName: string;
  columnType: PolarsDataType;
  onChange: (value: any) => void;
  outputConfig?: OutputConfiguration;
  validationError?: string;
}

const CellEditor: React.FC<CellEditorProps> = ({ value, columnName, columnType, onChange, outputConfig, validationError }) => {
  // Determine if this is a numeric type
  const typeStr = typeof columnType === 'string' ? columnType :
    (typeof columnType === 'object' && 'type' in columnType ? columnType.type : 'String');
  const isNumeric = typeStr.includes('Int') || typeStr.includes('Float');

  // Simply delegate to EnhancedFieldEditor with outputConfig support
  return (
    <div className="relative">
      <EnhancedFieldEditor
        value={value}
        onChange={onChange}
        fieldType={columnType}
        fieldName={columnName}
        outputConfig={outputConfig}
        className={`w-full ${validationError ? 'border-red-500 bg-red-50' : ''}`}
        allowEmpty={!isNumeric}
        allowNull={isNumeric}
        allowInfinity={isNumeric}
      />
      {validationError && (
        <div className="absolute z-10 mt-1 p-2 bg-red-100 border border-red-300 rounded-md shadow-lg text-xs text-red-800 max-w-xs">
          {validationError}
        </div>
      )}
    </div>
  );
};

// Helper function to get default value for a type
function getDefaultValueForType(type: string): any {
  if (type.includes('Int') || type.includes('Float')) return 0;
  if (type === 'Boolean') return false;
  if (type === 'Date') return new Date().toISOString().split('T')[0];
  if (type === 'Datetime') return new Date().toISOString().slice(0, 16);
  if (type === 'Decimal' || type.startsWith('Decimal(')) return '0';
  return '';
}

// Helper function to convert value between types
function convertValue(value: any, fromType: string, toType: string): any {
  // If value is null or undefined, return default for target type
  if (value === null || value === undefined || value === '') {
    return getDefaultValueForType(toType);
  }

  // To String - just stringify everything
  if (toType === 'String') {
    return String(value);
  }

  // To Boolean
  if (toType === 'Boolean') {
    if (typeof value === 'boolean') return value;
    if (typeof value === 'string') {
      const lower = value.toLowerCase().trim();
      if (lower === 'true' || lower === '1' || lower === 'yes') return true;
      if (lower === 'false' || lower === '0' || lower === 'no') return false;
    }
    if (typeof value === 'number') return value !== 0;
    return Boolean(value);
  }

  // To numeric types
  if (toType.includes('Int') || toType.includes('Float')) {
    if (typeof value === 'number') {
      // If converting to Int, floor it
      return toType.includes('Int') ? Math.floor(value) : value;
    }
    if (typeof value === 'boolean') return value ? 1 : 0;
    if (typeof value === 'string') {
      const parsed = parseFloat(value);
      if (!isNaN(parsed)) {
        return toType.includes('Int') ? Math.floor(parsed) : parsed;
      }
    }
    return 0;
  }

  // To Date
  if (toType === 'Date') {
    if (typeof value === 'string') {
      // If it's already ISO date format, use it
      if (/^\d{4}-\d{2}-\d{2}/.test(value)) {
        return value.split('T')[0];
      }
      // Try to parse as date
      const date = new Date(value);
      if (!isNaN(date.getTime())) {
        return date.toISOString().split('T')[0];
      }
    }
    return new Date().toISOString().split('T')[0];
  }

  // To Datetime
  if (toType === 'Datetime') {
    if (typeof value === 'string') {
      // If it's already ISO format, use it
      if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(value)) {
        return value.slice(0, 16);
      }
      // Try to parse as date
      const date = new Date(value);
      if (!isNaN(date.getTime())) {
        return date.toISOString().slice(0, 16);
      }
    }
    return new Date().toISOString().slice(0, 16);
  }

  // To Decimal - just convert to string representation
  if (toType === 'Decimal' || toType.startsWith('Decimal(')) {
    if (typeof value === 'number') {
      return value.toString();
    }
    if (typeof value === 'boolean') {
      return value ? '1' : '0';
    }
    if (typeof value === 'string') {
      const parsed = parseFloat(value);
      if (!isNaN(parsed)) {
        return parsed.toString();
      }
    }
    return '0';
  }

  // Default: return as-is
  return value;
}
