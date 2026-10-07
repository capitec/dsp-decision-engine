import React from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Database, Plus, Trash2, Upload, ChevronUp, ChevronDown } from "lucide-react";
import { UnifiedTypeSelector } from './UnifiedTypeSelector';
import type { PolarsDataType, UITypeDefs } from '../flat-rules/output-types';

// Schema field type: [name, type]
export type SchemaField = [string, PolarsDataType];

export interface SchemaFieldListProps {
  // Data
  fields: SchemaField[];
  onChange: (fields: SchemaField[]) => void;

  // Operations
  allowAdd?: boolean; // Default: true
  allowRemove?: boolean; // Default: true
  allowRename?: boolean; // Default: true
  allowTypeChange?: boolean; // Default: true
  allowReorder?: boolean; // Default: true

  // Constraints
  minFields?: number; // Default: 0
  maxFields?: number; // Default: unlimited

  // Type selector configuration
  availableTypes?: string[];
  excludeTypes?: string[];
  includeCustomTypes?: boolean;
  customTypeDefs?: UITypeDefs;
  showSearch?: boolean; // Show search in type selector

  // External actions
  showLoadFromDataset?: boolean; // Show "Load from Dataset" button
  onLoadFromDataset?: () => void;

  // Display
  emptyStateTitle?: string;
  emptyStateDescription?: string;
  addButtonLabel?: string; // Default: "Add Field"
  className?: string;

  // Container customization
  showCard?: boolean; // Wrap in Card (default: true)
  showHeader?: boolean; // Show header with action buttons (default: true)
  maxHeight?: string; // Max height for scrollable list (e.g., "300px")
}

export const SchemaFieldList: React.FC<SchemaFieldListProps> = ({
  fields,
  onChange,
  allowAdd = true,
  allowRemove = true,
  allowRename = true,
  allowTypeChange = true,
  allowReorder = true,
  minFields = 0,
  maxFields = Number.MAX_SAFE_INTEGER,
  availableTypes,
  excludeTypes,
  includeCustomTypes = false,
  customTypeDefs = {},
  showSearch = false,
  showLoadFromDataset = false,
  onLoadFromDataset,
  emptyStateTitle = "No Fields Defined",
  emptyStateDescription = "Add fields to define your schema",
  addButtonLabel = "Add Field",
  className = "",
  showCard = true,
  showHeader = true,
  maxHeight,
}) => {
  // Operations
  const handleAddField = () => {
    if (fields.length >= maxFields) {
      alert(`Maximum ${maxFields} fields allowed`);
      return;
    }

    const newField: SchemaField = [`field_${fields.length + 1}`, 'String'];
    onChange([...fields, newField]);
  };

  const handleRemoveField = (index: number) => {
    if (fields.length <= minFields) {
      alert(`Minimum ${minFields} fields required`);
      return;
    }

    const newFields = fields.filter((_, i) => i !== index);
    onChange(newFields);
  };

  const handleUpdateFieldName = (index: number, newName: string) => {
    const newFields = [...fields];
    newFields[index] = [newName, newFields[index][1]];
    onChange(newFields);
  };

  const handleUpdateFieldType = (index: number, newType: PolarsDataType) => {
    const newFields = [...fields];
    newFields[index] = [newFields[index][0], newType];
    onChange(newFields);
  };

  const handleMoveField = (index: number, direction: 'up' | 'down') => {
    if (direction === 'up' && index === 0) return;
    if (direction === 'down' && index === fields.length - 1) return;

    const newFields = [...fields];
    const targetIndex = direction === 'up' ? index - 1 : index + 1;
    [newFields[index], newFields[targetIndex]] = [newFields[targetIndex], newFields[index]];
    onChange(newFields);
  };

  // Header with actions
  const headerSection = showHeader && (
    <div className="flex gap-2 mb-4">
      {showLoadFromDataset && onLoadFromDataset && (
        <Button onClick={onLoadFromDataset} variant="outline" size="sm">
          <Upload className="w-4 h-4 mr-2" />
          Load from Dataset
        </Button>
      )}
      {allowAdd && fields.length < maxFields && (
        <Button onClick={handleAddField} size="sm">
          <Plus className="w-4 h-4 mr-2" />
          {addButtonLabel}
        </Button>
      )}
    </div>
  );

  // Empty state
  const emptyState = (
    <div className="text-center py-8">
      <Database className="h-10 w-10 mx-auto text-gray-300 mb-3" />
      <h3 className="text-sm font-medium text-gray-900 mb-1">
        {emptyStateTitle}
      </h3>
      <p className="text-sm text-gray-600 mb-4">
        {emptyStateDescription}
      </p>
      <div className="flex gap-2 justify-center">
        {allowAdd && (
          <Button onClick={handleAddField} variant="outline" size="sm">
            <Plus className="w-4 h-4 mr-2" />
            {addButtonLabel}
          </Button>
        )}
        {showLoadFromDataset && onLoadFromDataset && (
          <Button onClick={onLoadFromDataset} size="sm">
            <Upload className="w-4 h-4 mr-2" />
            Load from Dataset
          </Button>
        )}
      </div>
    </div>
  );

  // Field list
  const fieldList = (
    <div
      className="space-y-2"
      style={maxHeight ? { maxHeight, overflowY: 'auto' } : undefined}
    >
      {fields.map(([fieldName, fieldType], index) => (
        <FieldRow
          key={index}
          index={index}
          fieldName={fieldName}
          fieldType={fieldType}
          allowRename={allowRename}
          allowTypeChange={allowTypeChange}
          allowReorder={allowReorder}
          allowRemove={allowRemove && fields.length > minFields}
          isFirst={index === 0}
          isLast={index === fields.length - 1}
          availableTypes={availableTypes}
          excludeTypes={excludeTypes}
          includeCustomTypes={includeCustomTypes}
          customTypeDefs={customTypeDefs}
          showSearch={showSearch}
          onUpdateName={handleUpdateFieldName}
          onUpdateType={handleUpdateFieldType}
          onMove={handleMoveField}
          onRemove={handleRemoveField}
        />
      ))}
    </div>
  );

  // Content
  const content = (
    <>
      {headerSection}
      {fields.length === 0 ? emptyState : fieldList}
    </>
  );

  // Render with or without card wrapper
  if (showCard) {
    return (
      <Card className={className}>
        <CardContent className="pt-6">
          {content}
        </CardContent>
      </Card>
    );
  }

  return <div className={className}>{content}</div>;
};

// FieldRow sub-component
interface FieldRowProps {
  index: number;
  fieldName: string;
  fieldType: PolarsDataType;
  allowRename: boolean;
  allowTypeChange: boolean;
  allowReorder: boolean;
  allowRemove: boolean;
  isFirst: boolean;
  isLast: boolean;
  availableTypes?: string[];
  excludeTypes?: string[];
  includeCustomTypes: boolean;
  customTypeDefs: UITypeDefs;
  showSearch: boolean;
  onUpdateName: (index: number, name: string) => void;
  onUpdateType: (index: number, type: PolarsDataType) => void;
  onMove: (index: number, direction: 'up' | 'down') => void;
  onRemove: (index: number) => void;
}

const FieldRow: React.FC<FieldRowProps> = ({
  index,
  fieldName,
  fieldType,
  allowRename,
  allowTypeChange,
  allowReorder,
  allowRemove,
  isFirst,
  isLast,
  availableTypes,
  excludeTypes,
  includeCustomTypes,
  customTypeDefs,
  showSearch,
  onUpdateName,
  onUpdateType,
  onMove,
  onRemove
}) => {
  return (
    <div className="flex items-center gap-2 p-2 border rounded-lg">
      {/* Reorder buttons */}
      {allowReorder && (
        <div className="flex flex-col gap-0.5">
          <Button
            variant="ghost"
            size="sm"
            className="h-5 w-5 p-0"
            onClick={() => onMove(index, 'up')}
            disabled={isFirst}
          >
            <ChevronUp className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-5 w-5 p-0"
            onClick={() => onMove(index, 'down')}
            disabled={isLast}
          >
            <ChevronDown className="h-3 w-3" />
          </Button>
        </div>
      )}

      {/* Field name input */}
      <div className="flex-1">
        <Input
          value={fieldName}
          onChange={(e) => onUpdateName(index, e.target.value)}
          placeholder="Field name"
          disabled={!allowRename}
          className="h-9"
        />
      </div>

      {/* Type selector */}
      {allowTypeChange && (
        <UnifiedTypeSelector
          value={fieldType}
          onValueChange={(value) => onUpdateType(index, value)}
          availableTypes={availableTypes}
          excludeTypes={excludeTypes}
          includeCustomTypes={includeCustomTypes}
          customTypeDefs={customTypeDefs}
          showSearch={showSearch}
          enableComplexTypes={true}
          className="w-[140px]"
        />
      )}

      {/* Remove button */}
      {allowRemove && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onRemove(index)}
          className="h-9"
        >
          <Trash2 className="h-4 w-4 text-red-600" />
        </Button>
      )}
    </div>
  );
};
