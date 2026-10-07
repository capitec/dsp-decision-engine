import React from 'react';
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { NumericInput } from "@/components/ui/numeric-input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ChipInput } from "@/components/ui/chip-input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, X } from "lucide-react";
import { ProjectOutputField, CustomType, EnumType, RecordType } from '@/types/output-field-types';

export interface FieldEditorProps {
  field: ProjectOutputField;
  value: any;
  onChange: (value: any) => void;
  variant?: 'full' | 'compact'; // full = TypedLeafNodeEditor style, compact = DefaultValueEditor style
  onUpdateType?: (fieldId: string) => Promise<void>; // For missing type updates
  className?: string;
}

export const FieldEditor: React.FC<FieldEditorProps> = ({ 
  field, 
  value, 
  onChange, 
  variant = 'full',
  onUpdateType,
  className = '' 
}) => {
  const isCompact = variant === 'compact';

  // Debug logging to see what field data is received
  console.log(`🎯 FieldEditor (${variant}) for field "${field.field_name}":`, {
    field_type: field.field_type,
    list_type: field.list_type,
    custom_type_id: field.custom_type_id,
    has_custom_type: !!field.custom_type,
    custom_type_name: field.custom_type?.name,
    custom_type_kind: field.custom_type?.type_kind,
    value: value
  });

  // Render primitive string field
  if (field.field_type === 'string') {
    if (isCompact) {
      return (
        <div className={`flex items-center gap-3 ${className}`}>
          <Label className="w-24 text-xs flex-shrink-0">
            {field.field_name}
          </Label>
          <Input
            value={value || ''}
            onChange={(e) => onChange(e.target.value)}
            placeholder={`Default ${field.field_name}`}
            className="text-xs"
          />
        </div>
      );
    } else {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label>
            {field.field_name}
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <Input
            value={value || ''}
            onChange={(e) => onChange(e.target.value)}
            placeholder={`Enter ${field.field_name}`}
          />
        </div>
      );
    }
  }

  // Render primitive number field
  if (field.field_type === 'number') {
    if (isCompact) {
      return (
        <div className={`flex items-center gap-3 ${className}`}>
          <Label className="w-24 text-xs flex-shrink-0">
            {field.field_name}
          </Label>
          <NumericInput
            value={value}
            onChange={onChange}
            placeholder={`Default ${field.field_name}`}
            className="text-xs"
            allowEmpty={true}
            allowInfinity={true}
            allowNaN={true}
          />
        </div>
      );
    } else {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label>
            {field.field_name}
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <NumericInput
            value={value}
            onChange={onChange}
            placeholder={`Enter ${field.field_name} (supports inf, nan)`}
            allowEmpty={true}
            allowInfinity={true}
            allowNaN={true}
          />
        </div>
      );
    }
  }

  // Render primitive boolean field
  if (field.field_type === 'boolean') {
    if (isCompact) {
      return (
        <div className={`flex items-center gap-3 ${className}`}>
          <Label className="w-24 text-xs flex-shrink-0">
            {field.field_name}
          </Label>
          <Switch
            checked={Boolean(value)}
            onCheckedChange={(checked) => onChange(checked)}
          />
        </div>
      );
    } else {
      return (
        <div className={`flex items-center space-x-2 ${className}`}>
          <Switch
            checked={Boolean(value)}
            onCheckedChange={(checked) => onChange(checked)}
          />
          <Label>
            {field.field_name}
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
        </div>
      );
    }
  }

  // Handle list of strings
  if (field.field_type === 'list' && field.list_type === 'string') {
    const arrayValue = Array.isArray(value) ? value.map(v => String(v)) : [];
    
    if (isCompact) {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label className="text-xs">
            {field.field_name} (List)
          </Label>
          <ChipInput
            values={arrayValue}
            onChange={(newValues) => onChange(newValues)}
            placeholder="Type and press Enter to add values"
            validator={(value) => value.trim().length > 0}
          />
        </div>
      );
    } else {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label>
            {field.field_name}
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <ChipInput
            values={arrayValue}
            onChange={(newValues) => onChange(newValues)}
            placeholder="Type and press Enter to add values"
            validator={(value) => value.trim().length > 0}
          />
        </div>
      );
    }
  }

  // Handle list of numbers
  if (field.field_type === 'list' && field.list_type === 'number') {
    const arrayValue = Array.isArray(value) ? value.map(v => String(v)) : [];
    
    const handleNumberListChange = (newStringValues: string[]) => {
      const numberValues = newStringValues
        .map(v => parseFloat(v))
        .filter(n => !isNaN(n));
      onChange(numberValues);
    };

    if (isCompact) {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label className="text-xs">
            {field.field_name} (Number List)
          </Label>
          <ChipInput
            values={arrayValue}
            onChange={handleNumberListChange}
            placeholder="Type numbers and press Enter"
            validator={(value) => !isNaN(parseFloat(value.trim()))}
            formatter={(value) => {
              const num = parseFloat(value.trim());
              return isNaN(num) ? value : num.toString();
            }}
          />
        </div>
      );
    } else {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label>
            {field.field_name}
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <ChipInput
            values={arrayValue}
            onChange={handleNumberListChange}
            placeholder="Type numbers and press Enter"
            validator={(value) => !isNaN(parseFloat(value.trim()))}
            formatter={(value) => {
              const num = parseFloat(value.trim());
              return isNaN(num) ? value : num.toString();
            }}
          />
        </div>
      );
    }
  }

  // Handle list of custom types (enums)
  if (field.field_type === 'list' && field.list_type === 'custom' && field.custom_type?.type_kind === 'enum') {
    const customType = field.custom_type;
    const enumType = customType as EnumType;
    const arrayValue = Array.isArray(value) ? value : [];

    const handleEnumListChange = (newValue: string) => {
      if (newValue !== '__ADD_VALUE__' && !arrayValue.includes(newValue)) {
        const newArray = [...arrayValue, newValue];
        onChange(newArray);
      }
    };

    const removeEnumValue = (idx: number) => {
      const newArray = arrayValue.filter((_, i) => i !== idx);
      onChange(newArray);
    };

    if (isCompact) {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label className="text-xs">
            {field.field_name} (Enum List)
          </Label>
          <div className="flex flex-wrap gap-1 min-h-[32px] p-2 border rounded bg-white text-xs">
            {arrayValue.length === 0 ? (
              <span className="text-gray-400">No values selected</span>
            ) : (
              arrayValue.map((val: any, idx: number) => (
                <Badge key={idx} variant="secondary" className="text-xs">
                  {val}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-auto w-auto p-0 ml-1"
                    onClick={() => removeEnumValue(idx)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              ))
            )}
          </div>
          <Select value="__ADD_VALUE__" onValueChange={handleEnumListChange}>
            <SelectTrigger className="text-xs">
              <SelectValue placeholder="Add enum value" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__ADD_VALUE__" disabled>Add enum value</SelectItem>
              {(enumType.definition?.values || [])
                .filter(enumValue => !arrayValue.includes(enumValue))
                .map((enumValue) => (
                  <SelectItem key={enumValue} value={enumValue} className="text-xs">
                    {enumValue}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>
      );
    } else {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label>
            {field.field_name} (List of {customType.name})
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <div className="flex flex-wrap gap-1 min-h-[40px] p-2 border rounded bg-white">
            {arrayValue.length === 0 ? (
              <span className="text-gray-400 text-sm">No values selected</span>
            ) : (
              arrayValue.map((val, idx) => (
                <Badge key={idx} variant="secondary" className="text-xs">
                  {val}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-auto w-auto p-0 ml-1"
                    onClick={() => removeEnumValue(idx)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              ))
            )}
          </div>
          <Select value="__ADD_VALUE__" onValueChange={handleEnumListChange}>
            <SelectTrigger>
              <SelectValue placeholder="Add enum value" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__ADD_VALUE__" disabled>Add enum value</SelectItem>
              {(enumType.definition?.values || [])
                .filter(enumValue => !arrayValue.includes(enumValue))
                .map((enumValue) => (
                  <SelectItem key={enumValue} value={enumValue}>
                    {enumValue}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
          <div className="text-xs text-gray-500">
            Select multiple enum values from {customType.name}
          </div>
        </div>
      );
    }
  }

  // Handle list of custom types (records)
  if (field.field_type === 'list' && field.list_type === 'custom' && field.custom_type?.type_kind === 'record') {
    const customType = field.custom_type;
    const recordType = customType as RecordType;
    // Handle both flattened and nested structure during transition
    const records = recordType.definition?.records || [];
    const arrayValue = Array.isArray(value) ? value : [];

    const handleRecordListChange = (newValue: string) => {
      if (newValue !== '__ADD_VALUE__') {
        const recordId = parseInt(newValue);
        if (!arrayValue.includes(recordId)) {
          const newArray = [...arrayValue, recordId];
          onChange(newArray);
        }
      }
    };

    const removeRecordValue = (idx: number) => {
      const newArray = arrayValue.filter((_, i) => i !== idx);
      onChange(newArray);
    };

    // Helper function to get display text for a record
    const getRecordDisplayText = (recordId: string | number) => {
      const record = records.find(r => r.id === Number(recordId));
      if (!record) return `Record ${recordId}`;
      
      if (recordType.display_format) {
        return recordType.display_format.replace(/\{(\w+)\}/g, (match, fieldName) => {
          return record[fieldName] !== undefined ? String(record[fieldName]) : match;
        });
      }
      return `Record ${record.id}`;
    };

    if (isCompact) {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label className="text-xs">
            {field.field_name} (Record List)
          </Label>
          <div className="flex flex-wrap gap-1 min-h-[32px] p-2 border rounded bg-white text-xs">
            {arrayValue.length === 0 ? (
              <span className="text-gray-400">No records selected</span>
            ) : (
              arrayValue.map((recordId: number, idx: number) => (
                <Badge key={idx} variant="secondary" className="text-xs">
                  {getRecordDisplayText(recordId)}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-auto w-auto p-0 ml-1"
                    onClick={() => removeRecordValue(idx)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              ))
            )}
          </div>
          <Select value="__ADD_VALUE__" onValueChange={handleRecordListChange}>
            <SelectTrigger className="text-xs">
              <SelectValue placeholder="Add record" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__ADD_VALUE__" disabled>Add record</SelectItem>
              {records
                .filter(record => !arrayValue.includes(record.id))
                .map((record) => {
                  const displayText = getRecordDisplayText(record.id as number);
                  return (
                    <SelectItem key={record.id} value={record.id.toString()} className="text-xs">
                      {displayText}
                    </SelectItem>
                  );
                })}
            </SelectContent>
          </Select>
        </div>
      );
    } else {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label>
            {field.field_name} (List of {customType.name})
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <div className="flex flex-wrap gap-1 min-h-[40px] p-2 border rounded bg-white">
            {arrayValue.length === 0 ? (
              <span className="text-gray-400 text-sm">No records selected</span>
            ) : (
              arrayValue.map((recordId: number, idx: number) => (
                <Badge key={idx} variant="secondary" className="text-xs">
                  {getRecordDisplayText(recordId)}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-auto w-auto p-0 ml-1"
                    onClick={() => removeRecordValue(idx)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                </Badge>
              ))
            )}
          </div>
          <Select value="__ADD_VALUE__" onValueChange={handleRecordListChange}>
            <SelectTrigger>
              <SelectValue placeholder="Add record" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__ADD_VALUE__" disabled>Add record</SelectItem>
              {records
                .filter(record => !arrayValue.includes(record.id))
                .map((record) => {
                  const displayText = getRecordDisplayText(record.id as number);
                  return (
                    <SelectItem key={record.id} value={record.id.toString()}>
                      {displayText}
                    </SelectItem>
                  );
                })}
            </SelectContent>
          </Select>
          <div className="text-xs text-gray-500">
            Select multiple records from {customType.name}
          </div>
        </div>
      );
    }
  }

  // Handle missing custom type definition
  if (field.field_type === 'custom' && !field.custom_type) {
    return (
      <div className={`space-y-2 ${className}`}>
        <Label className="text-orange-600">
          {field.field_name} (Type definition missing)
        </Label>
        <div className="flex items-center gap-2 p-2 bg-orange-50 border border-orange-200 rounded">
          <AlertTriangle className="h-4 w-4 text-orange-600" />
          <div className="flex-1 text-sm text-orange-700">
            This field's custom type definition needs to be updated.
          </div>
          {onUpdateType && (
            <Button
              size="sm"
              variant="outline"
              className="text-orange-700 border-orange-300 hover:bg-orange-100"
              onClick={() => onUpdateType(field.id)}
            >
              Update Type
            </Button>
          )}
        </div>
      </div>
    );
  }

  // Handle single custom type (enum)
  if (field.field_type === 'custom' && field.custom_type?.type_kind === 'enum') {
    const customType = field.custom_type;
    const enumType = customType as EnumType;

    if (isCompact) {
      return (
        <div className={`flex items-center gap-3 ${className}`}>
          <Label className="w-24 text-xs flex-shrink-0">
            {field.field_name}
          </Label>
          <Select
            value={value || ''}
            onValueChange={(newValue) => onChange(newValue)}
          >
            <SelectTrigger className="text-xs">
              <SelectValue placeholder={`Select ${field.field_name}`} />
            </SelectTrigger>
            <SelectContent>
              {(enumType.definition?.values || []).map((enumValue) => (
                <SelectItem key={enumValue} value={enumValue} className="text-xs">
                  {enumValue}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      );
    } else {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label>
            {field.field_name}
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <Select
            value={value || ''}
            onValueChange={(newValue) => onChange(newValue)}
          >
            <SelectTrigger>
              <SelectValue placeholder={`Select ${field.field_name}`} />
            </SelectTrigger>
            <SelectContent>
              {(enumType.definition?.values || []).map((enumValue) => (
                <SelectItem key={enumValue} value={enumValue}>
                  {enumValue}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      );
    }
  }

  // Handle single custom type (record)
  if (field.field_type === 'custom' && field.custom_type?.type_kind === 'record') {
    const customType = field.custom_type;
    const recordType = customType as RecordType;
    // Handle both flattened and nested structure during transition
    const records = recordType.definition?.records || [];

    const handleRecordChange = (recordId: string) => {
      // Store just the ID instead of the full record object
      onChange(recordId ? parseInt(recordId) : null);
    };

    // Resolve the stored ID to the actual record object for display
    const selectedRecord = React.useMemo(() => {
      if (value === null || value === undefined) return null;
      return records.find(r => r.id === value) || null;
    }, [value, records]);

    if (isCompact) {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label className="text-xs">
            {field.field_name} (Record)
          </Label>
          <Select
            value={value?.toString() || ''}
            onValueChange={handleRecordChange}
          >
            <SelectTrigger className="text-xs">
              <SelectValue placeholder={`Select ${field.field_name}`} />
            </SelectTrigger>
            <SelectContent>
              {records.map((record) => {
                let displayText = `ID: ${record.id}`;
                if (recordType.display_format) {
                  displayText = recordType.display_format.replace(/\{(\w+)\}/g, (match, fieldName) => {
                    return record[fieldName] !== undefined ? String(record[fieldName]) : match;
                  });
                }
                return (
                  <SelectItem key={record.id} value={record.id.toString()} className="text-xs">
                    {displayText}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
          {selectedRecord && (
            <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
              Selected: {JSON.stringify(selectedRecord, null, 2)}
            </div>
          )}
        </div>
      );
    } else {
      return (
        <div className={`space-y-2 ${className}`}>
          <Label>
            {field.field_name}
            {field.is_required && <span className="text-red-500 ml-1">*</span>}
          </Label>
          <Select
            value={value?.toString() || ''}
            onValueChange={handleRecordChange}
          >
            <SelectTrigger>
              <SelectValue placeholder={`Select ${field.field_name}`} />
            </SelectTrigger>
            <SelectContent>
              {records.map((record) => {
                let displayText = `ID: ${record.id}`;
                if (recordType.display_format) {
                  displayText = recordType.display_format.replace(/\{(\w+)\}/g, (match, fieldName) => {
                    return record[fieldName] !== undefined ? String(record[fieldName]) : match;
                  });
                }
                return (
                  <SelectItem key={record.id} value={record.id.toString()}>
                    <div className="flex flex-col">
                      <span>{displayText}</span>
                      <span className="text-xs text-gray-500">ID: {record.id}</span>
                    </div>
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
          {selectedRecord && (
            <div className="text-xs text-gray-500">
              Selected: {JSON.stringify(selectedRecord, null, 2)}
            </div>
          )}
        </div>
      );
    }
  }

  // Fallback for unsupported field types
  if (isCompact) {
    return (
      <div className={`flex items-center gap-3 ${className}`}>
        <Label className="w-24 text-xs flex-shrink-0 text-orange-600">
          {field.field_name}
        </Label>
        <div className="text-xs text-orange-600">
          Unsupported field type: {field.field_type}_{field.list_type || 'undefined'}
        </div>
      </div>
    );
  } else {
    return (
      <div className={`space-y-2 ${className}`}>
        <Label className="text-orange-600">
          {field.field_name} (Unsupported type)
        </Label>
        <div className="text-sm text-orange-600 p-2 bg-orange-50 border border-orange-200 rounded">
          Unsupported field type: {field.field_type}
          {field.list_type && ` with list_type: ${field.list_type}`}
        </div>
      </div>
    );
  }
};