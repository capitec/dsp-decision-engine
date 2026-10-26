import React from 'react';
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { NumericInput } from "@/components/ui/numeric-input";
import { ChipInput } from "@/components/ui/chip-input";
import { MultiSelect } from "@/components/ui/multi-select";
import type { PolarsDataType } from '../flat-rules/output-types';
import type { OutputConfiguration } from '../flat-rules/output-types';
import { getPolarsTypeName } from '@/types/datasource-types';

/**
 * Enhanced unified field editor component for all cell editing scenarios
 * Consolidates 5 previous implementations:
 * - SharedFieldEditor
 * - CellEditor (UnifiedOutputTable)
 * - StructFieldCellEditor (CustomTypeEditor)
 * - SimpleCellEditor (ParameterManagementTab)
 * - FieldEditor (OutputDataEditor)
 */
interface EnhancedFieldEditorProps {
  // Core props
  value: any;
  onChange: (value: any) => void;
  fieldType: PolarsDataType;
  fieldName?: string;

  // Custom type support
  uiField?: any;
  options?: Array<{value: any, label: string, key?: string}>;
  outputConfig?: OutputConfiguration;

  // Input configuration
  className?: string;
  disabled?: boolean;
  placeholder?: string;
  allowEmpty?: boolean;
  allowNaN?: boolean;
  allowNull?: boolean;
  allowInfinity?: boolean;

  // Validation (NEW - from CustomTypeEditor)
  validator?: (value: any) => boolean;
  validationError?: string;
  showValidationPopup?: boolean;

  // Display (NEW - from CustomTypeEditor)
  isDisplayField?: boolean;
  showFocusState?: boolean;
}

export const EnhancedFieldEditor: React.FC<EnhancedFieldEditorProps> = ({
  value,
  onChange,
  fieldType,
  fieldName,
  uiField,
  options = [],
  outputConfig,
  className = "w-full",
  disabled = false,
  placeholder,
  allowEmpty = true,
  allowNaN = false,
  allowNull = false,
  allowInfinity = false,
  validator,
  validationError,
  showValidationPopup = false,
  isDisplayField = false,
  showFocusState = false,
}) => {
  const renderInput = () => {
    // Get type string for comparisons
    const typeStr = typeof fieldType === 'string' ? fieldType :
      (typeof fieldType === 'object' && 'type' in fieldType ? fieldType.type : 'String');

    // List/Set/Array types - use ChipInput or MultiSelect depending on inner type
    if (typeof fieldType === 'object' && 'type' in fieldType && (fieldType.type === 'List' || fieldType.type === 'Set' || fieldType.type === 'Array')) {
      const arrayValue = Array.isArray(value) ? value : [];
      const innerType = typeof fieldType === 'object' && 'inner' in fieldType ? fieldType.inner : 'String';
      const innerTypeStr = typeof innerType === 'string' ? innerType :
        (typeof innerType === 'object' && 'type' in innerType ? innerType.type : 'String');
      const maxItems = fieldType.type === 'Array' && 'width' in fieldType ? fieldType.width : undefined;
      const isSet = fieldType.type === 'Set'; // Enforce uniqueness for Set types

      // Check if inner type is a custom type (object with type: "Custom" and type_id)
      let customTypeId: string | null = null;
      let customTypeDef: any = null;

      if (typeof innerType === 'object' && 'type' in innerType && innerType.type === 'Custom' && 'type_id' in innerType && outputConfig) {
        customTypeId = (innerType as any).type_id;
        customTypeDef = customTypeId ? outputConfig.type_defs[customTypeId] : null;
      }

      // Check if inner type is a custom categorical
      const isCustomCategorical = customTypeDef && customTypeDef.type === 'categorical';

      if (isCustomCategorical && customTypeDef) {
        // For Set types, use MultiSelect (enforces uniqueness with checkboxes)
        // For List/Array types, use ChipInput with suggestions (allows duplicates)
        if (isSet) {
          return (
            <MultiSelect
              options={customTypeDef.definition.categories}
              values={arrayValue.map(v => String(v))}
              onChange={(stringValues) => {
                onChange(stringValues);
              }}
              placeholder={placeholder || `Select ${customTypeDef.name} values...`}
              className={className}
              disabled={disabled}
              maxItems={maxItems}
            />
          );
        } else {
          // List or Array - allow duplicates
          return (
            <ChipInput
              values={arrayValue.map(v => String(v))}
              onChange={(stringValues) => {
                const limitedValues = maxItems ? stringValues.slice(0, maxItems) : stringValues;
                onChange(limitedValues);
              }}
              placeholder={placeholder || `Add ${customTypeDef.name} values...${maxItems ? ` (max ${maxItems})` : ''}`}
              className={className}
              disabled={disabled}
              suggestions={customTypeDef.definition.categories}
              allowDuplicates={true}
              validator={(v) => {
                if (validator) return validator(v);
                return customTypeDef.definition.categories.includes(v);
              }}
              maxItems={maxItems}
            />
          );
        }
      }

      // Check if inner type is a custom struct
      const isCustomStruct = customTypeDef && customTypeDef.type === 'struct';

      if (isCustomStruct && customTypeDef && customTypeDef.definition.data.length > 0) {
        const displayField = customTypeDef.definition.display_field;
        const displayValues = customTypeDef.definition.data.map((record: any) => String(record[displayField]));

        // Extract display values from $key references or legacy full objects
        const currentDisplayValues = arrayValue.map((v: any) => {
          if (typeof v === 'object' && v !== null) {
            return v.$key !== undefined ? String(v.$key) : String(v[displayField] || '');
          }
          return String(v);
        });

        // For Set types, use MultiSelect (enforces uniqueness with checkboxes)
        // For List/Array types, use ChipInput with suggestions (allows duplicates)
        if (isSet) {
          return (
            <MultiSelect
              options={displayValues}
              values={currentDisplayValues}
              onChange={(selectedDisplayValues) => {
                // Look up original records to preserve display field types
                const selectedRefs = selectedDisplayValues.map(displayValue => {
                  const originalRecord = customTypeDef.definition.data.find(
                    (record: any) => String(record[displayField]) === displayValue
                  );
                  const typedValue = originalRecord ? originalRecord[displayField] : displayValue;
                  return { $key: typedValue };
                });
                onChange(selectedRefs);
              }}
              placeholder={placeholder || `Select ${customTypeDef.name} values...`}
              className={className}
              disabled={disabled}
              maxItems={maxItems}
            />
          );
        } else {
          // List or Array - allow duplicates
          return (
            <ChipInput
              values={currentDisplayValues}
              onChange={(selectedDisplayValues) => {
                const limitedValues = maxItems ? selectedDisplayValues.slice(0, maxItems) : selectedDisplayValues;
                // Look up original records to preserve display field types
                const selectedRefs = limitedValues.map(displayValue => {
                  const originalRecord = customTypeDef.definition.data.find(
                    (record: any) => String(record[displayField]) === displayValue
                  );
                  const typedValue = originalRecord ? originalRecord[displayField] : displayValue;
                  return { $key: typedValue };
                });
                onChange(selectedRefs);
              }}
              placeholder={placeholder || `Add ${customTypeDef.name} values...${maxItems ? ` (max ${maxItems})` : ''}`}
              className={className}
              disabled={disabled}
              suggestions={displayValues}
              allowDuplicates={true}
              maxItems={maxItems}
            />
          );
        }
      }

      // Check if inner type is Enum (not a custom type, but a direct Enum object)
      const isEnum = typeof innerType === 'object' && 'type' in innerType &&
        innerType.type === 'Enum' && 'categories' in innerType;

      if (isEnum) {
        const enumType = innerType as any;
        const categories = enumType.categories || [];

        // For Set types, use MultiSelect (enforces uniqueness with checkboxes)
        // For List/Array types, use ChipInput with suggestions (allows duplicates)
        if (isSet) {
          return (
            <MultiSelect
              options={categories}
              values={arrayValue.map(v => String(v))}
              onChange={(stringValues) => {
                onChange(stringValues);
              }}
              placeholder={placeholder || `Select values...`}
              className={className}
              disabled={disabled}
              maxItems={maxItems}
            />
          );
        } else {
          // List or Array - allow duplicates
          return (
            <ChipInput
              values={arrayValue.map(v => String(v))}
              onChange={(stringValues) => {
                const limitedValues = maxItems ? stringValues.slice(0, maxItems) : stringValues;
                onChange(limitedValues);
              }}
              placeholder={placeholder || `Add values...${maxItems ? ` (max ${maxItems})` : ''}`}
              className={className}
              disabled={disabled}
              suggestions={categories}
              allowDuplicates={true}
              validator={(v) => {
                if (validator) return validator(v);
                return categories.includes(v);
              }}
              maxItems={maxItems}
            />
          );
        }
      }

      // Standard list/set/array of primitive types - use ChipInput for free-form entry
      const isNumericList = innerTypeStr.includes('Int') || innerTypeStr.includes('Float');

      return (
        <ChipInput
          values={arrayValue.map(v => String(v))}
          onChange={(stringValues) => {
            // Enforce uniqueness for Set types
            const uniqueValues = isSet ? Array.from(new Set(stringValues)) : stringValues;
            const limitedValues = maxItems ? uniqueValues.slice(0, maxItems) : uniqueValues;

            if (isNumericList) {
              const numericValues = limitedValues.map(v => {
                const num = Number(v);
                return isNaN(num) ? 0 : (innerTypeStr.includes('Int') ? Math.floor(num) : num);
              });
              // Enforce uniqueness again for numeric Set types
              const finalValues = isSet ? Array.from(new Set(numericValues)) : numericValues;
              onChange(finalValues);
            } else {
              onChange(limitedValues);
            }
          }}
          placeholder={placeholder || `Add ${innerTypeStr} values...${isSet ? ' (unique)' : ''}${maxItems ? ` (max ${maxItems})` : ''}`}
          className={className}
          disabled={disabled}
          allowDuplicates={!isSet} // List/Array allow duplicates, Set doesn't
          validator={(v) => {
            if (validator) return validator(v);
            if (isNumericList) return !isNaN(Number(v));
            return v.trim().length > 0;
          }}
          maxItems={maxItems}
        />
      );
    }

    // Custom types - check if it's a struct or categorical type
    if (outputConfig) {
      let customTypeDef: any = null;

      // Check for Custom type format: {type: 'Custom', type_id: 'uuid'}
      if (typeStr === 'Custom' && typeof fieldType === 'object' && 'type_id' in fieldType) {
        const explicitType = fieldType as any;
        customTypeDef = outputConfig.type_defs[explicitType.type_id];
      }

      // Handle struct custom types
      if (customTypeDef && customTypeDef.type === 'struct' && customTypeDef.definition.data.length > 0) {
        const displayField = customTypeDef.definition.display_field;
        // Extract display value from $key reference or legacy full object
        const currentDisplayValue = value && typeof value === 'object'
          ? (value.$key !== undefined ? String(value.$key) : String(value[displayField]) || '')
          : '';

        return (
          <Select
            value={currentDisplayValue}
            onValueChange={(displayValue) => {
              // Look up the original record to preserve the display field's type
              const originalRecord = customTypeDef.definition.data.find(
                (record: any) => String(record[displayField]) === displayValue
              );
              const typedValue = originalRecord ? originalRecord[displayField] : displayValue;
              onChange({ $key: typedValue });
            }}
            disabled={disabled}
          >
            <SelectTrigger className={className}>
              <SelectValue placeholder={placeholder || "Select..."} />
            </SelectTrigger>
            <SelectContent>
              {customTypeDef.definition.data.map((record: any, idx: number) => {
                const displayValue = String(record[displayField]);
                return (
                  <SelectItem key={idx} value={displayValue}>
                    {displayValue}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        );
      }

      // Handle categorical custom types
      if (customTypeDef && customTypeDef.type === 'categorical' && customTypeDef.definition.categories.length > 0) {
        const currentValue = value !== null && value !== undefined ? String(value) : '';

        return (
          <Select
            value={currentValue}
            onValueChange={(selectedValue) => {
              onChange(selectedValue);
            }}
            disabled={disabled}
          >
            <SelectTrigger className={className}>
              <SelectValue placeholder={placeholder || "Select..."} />
            </SelectTrigger>
            <SelectContent>
              {customTypeDef.definition.categories.map((category: string, idx: number) => (
                <SelectItem key={idx} value={category}>
                  {category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      }

      // Fallback: show JSON for custom type without data (only if we detected it's a custom type)
      if (customTypeDef && customTypeDef.type === 'struct') {
        return (
          <Input
            value={JSON.stringify(value || {})}
            onChange={(e) => {
              try {
                const parsed = JSON.parse(e.target.value);
                onChange(parsed);
              } catch {
                // Invalid JSON, ignore
              }
            }}
            className={`${className} font-mono text-xs`}
            placeholder="{}"
            disabled={disabled}
          />
        );
      }
    }

    // Handle struct types (custom types with dropdowns) - from SharedFieldEditor/OutputDataEditor
    if (uiField?.type === 'struct' && options.length > 0) {
      const selectedOption = options.find(opt =>
        JSON.stringify(opt.value) === JSON.stringify(value)
      );

      return (
        <Select
          value={selectedOption?.key || ''}
          onValueChange={(key) => {
            const option = options.find(opt => opt.key === key);
            onChange(option?.value || {});
          }}
          disabled={disabled}
        >
          <SelectTrigger className={className}>
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent>
            {options.map((option) => (
              <SelectItem key={option.key || 'unknown'} value={option.key || 'unknown'}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    }

    // Categorical types - use dropdown with custom categories
    // Handle both object format {type: 'Categorical', type_id: 'uuid'} and string format "custom:uuid"
    if (outputConfig) {
      let categoricalTypeDef: any = null;

      // Check object format
      if (typeStr === 'Categorical' && typeof fieldType === 'object' && 'type_id' in fieldType) {
        const explicitType = fieldType as any;
        categoricalTypeDef = outputConfig.type_defs[explicitType.type_id];
      }
      // Check string format "custom:uuid"
      else if (typeof fieldType === 'string' && fieldType.startsWith('custom:')) {
        const typeId = fieldType.replace('custom:', '');
        const typeDef = outputConfig.type_defs[typeId];
        if (typeDef && typeDef.type === 'categorical') {
          categoricalTypeDef = typeDef;
        }
      }

      if (categoricalTypeDef && categoricalTypeDef.type === 'categorical') {
        return (
          <Select value={String(value || '')} onValueChange={onChange} disabled={disabled}>
            <SelectTrigger className={className}>
              <SelectValue placeholder={placeholder || "Select..."} />
            </SelectTrigger>
            <SelectContent>
              {categoricalTypeDef.definition.categories.map((category: string, idx: number) => (
                <SelectItem key={idx} value={category}>
                  {category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      }
    }

    // Handle categorical types (from SharedFieldEditor/OutputDataEditor)
    if (uiField?.type === 'categorical' && options.length > 0) {
      return (
        <Select value={value || ''} onValueChange={onChange} disabled={disabled}>
          <SelectTrigger className={className}>
            <SelectValue placeholder={placeholder} />
          </SelectTrigger>
          <SelectContent>
            {options.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    }

    // Decimal types - validated string input (from UnifiedOutputTable)
    if (typeStr === 'Decimal' || (typeof fieldType === 'object' && 'type' in fieldType && fieldType.type === 'Decimal')) {
      const scale = typeof fieldType === 'object' && 'scale' in fieldType ? fieldType.scale : 2;
      const placeholderText = placeholder || '0.' + '0'.repeat(scale);

      return (
        <Input
          type="text"
          value={value || ''}
          onChange={(e) => {
            const val = e.target.value;
            // Allow typing decimal numbers (and negative)
            if (/^-?\d*\.?\d*$/.test(val) || val === '') {
              onChange(val);
            }
          }}
          className={`${className} font-mono`}
          placeholder={placeholderText}
          disabled={disabled}
        />
      );
    }

    // Handle boolean types
    if (fieldType === 'Boolean' || typeStr === 'Boolean') {
      return (
        <Select
          value={value?.toString() || 'false'}
          onValueChange={(val) => onChange(val === 'true')}
          disabled={disabled}
        >
          <SelectTrigger className={className}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="true">True</SelectItem>
            <SelectItem value="false">False</SelectItem>
          </SelectContent>
        </Select>
      );
    }

    // Handle date types
    if (fieldType === 'Date' || typeStr === 'Date') {
      return (
        <Input
          type="date"
          value={value || ''}
          onChange={(e) => onChange(e.target.value)}
          className={className}
          disabled={disabled}
        />
      );
    }

    // Handle datetime types
    if (getPolarsTypeName(fieldType) === 'Datetime' || typeStr === 'Datetime' || typeStr.includes('Datetime')) {
      return (
        <Input
          type="datetime-local"
          value={value || ''}
          onChange={(e) => onChange(e.target.value)}
          className={className}
          disabled={disabled}
        />
      );
    }

    // Handle numeric types (with NaN/null/Infinity support)
    if (typeof fieldType === 'string' && (fieldType.includes('Int') || fieldType.includes('Float')) ||
        typeStr.includes('Int') || typeStr.includes('Float')) {
      const isFloat = (typeof fieldType === 'string' && fieldType.includes('Float')) || typeStr.includes('Float');
      return (
        <NumericInput
          value={value}
          onChange={onChange}
          className={className}
          step={isFloat ? 'any' : '1'}
          placeholder={placeholder || (allowEmpty ? 'null' : '0')}
          allowEmpty={allowEmpty}
          allowNaN={allowNaN}
          allowNull={allowNull}
          allowInfinity={allowInfinity}
          disabled={disabled}
        />
      );
    }

    // Handle string and other types (default)
    return (
      <Input
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value)}
        className={className}
        placeholder={placeholder || `Enter ${fieldName || 'value'}...`}
        disabled={disabled}
      />
    );
  };

  const input = renderInput();

  // Add validation error display if needed (from CustomTypeEditor)
  if (validationError && showValidationPopup) {
    return (
      <div className="relative">
        {input}
        <div className="absolute z-10 mt-1 p-2 bg-red-100 border border-red-300 rounded-md shadow-lg text-xs text-red-800 max-w-48">
          {validationError}
        </div>
      </div>
    );
  }

  // Add display field highlighting if needed (from CustomTypeEditor)
  if (isDisplayField && !validationError) {
    const element = input as React.ReactElement<any>;
    return React.cloneElement(element, {
      className: `${element.props.className || ''} border-blue-300 bg-blue-50`.trim()
    });
  }

  return input;
};
