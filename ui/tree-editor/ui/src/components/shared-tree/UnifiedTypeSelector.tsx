import React, { useState, useMemo } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search, Settings } from "lucide-react";
import { COMMON_POLARS_TYPES } from '../flat-rules/output-types';
import type { UITypeDef, UITypeDefs } from '../flat-rules/output-types';
import type { PolarsType } from '@/types/datasource-types';
import { getPolarsTypeName, isPolarsComplexType, isCustomType, getCustomTypeId } from '@/types/datasource-types';

// Most commonly used types
const MOST_COMMON_TYPES = [
  'String',
  'Boolean',
  'Int64',
  'Float64',
  'Decimal',
  'List',
  'Date',
  'Datetime',
  'Int32',
  'Float32',
  'UInt64',
];

export interface UnifiedTypeSelectorProps {
  // Value - now properly typed as PolarsType
  value: PolarsType;
  onValueChange: (type: PolarsType) => void;

  // Type options
  availableTypes?: string[]; // Default: COMMON_POLARS_TYPES
  excludeTypes?: string[]; // Types to exclude
  includeCustomTypes?: boolean; // Include types from type_defs
  customTypeDefs?: UITypeDefs; // Custom type definitions
  customTypeFilter?: (typeDef: UITypeDef, depth: number) => boolean; // Hide custom types (depth 0 = the value itself, 1 = a List/Set/Array item)

  // UI features
  showSearch?: boolean; // Show search bar
  showCategories?: boolean; // Group into Primitive/Custom
  showManageButton?: boolean; // Show "Manage Custom Types" button
  onManageCustomTypes?: () => void;

  // Complex type configuration
  enableComplexTypes?: boolean; // Allow configuration of Decimal/List/Array types
  maxNestingDepth?: number; // Prevent infinite recursion for Lists (default 3)
  currentDepth?: number; // Internal tracking for recursion

  // Display
  className?: string;
  disabled?: boolean;
  placeholder?: string;
  triggerClassName?: string; // Custom className for SelectTrigger
  children?: React.ReactNode; // Custom trigger content (e.g., Badge)
}

/**
 * UnifiedTypeSelector - A comprehensive type selector component that consolidates
 * all type selection patterns across the flat-rules UI.
 *
 * Now properly supports PolarsType (both simple strings and complex objects).
 *
 * Features:
 * - Search functionality for filtering types
 * - Category grouping (Primitive/Custom types)
 * - Custom type support (struct/categorical)
 * - Type exclusion filtering
 * - Inline Decimal configuration (precision, scale)
 * - Recursive List inner type configuration
 * - "Manage Custom Types" button
 *
 * @example
 * // Simple usage
 * <UnifiedTypeSelector
 *   value="Int64"
 *   onValueChange={handleTypeChange}
 * />
 *
 * @example
 * // Complex type usage (Decimal)
 * <UnifiedTypeSelector
 *   value={{type: "Decimal", scale: 2, precision: 10}}
 *   onValueChange={handleTypeChange}
 *   enableComplexTypes={true}
 * />
 *
 * @example
 * // List with nested types
 * <UnifiedTypeSelector
 *   value={{type: "List", inner: "String"}}
 *   onValueChange={handleTypeChange}
 *   enableComplexTypes={true}
 * />
 */
export const UnifiedTypeSelector: React.FC<UnifiedTypeSelectorProps> = ({
  value,
  onValueChange,
  availableTypes = COMMON_POLARS_TYPES,
  excludeTypes = [],
  includeCustomTypes = false,
  customTypeDefs = {},
  customTypeFilter,
  showSearch = false,
  showCategories = false,
  showManageButton = false,
  onManageCustomTypes,
  enableComplexTypes = false,
  maxNestingDepth = 3,
  currentDepth = 0,
  className = 'w-[180px]',
  disabled = false,
  placeholder = 'Select type',
  triggerClassName,
  children
}) => {
  const [searchQuery, setSearchQuery] = useState('');

  // Extract the current value for the dropdown
  // For custom types, use the UUID directly; for others, use the type name
  const currentTypeName = useMemo(() => {
    const typeId = getCustomTypeId(value);
    if (typeId) {
      return typeId; // Return UUID directly for custom types
    }
    return getPolarsTypeName(value);
  }, [value]);

  // Build type list
  const typeList = useMemo(() => {
    const types: Array<{ value: string; label: string; category: 'primitive' | 'custom'; isCustom: boolean }> = [];

    // Add primitive types
    availableTypes
      .filter(t => !excludeTypes.includes(t))
      .forEach(type => {
        types.push({ value: type, label: type, category: 'primitive', isCustom: false });
      });

    // Add custom types - use UUID directly as value
    if (includeCustomTypes) {
      Object.entries(customTypeDefs).forEach(([typeId, typeDef]) => {
        // Keep the current selection listed so it still displays, even if it is no longer allowed
        if (customTypeFilter && typeId !== currentTypeName && !customTypeFilter(typeDef, currentDepth)) return;
        types.push({
          value: typeId,
          label: `${typeDef.name} (${typeDef.type})`,
          category: 'custom',
          isCustom: true
        });
      });
    }

    return types;
  }, [availableTypes, excludeTypes, includeCustomTypes, customTypeDefs, customTypeFilter, currentTypeName, currentDepth]);

  // Filter types based on search
  const filteredTypes = useMemo(() => {
    if (!searchQuery.trim()) return typeList;
    const search = searchQuery.toLowerCase();
    return typeList.filter(t => t.label.toLowerCase().includes(search));
  }, [typeList, searchQuery]);

  // Check if current value is a complex type
  const isComplexType = isPolarsComplexType(value);
  const isDecimal = currentTypeName === 'Decimal';
  const isList = currentTypeName === 'List';
  const isSet = currentTypeName === 'Set';
  const isArray = currentTypeName === 'Array';

  // Handle type change from the main dropdown
  const handleTypeChange = (newTypeName: string) => {
    // Check if this is a custom type selection (UUID in customTypeDefs)
    const selectedType = typeList.find(t => t.value === newTypeName);
    if (selectedType?.isCustom) {
      onValueChange({ type: 'Custom', type_id: newTypeName });
      return;
    }

    // If switching to a complex type that requires configuration, create the object
    if (enableComplexTypes) {
      switch (newTypeName) {
        case 'Decimal':
          onValueChange({ type: 'Decimal', scale: 2, precision: 10 });
          break;
        case 'List':
          onValueChange({ type: 'List', inner: 'String' as PolarsType });
          break;
        case 'Set':
          onValueChange({ type: 'Set', inner: 'String' as PolarsType });
          break;
        case 'Array':
          onValueChange({ type: 'Array', inner: 'String' as PolarsType, width: 1 });
          break;
        default:
          // Simple primitive type - cast to PolarsType (which can be a string)
          onValueChange(newTypeName as PolarsType);
      }
    } else {
      // Simple mode - just pass the string (cast to PolarsType)
      onValueChange(newTypeName as PolarsType);
    }
  };

  // Handle changes to complex type parameters
  const handleDecimalPrecisionChange = (precision: string) => {
    if (isComplexType && value.type === 'Decimal') {
      const precisionNum = precision === '' ? undefined : parseInt(precision, 10);
      onValueChange({ ...value, precision: isNaN(precisionNum!) ? undefined : precisionNum });
    }
  };

  const handleDecimalScaleChange = (scale: string) => {
    if (isComplexType && value.type === 'Decimal') {
      const scaleNum = parseInt(scale, 10);
      onValueChange({ ...value, scale: isNaN(scaleNum) ? 0 : scaleNum });
    }
  };

  const handleListInnerTypeChange = (innerType: PolarsType) => {
    if (isComplexType && value.type === 'List') {
      onValueChange({ ...value, inner: innerType });
    }
  };

  const handleSetInnerTypeChange = (innerType: PolarsType) => {
    if (isComplexType && value.type === 'Set') {
      onValueChange({ ...value, inner: innerType });
    }
  };

  const handleArrayInnerTypeChange = (innerType: PolarsType) => {
    if (isComplexType && value.type === 'Array') {
      onValueChange({ ...value, inner: innerType });
    }
  };

  const handleArrayWidthChange = (width: string) => {
    if (isComplexType && value.type === 'Array') {
      const widthNum = parseInt(width, 10);
      onValueChange({ ...value, width: isNaN(widthNum) ? 1 : widthNum });
    }
  };

  // Check if we've hit max nesting depth
  const canNestFurther = currentDepth < maxNestingDepth;

  return (
    <div className="flex items-center gap-1">
      <Select
        value={currentTypeName}
        onValueChange={handleTypeChange}
        disabled={disabled}
      >
        <SelectTrigger className={triggerClassName || className}>
          {children || <SelectValue placeholder={placeholder} />}
        </SelectTrigger>
        <SelectContent>
          {/* Search bar */}
          {showSearch && (
            <div className="flex items-center px-2 py-1.5 border-b">
              <Search className="h-4 w-4 text-gray-400 mr-2" />
              <Input
                placeholder="Search types..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-7 text-xs border-0 p-0 focus-visible:ring-0"
                onClick={(e) => e.stopPropagation()}
              />
            </div>
          )}

          {/* Type list */}
          {filteredTypes.length === 0 ? (
            <div className="py-2 px-2 text-sm text-gray-500">No types found</div>
          ) : showCategories ? (
            <>
              {/* Primitive types */}
              {filteredTypes.some(t => t.category === 'primitive') && (
                <>
                  <div className="px-2 py-1.5 text-xs font-semibold text-gray-500 uppercase">
                    Primitive Types
                  </div>
                  {filteredTypes
                    .filter(t => t.category === 'primitive')
                    .map(type => (
                      <SelectItem key={type.value} value={type.value}>
                        {type.label}
                      </SelectItem>
                    ))}
                </>
              )}

              {/* Custom types */}
              {filteredTypes.some(t => t.category === 'custom') && (
                <>
                  <div className="px-2 py-1.5 text-xs font-semibold text-gray-500 uppercase border-t mt-1">
                    Custom Types
                  </div>
                  {filteredTypes
                    .filter(t => t.category === 'custom')
                    .map(type => (
                      <SelectItem key={type.value} value={type.value}>
                        {type.label}
                      </SelectItem>
                    ))}
                </>
              )}
            </>
          ) : (
            // No categories - flat list
            filteredTypes.map(type => (
              <SelectItem key={type.value} value={type.value}>
                {type.label}
              </SelectItem>
            ))
          )}

          {/* Manage Custom Types button */}
          {showManageButton && onManageCustomTypes && (
            <div className="border-t mt-1">
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onManageCustomTypes();
                }}
              >
                <Settings className="h-4 w-4 mr-2" />
                Manage Custom Types
              </Button>
            </div>
          )}
        </SelectContent>
      </Select>

      {/* Decimal configuration */}
      {enableComplexTypes && isDecimal && isComplexType && value.type === 'Decimal' && (
        <>
          <span className="text-sm text-gray-600">(</span>
          <Input
            type="number"
            placeholder="precision"
            value={value.precision ?? ''}
            onChange={(e) => handleDecimalPrecisionChange(e.target.value)}
            className="w-20 text-sm px-2"
            title="Precision (total digits)"
          />
          <span className="text-sm text-gray-600">,</span>
          <Input
            type="number"
            placeholder="scale"
            value={value.scale}
            onChange={(e) => handleDecimalScaleChange(e.target.value)}
            className="w-16 text-sm px-2"
            title="Scale (decimal places)"
          />
          <span className="text-sm text-gray-600">)</span>
        </>
      )}

      {/* List inner type configuration */}
      {enableComplexTypes && isList && isComplexType && value.type === 'List' && canNestFurther && (
        <>
          <span className="text-sm text-gray-600">(</span>
          <UnifiedTypeSelector
            value={value.inner}
            onValueChange={handleListInnerTypeChange}
            availableTypes={MOST_COMMON_TYPES}
            includeCustomTypes={includeCustomTypes}
            customTypeDefs={customTypeDefs}
            customTypeFilter={customTypeFilter}
            showSearch={includeCustomTypes && Object.keys(customTypeDefs || {}).length > 0}
            showCategories={includeCustomTypes && Object.keys(customTypeDefs || {}).length > 0}
            enableComplexTypes={true}
            maxNestingDepth={maxNestingDepth}
            currentDepth={currentDepth + 1}
            className="w-auto min-w-[100px]"
            placeholder="Inner type"
          />
          <span className="text-sm text-gray-600">)</span>
        </>
      )}

      {/* Set inner type configuration */}
      {enableComplexTypes && isSet && isComplexType && value.type === 'Set' && canNestFurther && (
        <>
          <span className="text-sm text-gray-600">(</span>
          <UnifiedTypeSelector
            value={value.inner}
            onValueChange={handleSetInnerTypeChange}
            availableTypes={MOST_COMMON_TYPES}
            includeCustomTypes={includeCustomTypes}
            customTypeDefs={customTypeDefs}
            customTypeFilter={customTypeFilter}
            showSearch={includeCustomTypes && Object.keys(customTypeDefs || {}).length > 0}
            showCategories={includeCustomTypes && Object.keys(customTypeDefs || {}).length > 0}
            enableComplexTypes={true}
            maxNestingDepth={maxNestingDepth}
            currentDepth={currentDepth + 1}
            className="w-auto min-w-[100px]"
            placeholder="Inner type"
          />
          <span className="text-sm text-gray-600">)</span>
        </>
      )}

      {/* Array configuration (inner type + width) */}
      {enableComplexTypes && isArray && isComplexType && value.type === 'Array' && canNestFurther && (
        <>
          <span className="text-sm text-gray-600">(</span>
          <UnifiedTypeSelector
            value={value.inner}
            onValueChange={handleArrayInnerTypeChange}
            availableTypes={MOST_COMMON_TYPES}
            includeCustomTypes={includeCustomTypes}
            customTypeDefs={customTypeDefs}
            customTypeFilter={customTypeFilter}
            showSearch={includeCustomTypes && Object.keys(customTypeDefs || {}).length > 0}
            showCategories={includeCustomTypes && Object.keys(customTypeDefs || {}).length > 0}
            enableComplexTypes={true}
            maxNestingDepth={maxNestingDepth}
            currentDepth={currentDepth + 1}
            className="w-auto min-w-[100px]"
            placeholder="Inner type"
          />
          <span className="text-sm text-gray-600">,</span>
          <Input
            type="number"
            min="1"
            placeholder="width"
            value={value.width}
            onChange={(e) => handleArrayWidthChange(e.target.value)}
            className="w-16 text-sm px-2"
            title="Array width"
          />
          <span className="text-sm text-gray-600">)</span>
        </>
      )}

      {/* Warning if max depth reached */}
      {enableComplexTypes && (isList || isArray) && !canNestFurther && (
        <span className="text-xs text-orange-600 ml-1" title="Max nesting depth reached">
          (...)
        </span>
      )}
    </div>
  );
};
