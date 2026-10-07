// Shared types for data source components
// These match the backend Pydantic models

export interface DataSourceInfo {
  name: string;
  provider: string;
  display_name: string;
  description?: string | null;
  icon?: string | null;
  capabilities: string[]; // Pydantic Set serializes to array in JSON
}

export interface DataSourceList {
  datasources: DataSourceInfo[];
}

export interface PaginationInfo {
  page: number;
  page_size: number;
  total_count: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface DataSourceTablesResponse {
  datasource: string;
  tables: string[];
  pagination: PaginationInfo;
}

export interface DataSourceSearchResponse {
  datasource: string;
  query: string;
  tables: string[];
  pagination: PaginationInfo;
}

export interface TableInfo {
  name: string;
  row_count?: number | null;
  size_bytes?: number | null;
  last_modified?: string | null;
}

// PolarsSchema type definition
// Matches the decider PolarsSchema

// Primitive/simple Polars types (no constructor arguments)
export type PolarsPrimitiveType =
  // String types
  | "String"
  | "Binary"
  // Boolean
  | "Boolean"
  | "Null"
  // Integer types - signed
  | "Int8" | "Int16" | "Int32" | "Int64" | "Int128"
  // Integer types - unsigned
  | "UInt8" | "UInt16" | "UInt32" | "UInt64" | "UInt128"
  // Float types
  | "Float16" | "Float32" | "Float64";

// Complex Polars types that require constructor arguments
export interface PolarsDatetimeType {
  type: "Datetime";
  time_unit?: "ms" | "us" | "ns";
  time_zone?: string | null;
}

export interface PolarsDecimalType {
  type: "Decimal";
  precision?: number | null;
  scale: number;
}

export interface PolarsDurationType {
  type: "Duration";
  time_unit?: "ms" | "us" | "ns";
}

export interface PolarsListType {
  type: "List";
  inner: PolarsType; // Recursive - can be any Polars type
}

export interface PolarsSetType {
  type: "Set";
  inner: PolarsType; // Recursive - can be any Polars type (enforces uniqueness)
}

export interface PolarsArrayType {
  type: "Array";
  inner: PolarsType;
  width: number;
}

export interface PolarsStructType {
  type: "Struct";
  fields: Array<[string, PolarsType]>; // Nested schema
}

export interface PolarsCategoricalType {
  type: "Categorical";
  ordering?: string;
}

export interface PolarsEnumType {
  type: "Enum";
  categories?: string[];
}

export interface PolarsCustomType {
  type: "Custom";
  type_id: string; // UUID reference to custom type definition
}

// Simple types that map to type name strings
export type PolarsSimpleComplexType = "Date" | "Time";

// Union of all complex types
export type PolarsComplexType =
  | PolarsDatetimeType
  | PolarsDecimalType
  | PolarsDurationType
  | PolarsListType
  | PolarsSetType
  | PolarsArrayType
  | PolarsStructType
  | PolarsCategoricalType
  | PolarsEnumType
  | PolarsCustomType;

// A Polars type can be a simple string or a complex object
export type PolarsType =
  | PolarsPrimitiveType
  | PolarsSimpleComplexType
  | PolarsComplexType;

// Alias for compatibility with flat-rules
export type PolarsDataType = PolarsType;

// PolarsSchema is an ordered array of [field_name, type] tuples
export type PolarsSchema = Array<[string, PolarsType]>;

// Common Polars types used in parameter/variable management and type selectors
export const COMMON_POLARS_TYPES: string[] = [
  'String',
  'Int8', 'Int16', 'Int32', 'Int64',
  'UInt8', 'UInt16', 'UInt32', 'UInt64',
  'Float32', 'Float64',
  'Boolean',
  'Date', 'Datetime', 'Time',
  'Categorical', 'Decimal', 'List',
  ''  // Empty string for separator/placeholder
];

// Type guard helpers
export function isPolarsPrimitiveType(type: PolarsType): type is PolarsPrimitiveType {
  return typeof type === 'string' && (
    type === 'String' || type === 'Binary' || type === 'Boolean' || type === 'Null' ||
    type === 'Int8' || type === 'Int16' || type === 'Int32' || type === 'Int64' || type === 'Int128' ||
    type === 'UInt8' || type === 'UInt16' || type === 'UInt32' || type === 'UInt64' || type === 'UInt128' ||
    type === 'Float16' || type === 'Float32' || type === 'Float64'
  );
}

export function isPolarsSimpleType(type: PolarsType): type is PolarsPrimitiveType | PolarsSimpleComplexType {
  return typeof type === 'string';
}

export function isPolarsComplexType(type: PolarsType): type is PolarsComplexType {
  return typeof type === 'object' && type !== null && 'type' in type;
}

// Get the type name from any PolarsType
export function getPolarsTypeName(type: PolarsType): string {
  if (typeof type === 'string') {
    return type;
  }
  if (isPolarsComplexType(type)) {
    return type.type;
  }
  return 'Unknown';
}

// Check if a type is a custom type reference
export function isCustomType(type: PolarsType): type is PolarsCustomType {
  return isPolarsComplexType(type) && type.type === 'Custom';
}

// Extract custom type ID if the type is a custom reference
export function getCustomTypeId(type: PolarsType): string | null {
  return isCustomType(type) ? type.type_id : null;
}

export interface TableDetailResponse {
  datasource: string;
  table: string;
  info: TableInfo;
  table_schema: PolarsSchema;
}

export interface TablePreviewData {
  columns: string[];
  rows: any[][];
  row_count: number;
  total_columns: number;
}

export interface TablePreviewResponse {
  datasource: string;
  table: string;
  limit: number;
  data: TablePreviewData;
}

export interface UploadResponse {
  success: boolean;
  table_name?: string;
  message: string;
}