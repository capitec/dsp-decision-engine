// Legacy output field types for backward compatibility
// These are project-scoped custom types (not global)

export type TypeKind = 'enum' | 'record';

export interface EnumDefinition {
  values: string[];
}

export interface RecordDefinition {
  records: Array<Record<string, any>>;
  fields: Record<string, 'string' | 'number' | 'boolean'> | Array<[string, 'string' | 'number' | 'boolean']>;
}

export interface BaseCustomType {
  id: string;
  name: string;
  description?: string;
}

export interface EnumType extends BaseCustomType {
  type_kind: 'enum';
  definition: EnumDefinition;
}

export interface RecordType extends BaseCustomType {
  type_kind: 'record';
  definition: RecordDefinition;
  display_format?: string;
}

export type CustomType = EnumType | RecordType;

export interface ProjectOutputField {
  id: string;
  field_name: string;
  field_type: 'string' | 'number' | 'boolean' | 'custom' | 'list';
  list_type?: 'string' | 'number' | 'custom';
  custom_type_id?: string;
  custom_type?: CustomType;
  is_required?: boolean;
  default_value?: any;
}
