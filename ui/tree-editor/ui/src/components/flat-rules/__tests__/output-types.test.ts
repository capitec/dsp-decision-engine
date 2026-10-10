import { describe, it, expect } from 'vitest';
import {
  validateOutputSchema,
  validateUITypeDef,
  createDefaultStructTypeDef,
  createDefaultCategoricalTypeDef,
  createDefaultOutputConfiguration,
  parseDTypeEntry,
  createDTypeEntry,
  updateDTypesForTypeDefChange,
} from '../output-types';

const struct = (over: any = {}): any => ({
  name: 'Person',
  type: 'struct',
  definition: {
    fields: [['id', 'String'], ['age', 'Int64']],
    display_field: 'id',
    data: [{ id: 'ann', age: 3 }, { id: 'bob', age: 4 }],
    ...over,
  },
});

describe('validateOutputSchema', () => {
  it('accepts a schema with unique, well-formed names', () => {
    expect(validateOutputSchema([['Action', 'String'], ['score_2', 'Int64'], ['_x', 'Float64']])).toEqual([]);
  });

  it('requires at least one field', () => {
    expect(validateOutputSchema([])).toEqual(['At least one output field is required']);
    expect(validateOutputSchema({})).toEqual(['At least one output field is required']);
  });

  it('rejects duplicate names', () => {
    expect(validateOutputSchema([['A', 'String'], ['A', 'Int64']])).toContain('Field names must be unique');
  });

  it.each(['1a', 'a b', 'a-b', 'é', 'a.b'])('rejects the name %j', name => {
    expect(validateOutputSchema([[name, 'String']])).toEqual([
      `Field name "${name}" is invalid. Use letters, numbers, and underscores only.`,
    ]);
  });

  it('reports an empty name as empty and as invalid', () => {
    const errors = validateOutputSchema([['', 'String']]);
    expect(errors).toContain('Field names cannot be empty');
    expect(errors).toHaveLength(2);
  });

  it('reads the older object form', () => {
    expect(validateOutputSchema({ A: 'String' })).toEqual([]);
    expect(validateOutputSchema({ '1a': 'String' })).toHaveLength(1);
  });
});

describe('validateUITypeDef', () => {
  it('accepts a well-formed struct and categorical type', () => {
    expect(validateUITypeDef(struct())).toEqual([]);
    expect(validateUITypeDef({ name: 'Tier', type: 'categorical', definition: { categories: ['a'] } })).toEqual([]);
  });

  it('requires a name', () => {
    expect(validateUITypeDef({ ...struct(), name: '  ' })).toContain('Type name is required');
  });

  it('requires fields and at least one data record in a struct', () => {
    const errors = validateUITypeDef(struct({ fields: [], data: [] }));
    expect(errors).toContain('Struct type must have at least one field');
    expect(errors).toContain('Struct type must have at least one data record');
  });

  it('requires a display field that is one of the fields', () => {
    expect(validateUITypeDef(struct({ display_field: '' }))).toContain('Struct type must have a display field selected');
    expect(validateUITypeDef(struct({ display_field: 'nope' }))).toContain('Display field must be one of the defined fields');
  });

  it('requires display values to be unique and non-empty', () => {
    expect(validateUITypeDef(struct({ data: [{ id: 'a' }, { id: 'a' }] }))).toContain(
      'Display field values must be unique across all records'
    );
    const errors = validateUITypeDef(struct({ data: [{ id: 'a' }, { id: '' }, {}] }));
    expect(errors).toContain('Display field values cannot be empty');
    expect(errors).not.toContain('Display field values must be unique across all records');
  });

  it('requires at least one category', () => {
    expect(validateUITypeDef({ name: 'T', type: 'categorical', definition: { categories: [] } })).toEqual([
      'Categorical type must have at least one category',
    ]);
  });

  it('reports a struct field the engine would refuse, naming the field', () => {
    const defs: any = { inner: struct() };
    const errors = validateUITypeDef(struct({ fields: [['id', 'String'], ['owner', { type: 'Custom', type_id: 'inner' }]] }), defs);
    expect(errors).toEqual([expect.stringMatching(/^Field "owner": A struct can't contain another struct/)]);
  });
});

describe('defaults', () => {
  it('a new struct starts with an id field that is its display field, and no data', () => {
    expect(createDefaultStructTypeDef().definition).toEqual({ fields: [['id', 'String']], data: [], display_field: 'id' });
  });

  it('a new categorical type has no categories and a new output configuration is empty', () => {
    expect(createDefaultCategoricalTypeDef().definition.categories).toEqual([]);
    expect(createDefaultOutputConfiguration()).toEqual({ dtypes: [], type_defs: {}, output: [], default: {} });
  });

  it('a new struct type is not valid until it has data and a name', () => {
    expect(validateUITypeDef(createDefaultStructTypeDef())).toEqual([
      'Type name is required',
      'Struct type must have at least one data record',
    ]);
  });
});

describe('dtypes entries', () => {
  const defs: any = { t: { name: 't', type: 'categorical', definition: { categories: ['a'] } } };

  it('parseDTypeEntry reads a custom reference and its kind', () => {
    expect(parseDTypeEntry('Tier', { type: 'Custom', type_id: 't' } as any, defs)).toEqual({
      fieldName: 'Tier',
      fieldType: { type: 'Custom', type_id: 't' },
      typeId: 't',
      typeKind: 'categorical',
    });
  });

  it('parseDTypeEntry leaves primitives and unknown custom types without a kind', () => {
    expect(parseDTypeEntry('A', 'String', defs)).toEqual({ fieldName: 'A', fieldType: 'String' });
    expect(parseDTypeEntry('A', { type: 'Custom', type_id: 'gone' } as any, defs).typeKind).toBeUndefined();
  });

  it('createDTypeEntry builds a custom reference only when a type id and definitions are given', () => {
    expect(createDTypeEntry('A', 'custom-struct', 'id1', defs)).toEqual(['A', { type: 'Custom', type_id: 'id1' }]);
    expect(createDTypeEntry('A', 'Int64')).toEqual(['A', 'Int64']);
    expect(createDTypeEntry('A', 'custom-struct')).toEqual(['A', 'custom-struct']);
  });

  it('updateDTypesForTypeDefChange keeps every column, including those of the changed type', () => {
    const dtypes: any = [['A', 'String'], ['T', { type: 'Custom', type_id: 't', extra: 1 }]];
    expect(updateDTypesForTypeDefChange(dtypes, 't', defs.t)).toEqual([['A', 'String'], ['T', { type: 'Custom', type_id: 't' }]]);
  });
});
