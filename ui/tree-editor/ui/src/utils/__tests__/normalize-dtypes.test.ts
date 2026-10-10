import { describe, it, expect, vi } from 'vitest';
import { normalizeDtypes, normalizeTypeDef, normalizeTypeDefs, normalizeTreeOutput } from '../normalize-dtypes';

describe('normalizeDtypes', () => {
  it('keeps the array form as it is', () => {
    const dtypes = [['A', 'string'], ['B', 'Int64']];
    expect(normalizeDtypes(dtypes)).toBe(dtypes);
  });

  it('turns the object form into pairs in key order', () => {
    expect(normalizeDtypes({ A: 'string', B: { type: 'Custom', type_id: 't' } })).toEqual([
      ['A', 'string'],
      ['B', { type: 'Custom', type_id: 't' }],
    ]);
  });

  it('gives an empty list for an empty object and warns for non-collections', () => {
    expect(normalizeDtypes({})).toEqual([]);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    for (const bad of [null, undefined, 'x', 3]) expect(normalizeDtypes(bad)).toEqual([]);
    expect(warn).toHaveBeenCalledTimes(4);
    warn.mockRestore();
  });
});

describe('normalizeTypeDef', () => {
  const struct = (fields: any) => ({ name: 's', type: 'struct', definition: { fields, display_field: 'a', data: [] } }) as any;

  it('turns struct fields given as an object into pairs, keeping the rest of the definition', () => {
    const out: any = normalizeTypeDef(struct({ a: 'String', b: 'Int64' }));
    expect(out.definition.fields).toEqual([['a', 'String'], ['b', 'Int64']]);
    expect(out.definition.display_field).toBe('a');
  });

  it('returns struct fields that are already pairs, and categorical types, untouched', () => {
    const s = struct([['a', 'String']]);
    expect(normalizeTypeDef(s)).toBe(s);
    const c = { name: 'c', type: 'categorical', definition: { categories: ['x'] } } as any;
    expect(normalizeTypeDef(c)).toBe(c);
  });
});

describe('normalizeTreeOutput', () => {
  it('normalises dtypes and every struct type def and keeps data and default', () => {
    const out = normalizeTreeOutput({
      dtypes: { A: 'string' },
      type_defs: { t: { name: 't', type: 'struct', definition: { fields: { a: 'String' }, display_field: 'a', data: [] } } },
      data: [{ A: 'x' }],
      default: { A: 'd' },
    } as any);
    expect(out.dtypes).toEqual([['A', 'string']]);
    expect((out.type_defs.t as any).definition.fields).toEqual([['a', 'String']]);
    expect(out.data).toEqual([{ A: 'x' }]);
    expect(out.default).toEqual({ A: 'd' });
  });

  it('accepts an output with no type_defs', () => {
    expect(normalizeTreeOutput({ dtypes: [], data: [], default: null } as any).type_defs).toEqual({});
    expect(normalizeTypeDefs({})).toEqual({});
  });
});
