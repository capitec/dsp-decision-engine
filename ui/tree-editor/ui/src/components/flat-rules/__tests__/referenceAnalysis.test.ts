import { describe, it, expect } from 'vitest';
import {
  findCategoryReferences,
  findStructRecordReferences,
  removeCategoryReferences,
  removeStructRecordReferences,
} from '../referenceAnalysis';

const custom = (type_id: string) => ({ type: 'Custom', type_id }) as any;
const list = (inner: any) => ({ type: 'List', inner }) as any;

const config = (): any => ({
  dtypes: [
    ['Tier', custom('tier')],
    ['Tiers', list(custom('tier'))],
    ['Other', custom('other')],
    ['Owner', custom('person')],
    ['Owners', list(custom('person'))],
    ['Name', 'String'],
  ],
  type_defs: {},
  output: [
    { Tier: 'gold', Tiers: ['gold', 'silver'], Other: 'gold', Owner: { $key: 'ann' }, Owners: [{ $key: 'ann' }, { $key: 'bob' }], Name: 'gold' },
    { Tier: 'silver', Tiers: [], Other: 'silver', Owner: null, Owners: [{ $key: 'bob' }], Name: 'x' },
    { Tier: 'gold', Tiers: null, Other: 'gold', Owner: { $key: 'bob' }, Owners: 'not a list', Name: 'y' },
  ],
  default: {},
});

describe('findCategoryReferences', () => {
  it('finds direct and list uses of the value, per record and field', () => {
    expect(findCategoryReferences('gold', 'tier', config())).toEqual([
      { fieldName: 'Tier', recordIndex: 0, inArray: false },
      { fieldName: 'Tiers', recordIndex: 0, inArray: true },
      { fieldName: 'Tier', recordIndex: 2, inArray: false },
    ]);
  });

  it('does not count columns of another type or plain strings with the same text', () => {
    const refs = findCategoryReferences('gold', 'tier', config());
    expect(refs.some(r => r.fieldName === 'Other' || r.fieldName === 'Name')).toBe(false);
  });

  it('finds nothing for a value no record holds or a type no column uses', () => {
    expect(findCategoryReferences('bronze', 'tier', config())).toEqual([]);
    expect(findCategoryReferences('gold', 'missing', config())).toEqual([]);
  });

  it('finds nothing when there are no records', () => {
    expect(findCategoryReferences('gold', 'tier', { ...config(), output: [] })).toEqual([]);
  });
});

describe('removeCategoryReferences', () => {
  it('blanks direct uses, filters list uses, and leaves other columns alone', () => {
    const out = removeCategoryReferences('gold', 'tier', config());
    expect(out.output[0]).toMatchObject({ Tier: '', Tiers: ['silver'], Other: 'gold', Name: 'gold' });
    expect(out.output[1]).toMatchObject({ Tier: 'silver', Tiers: [] });
    expect(out.output[2].Tiers).toBeNull();
  });

  it('leaves no reference behind and does not change its input', () => {
    const input = config();
    const out = removeCategoryReferences('gold', 'tier', input);
    expect(findCategoryReferences('gold', 'tier', out)).toEqual([]);
    expect(findCategoryReferences('gold', 'tier', input)).toHaveLength(3);
  });
});

describe('findStructRecordReferences', () => {
  it('finds direct and list uses by $key', () => {
    expect(findStructRecordReferences('ann', 'person', config())).toEqual([
      { fieldName: 'Owner', recordIndex: 0, inArray: false, displayValue: 'ann' },
      { fieldName: 'Owners', recordIndex: 0, inArray: true, displayValue: 'ann' },
    ]);
    expect(findStructRecordReferences('bob', 'person', config()).map(r => [r.recordIndex, r.fieldName])).toEqual([
      [0, 'Owners'],
      [1, 'Owners'],
      [2, 'Owner'],
    ]);
  });

  it('finds nothing for a type no column uses or a record nobody references', () => {
    expect(findStructRecordReferences('ann', 'tier', config())).toEqual([]);
    expect(findStructRecordReferences('carl', 'person', config())).toEqual([]);
  });
});

describe('removeStructRecordReferences', () => {
  it('nulls direct uses and filters list uses', () => {
    const out = removeStructRecordReferences('bob', 'person', config());
    expect(out.output[0].Owners).toEqual([{ $key: 'ann' }]);
    expect(out.output[1].Owners).toEqual([]);
    expect(out.output[2].Owner).toBeNull();
    expect(out.output[0].Owner).toEqual({ $key: 'ann' });
  });

  it('leaves no reference behind', () => {
    for (const name of ['ann', 'bob']) {
      expect(findStructRecordReferences(name, 'person', removeStructRecordReferences(name, 'person', config()))).toEqual([]);
    }
  });
});

// The default row holds references like any record, but only the data rows are searched and cleaned.
it.fails('deleting a category or a struct record clears it from the default row too', () => {
  const cfg = { ...config(), default: { Tier: 'gold', Owner: { $key: 'ann' } } };
  expect([
    removeCategoryReferences('gold', 'tier', cfg).default.Tier,
    removeStructRecordReferences('ann', 'person', cfg).default.Owner,
  ]).toEqual(['', null]);
});
