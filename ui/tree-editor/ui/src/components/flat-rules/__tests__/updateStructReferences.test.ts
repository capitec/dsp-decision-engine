import { describe, it, expect } from 'vitest';
import { updateStructReferencesWithMapping, updateStructReferences } from '../updateStructReferences';

const struct = (name: string, data: any[]): any => ({
  name,
  type: 'struct',
  definition: { fields: [['id', 'String']], display_field: 'id', data },
});

const output = (): any => ({
  dtypes: [['Owner', { type: 'Custom', type_id: 'person' }], ['Team', { type: 'List', inner: { type: 'Custom', type_id: 'person' } }]],
  type_defs: {},
  output: [
    { Owner: { $key: 'ann' }, Team: [{ $key: 'ann' }, { $key: 'bob' }], Plain: 'ann' },
    { Owner: { $key: 'bob' }, Team: [] },
  ],
  default: { Owner: { $key: 'ann' }, Team: [{ $key: 'bob' }] },
});

describe('updateStructReferencesWithMapping', () => {
  it('rewrites direct, list and default references and leaves plain strings alone', () => {
    const { updatedOutputConfig } = updateStructReferencesWithMapping({ ann: 'anna' }, {}, output());
    expect(updatedOutputConfig!.output[0]).toEqual({ Owner: { $key: 'anna' }, Team: [{ $key: 'anna' }, { $key: 'bob' }], Plain: 'ann' });
    expect(updatedOutputConfig!.output[1].Owner).toEqual({ $key: 'bob' });
    expect(updatedOutputConfig!.default).toEqual({ Owner: { $key: 'anna' }, Team: [{ $key: 'bob' }] });
  });

  it('applies a swap at once instead of one rename after the other', () => {
    const { updatedOutputConfig } = updateStructReferencesWithMapping({ ann: 'bob', bob: 'ann' }, {}, output());
    expect(updatedOutputConfig!.output[0].Owner).toEqual({ $key: 'bob' });
    expect(updatedOutputConfig!.output[0].Team).toEqual([{ $key: 'bob' }, { $key: 'ann' }]);
    expect(updatedOutputConfig!.output[1].Owner).toEqual({ $key: 'ann' });
  });

  it('rewrites references held by the data of other struct types, nested in lists', () => {
    const defs = { team: struct('team', [{ id: 't', lead: { $key: 'ann' }, members: [{ $key: 'ann' }, { $key: 'x' }] }]) };
    const { updatedTypeDefs } = updateStructReferencesWithMapping({ ann: 'anna' }, defs);
    expect((updatedTypeDefs.team as any).definition.data[0]).toEqual({ id: 't', lead: { $key: 'anna' }, members: [{ $key: 'anna' }, { $key: 'x' }] });
  });

  it('does not change its input and works without an output config', () => {
    const input = output();
    updateStructReferencesWithMapping({ ann: 'anna' }, {}, input);
    expect(input.output[0].Owner).toEqual({ $key: 'ann' });
    expect(updateStructReferencesWithMapping({ ann: 'anna' }, {}).updatedOutputConfig).toBeUndefined();
  });

  it('keeps a null default and an empty mapping changes nothing', () => {
    const cfg = { ...output(), default: null };
    expect(updateStructReferencesWithMapping({}, {}, cfg).updatedOutputConfig).toEqual(cfg);
  });

  it('the legacy single-value form maps one value', () => {
    const { updatedOutputConfig } = updateStructReferences('person', 'ann', 'anna', 'id', {}, output());
    expect(updatedOutputConfig!.output[0].Owner).toEqual({ $key: 'anna' });
  });

  // The mapping is by value only, so a same-named record of another struct type is rewritten as well.
  it.fails('renaming a record of one struct type leaves references to a same-named record of another type', () => {
    const cfg: any = {
      dtypes: [['Owner', { type: 'Custom', type_id: 'person' }], ['Pet', { type: 'Custom', type_id: 'pet' }]],
      type_defs: {},
      output: [{ Owner: { $key: 'rex' }, Pet: { $key: 'rex' } }],
      default: {},
    };
    const { updatedOutputConfig } = updateStructReferences('person', 'rex', 'max', 'id', {}, cfg);
    expect(updatedOutputConfig!.output[0].Pet).toEqual({ $key: 'rex' });
  });

  // `in` on the mapping object also sees inherited keys such as "constructor".
  it.fails('a record whose display value is "constructor" is not rewritten when it is not in the mapping', () => {
    const cfg: any = { dtypes: [], type_defs: {}, output: [{ Owner: { $key: 'constructor' } }], default: {} };
    const { updatedOutputConfig } = updateStructReferencesWithMapping({ ann: 'anna' }, {}, cfg);
    expect(updatedOutputConfig!.output[0].Owner).toEqual({ $key: 'constructor' });
  });
});
