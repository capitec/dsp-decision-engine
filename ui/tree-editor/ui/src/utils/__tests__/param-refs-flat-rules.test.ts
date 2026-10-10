import { describe, it, expect } from 'vitest';
import {
  REMOVE,
  mapParamRefs,
  collectParamRefs,
  renameParamRefs,
  clearParamRefs,
  paramRefName,
  isParamRef,
} from '../param-refs';

const p = (param: string) => ({ param });
const leaf = (result_idx = 0) => ({ id: 'l', type: 'leaf', result_idx });

// A flat-rule tree with a reference in every slot a reference can sit in.
const rule = () => ({
  id: 'root',
  type: 'composite',
  op: 'and',
  conditions: [
    { type: 'unary', feature: 'f', op: '>', threshold: p('thr') },
    { type: 'unary', feature: 'f', op: 'between', min: p('lo'), max: p('hi') },
    { type: 'unary', feature: 'f', op: 'isin', values: p('set') },
    { type: 'unary', feature: 'f', op: 'isin', values: [1, p('member'), 3] },
    { type: 'unary', feature: 'f', op: 'string_match', patterns: ['a', p('pat')] },
  ],
  then: {
    id: 'c',
    type: 'cases',
    op: 'ranges',
    feature: 'f',
    conditions: [
      { when: { min: null, max: p('band1') }, then: 0 },
      { when: { min: p('band1'), max: 9 }, then: 1 },
      { when: { values: p('whenSet') }, then: 1 },
      { when: { patterns: [p('whenPat'), 'x'] }, then: 1 },
    ],
    branches: [leaf(), { id: 'u', type: 'unary', condition: { op: '<', feature: 'f', threshold: p('inBranch') }, then: leaf() }],
    otherwise: 1,
  },
  otherwise: {
    id: 'o',
    type: 'unary',
    condition: { op: '==', feature: 'f', threshold: p('inOtherwise') },
    then: leaf(),
    otherwise: leaf(-1),
  },
});

const ALL = ['thr', 'lo', 'hi', 'set', 'member', 'pat', 'band1', 'whenSet', 'whenPat', 'inBranch', 'inOtherwise'];

describe('paramRefName', () => {
  it('prefers {param} over the legacy {key}, falls back to key when param is not a string, and keeps an empty name', () => {
    expect(paramRefName({ param: 'a', key: 'b' })).toBe('a');
    expect(paramRefName({ param: 5, key: 'b' })).toBe('b');
    expect(paramRefName({ param: '' })).toBe('');
  });

  it('is null for anything that is not an object with a string name', () => {
    for (const v of [null, undefined, 3, 'param', true, [], [{ param: 'a' }], { param: 5 }, { key: null }, {}]) {
      expect(paramRefName(v)).toBeNull();
      expect(isParamRef(v)).toBe(false);
    }
  });
});

describe('collectParamRefs', () => {
  it('finds a reference in every slot of a nested flat rule', () => {
    expect([...collectParamRefs(rule())].sort()).toEqual([...ALL].sort());
  });

  it('finds a reference five composites deep', () => {
    let node: any = { type: 'unary', condition: { op: '>', feature: 'f', threshold: p('deep') }, then: leaf() };
    for (let i = 0; i < 5; i++) node = { type: 'composite', op: 'and', conditions: [node], then: leaf() };
    expect([...collectParamRefs(node)]).toEqual(['deep']);
  });

  it('collects into a given set and finds nothing in literals', () => {
    const into = new Set(['already']);
    expect(collectParamRefs({ threshold: 3, patterns: ['a'] }, into)).toBe(into);
    expect([...into]).toEqual(['already']);
  });
});

describe('renameParamRefs', () => {
  it('renames every site and keeps the rest of the tree as it was', () => {
    const before = rule();
    const out: any = renameParamRefs(before, 'band1', 'edge');
    expect(collectParamRefs(out).has('band1')).toBe(false);
    expect(out.then.conditions[0].when.max).toEqual(p('edge'));
    expect(out.then.conditions[1].when.min).toEqual(p('edge'));
    expect(out.then.otherwise).toBe(1);
    expect(out.conditions[0]).toEqual(before.conditions[0]);
    expect(before.then.conditions[0].when.max).toEqual(p('band1'));
  });

  it('renames a legacy {key} reference to the {param} form', () => {
    expect(renameParamRefs({ threshold: { key: 'a' } }, 'a', 'b')).toEqual({ threshold: { param: 'b' } });
  });

  it('renames inside arrays and leaves other names alone', () => {
    const out: any = renameParamRefs(rule(), 'member', 'm2');
    expect(out.conditions[3].values).toEqual([1, p('m2'), 3]);
    expect(out.conditions[0].threshold).toEqual(p('thr'));
  });

  it('renaming to the same name or an absent name changes nothing', () => {
    expect(renameParamRefs(rule(), 'thr', 'thr')).toEqual(rule());
    expect(renameParamRefs(rule(), 'absent', 'x')).toEqual(rule());
  });
});

describe('clearParamRefs', () => {
  const cleared = (name: string): any => clearParamRefs(rule(), name);

  it('resets a threshold to 0', () => {
    expect(cleared('thr').conditions[0].threshold).toBe(0);
    expect(cleared('inOtherwise').otherwise.condition.threshold).toBe(0);
    expect(cleared('inBranch').then.branches[1].condition.threshold).toBe(0);
  });

  it('resets between bounds and ranges bounds to null', () => {
    expect(cleared('lo').conditions[1]).toMatchObject({ min: null, max: p('hi') });
    expect(cleared('hi').conditions[1]).toMatchObject({ min: p('lo'), max: null });
    expect(cleared('band1').then.conditions[0].when.max).toBeNull();
    expect(cleared('band1').then.conditions[1].when.min).toBeNull();
  });

  it('resets a whole-list reference to an empty list', () => {
    expect(cleared('set').conditions[2].values).toEqual([]);
    expect(cleared('whenSet').then.conditions[2].when.values).toEqual([]);
  });

  it('drops a reference that sits inside a list, keeping the other members', () => {
    expect(cleared('member').conditions[3].values).toEqual([1, 3]);
    expect(cleared('pat').conditions[4].patterns).toEqual(['a']);
    expect(cleared('whenPat').then.conditions[3].when.patterns).toEqual(['x']);
  });

  it('leaves no reference to the cleared name and keeps every other one', () => {
    for (const name of ALL) {
      const out = collectParamRefs(cleared(name));
      expect(out.has(name), name).toBe(false);
      expect(out.size, name).toBe(ALL.length - 1);
    }
  });

  it('keeps the structure: indices, branches and the then/otherwise nodes', () => {
    const out = cleared('thr');
    expect(out.then.otherwise).toBe(1);
    expect(out.then.branches).toHaveLength(2);
    expect(out.otherwise.otherwise).toEqual(leaf(-1));
  });
});

describe('mapParamRefs', () => {
  it('passes the property that holds a reference, and undefined for a reference inside a list', () => {
    const seen: Array<[string, string | undefined]> = [];
    mapParamRefs({ threshold: p('a'), values: [p('b')], when: { min: p('c') } }, (n, k) => {
      seen.push([n, k]);
    });
    expect(seen).toEqual([['a', 'threshold'], ['b', undefined], ['c', 'min']]);
  });

  it('keeps a reference when the callback returns undefined and swaps it for any other value', () => {
    const v = { a: p('x') };
    expect(mapParamRefs(v, () => undefined).a).toBe(v.a);
    expect(mapParamRefs(v, () => null).a).toBeNull();
    expect(mapParamRefs(v, () => 0).a).toBe(0);
  });

  it('REMOVE drops list members only, and returns new containers instead of mutating', () => {
    const v = { values: [p('x'), 1, p('x'), p('y')] };
    const out = mapParamRefs(v, n => (n === 'x' ? REMOVE : undefined));
    expect(out.values).toEqual([1, p('y')]);
    expect(v.values).toHaveLength(4);
  });

  it('passes primitives, null and empty containers through', () => {
    expect(mapParamRefs(null, () => 1)).toBeNull();
    expect(mapParamRefs(5, () => 1)).toBe(5);
    expect(mapParamRefs('s', () => 1)).toBe('s');
    expect(mapParamRefs([], () => 1)).toEqual([]);
    expect(mapParamRefs({}, () => 1)).toEqual({});
  });
});
