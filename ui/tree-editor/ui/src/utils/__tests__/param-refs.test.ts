import { describe, it, expect } from 'vitest';
import { makeParamRef, paramRefName, collectParamRefs, boundLabel } from '../param-refs';
import {
  renameParameterInNodes,
  deleteParameterFromNodes,
  extractParameterReferences,
  countParameterUsages,
} from '../../components/editor/VariableActions';

const node = (id: string, data: any) => ({ id, position: { x: 0, y: 0 }, data }) as any;

const nodes = [
  node('u', { type: 'unary', condition: { op: '>', feature: 'f', threshold: { param: 'a' } } }),
  node('legacy', { type: 'unary', condition: { op: '<', feature: 'f', threshold: { key: 'a' } } }),
  node('r', { type: 'cases', op: 'ranges', feature: 'f', conditions: [{ min: { param: 'a' }, max: 5 }, { min: 5, max: { param: 'b' } }] }),
  node('i', { type: 'cases', op: 'isin', feature: 'f', conditions: [{ values: { param: 'a' } }, { values: [1, { param: 'a' }] }] }),
  node('s', { type: 'cases', op: 'string_match', feature: 'f', conditions: [{ patterns: ['x', { param: 'a' }] }] }),
  node('c', { type: 'composite', condition: { logic_op: 'and', conditions: [{ op: '>', feature: 'f', threshold: { param: 'a' } }] } }),
  node('plain', { type: 'unary', condition: { op: '>', feature: 'f', threshold: 3 } }),
];

describe('parameter references', () => {
  it('are written as {param: name}', () => {
    expect(makeParamRef('rate')).toEqual({ param: 'rate' });
  });

  it('are read from both {param} and legacy {key}', () => {
    expect(paramRefName({ param: 'a' })).toBe('a');
    expect(paramRefName({ key: 'a' })).toBe('a');
    expect(paramRefName(3)).toBeNull();
    expect(paramRefName([1, 2])).toBeNull();
    expect(paramRefName({ other: 'a' })).toBeNull();
  });

  it('are found in thresholds, bounds, isin values, patterns and composites', () => {
    expect([...collectParamRefs(nodes.map(n => n.data))].sort()).toEqual(['a', 'b']);
    expect(extractParameterReferences(nodes)).toEqual(['a', 'b']);
    expect(countParameterUsages(nodes, 'a')).toBe(6);
    expect(countParameterUsages(nodes, 'b')).toBe(1);
  });

  it('rename covers every site, including legacy {key} refs', () => {
    const renamed: any[] = renameParameterInNodes(nodes, 'a', 'z');
    expect(countParameterUsages(renamed, 'a')).toBe(0);
    expect(countParameterUsages(renamed, 'z')).toBe(6);
    expect(renamed[1].data.condition.threshold).toEqual({ param: 'z' });
    expect(renamed[2].data.conditions[1].max).toEqual({ param: 'b' });
    expect(renamed[6]).toBe(nodes[6]);
  });

  it('delete resets each site to a neutral value', () => {
    const cleared: any[] = deleteParameterFromNodes(nodes, 'a');
    expect(countParameterUsages(cleared, 'a')).toBe(0);
    expect(cleared[0].data.condition.threshold).toBe(0);
    expect(cleared[2].data.conditions[0].min).toBeNull();
    expect(cleared[3].data.conditions[0].values).toEqual([]);
    expect(cleared[3].data.conditions[1].values).toEqual([1]);
    expect(cleared[4].data.conditions[0].patterns).toEqual(['x']);
  });
});

describe('boundLabel', () => {
  it('prints a parameter bound by name, a missing bound as -∞ and a number as itself', () => {
    expect(boundLabel({ param: 'limit' })).toBe('$limit');
    expect(boundLabel({ key: 'legacy' })).toBe('$legacy');
    expect(boundLabel(null)).toBe('-∞');
    expect(boundLabel(undefined)).toBe('-∞');
    expect(boundLabel(1000)).toBe('1000');
    expect(boundLabel(0)).toBe('0');
  });
});
