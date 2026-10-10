import { describe, it, expect } from 'vitest';
import {
  renameParameterInNodes,
  deleteParameterFromNodes,
  extractParameterReferences,
  countParameterUsages,
} from '../VariableActions';

const node = (id: string, data: any) => ({ id, position: { x: 0, y: 0 }, data }) as any;
const p = (param: string) => ({ param });

const nodes = [
  node('between', { type: 'unary', condition: { op: 'between', feature: 'f', min: p('lo'), max: p('hi') } }),
  node('ranges', { type: 'cases', op: 'ranges', feature: 'f', conditions: [{ min: null, max: p('lo') }, { min: p('lo'), max: p('hi') }] }),
  node('nested', {
    type: 'composite',
    op: 'and',
    conditions: [{ type: 'composite', op: 'not', conditions: [{ type: 'unary', feature: 'f', op: '>', threshold: p('lo') }] }],
  }),
  node('leaf', { type: 'leaf', result_idx: 0 }),
];

describe('parameter actions over tree nodes', () => {
  it('count each node once even when it references the parameter in several places', () => {
    expect(countParameterUsages(nodes, 'lo')).toBe(3);
    expect(countParameterUsages(nodes, 'hi')).toBe(2);
    expect(countParameterUsages(nodes, 'none')).toBe(0);
    expect(countParameterUsages([], 'lo')).toBe(0);
  });

  it('extract lists referenced names sorted and without duplicates', () => {
    expect(extractParameterReferences(nodes)).toEqual(['hi', 'lo']);
    expect(extractParameterReferences([node('l', { type: 'leaf', result_idx: 0 })])).toEqual([]);
  });

  it('rename keeps the identity of nodes that do not use the parameter and changes the ones that do', () => {
    const out = renameParameterInNodes(nodes, 'hi', 'top');
    expect(out[2]).toBe(nodes[2]);
    expect(out[3]).toBe(nodes[3]);
    expect(out[0].data.condition).toMatchObject({ min: p('lo'), max: p('top') });
    expect(nodes[0].data.condition.max).toEqual(p('hi'));
  });

  it('rename reaches a reference two composites down', () => {
    const out = renameParameterInNodes(nodes, 'lo', 'floor');
    expect(out[2].data.conditions[0].conditions[0].threshold).toEqual(p('floor'));
    expect(extractParameterReferences(out)).toEqual(['floor', 'hi']);
  });

  it('delete turns bounds into open bounds and thresholds into 0, leaving other parameters', () => {
    const out = deleteParameterFromNodes(nodes, 'lo');
    expect(out[0].data.condition).toMatchObject({ min: null, max: p('hi') });
    expect(out[1].data.conditions).toEqual([{ min: null, max: null }, { min: null, max: p('hi') }]);
    expect(out[2].data.conditions[0].conditions[0].threshold).toBe(0);
    expect(extractParameterReferences(out)).toEqual(['hi']);
  });

  it('delete and rename of an unused name return equal nodes', () => {
    expect(deleteParameterFromNodes(nodes, 'ghost')).toEqual(nodes);
    expect(renameParameterInNodes(nodes, 'ghost', 'x')).toEqual(nodes);
  });
});
