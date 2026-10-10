import { describe, it, expect } from 'vitest';
import { copyWithNewIds } from '../copyNode';

const leaf = (id: string, result_idx = 0): any => ({ id, type: 'leaf', result_idx });

const ids = (node: any, into: string[] = []): string[] => {
  if (!node || typeof node !== 'object') return into;
  if (typeof node.id === 'string') into.push(node.id);
  for (const key of ['then', 'otherwise']) if (node[key] && typeof node[key] === 'object') ids(node[key], into);
  for (const child of node.branches ?? []) ids(child, into);
  return into;
};

const cases = (): any => ({
  id: 'c',
  type: 'cases',
  op: 'ranges',
  feature: 'f',
  conditions: [{ when: { min: null, max: 5 }, then: 0 }],
  branches: [leaf('b0'), { id: 'u', type: 'unary', condition: { op: '>', feature: 'f', threshold: 1 }, then: leaf('ut'), otherwise: leaf('uo') }],
  otherwise: 1,
});

describe('copyWithNewIds', () => {
  it('gives every node of the copy a new id and leaves the original ids alone', () => {
    const original = cases();
    const copy = copyWithNewIds(original);
    const before = ids(original);
    const after = ids(copy);
    expect(after).toHaveLength(before.length);
    expect(new Set(after).size).toBe(after.length);
    expect(after.filter(id => before.includes(id))).toEqual([]);
    expect(ids(original)).toEqual(['c', 'b0', 'u', 'ut', 'uo']);
  });

  it('keeps a cases otherwise as a branch index, not a node', () => {
    const copy: any = copyWithNewIds(cases());
    expect(copy.otherwise).toBe(1);
    expect(typeof copy.otherwise).toBe('number');
  });

  it('keeps an otherwise index of 0', () => {
    const copy: any = copyWithNewIds({ ...cases(), otherwise: 0 });
    expect(copy.otherwise).toBe(0);
  });

  it('keeps a cases node without an otherwise without one', () => {
    const { otherwise, ...rest } = cases();
    expect((copyWithNewIds(rest) as any).otherwise).toBeUndefined();
  });

  it('keeps the condition-to-branch indices and the branch order', () => {
    const copy: any = copyWithNewIds(cases());
    expect(copy.conditions).toEqual(cases().conditions);
    expect(copy.branches.map((b: any) => b.type)).toEqual(['leaf', 'unary']);
    expect(copy.branches[1].then.type).toBe('leaf');
  });

  it('copies then and otherwise nodes of unary and composite nodes', () => {
    const original: any = {
      id: 'r',
      type: 'composite',
      op: 'and',
      conditions: [{ type: 'unary', feature: 'f', op: '>', threshold: 1 }],
      then: cases(),
      otherwise: leaf('o', -1),
    };
    const copy: any = copyWithNewIds(original);
    expect(copy.then.otherwise).toBe(1);
    expect(copy.then.id).not.toBe('c');
    expect(copy.otherwise).toMatchObject({ type: 'leaf', result_idx: -1 });
    expect(copy.otherwise.id).not.toBe('o');
  });

  it('a unary node without then/otherwise stays without them', () => {
    const copy: any = copyWithNewIds({ id: 'u', type: 'unary', condition: { op: '>', feature: 'f', threshold: 1 } } as any);
    expect(copy.then).toBeUndefined();
    expect(copy.otherwise).toBeUndefined();
  });

  it('copies a cases tree five levels deep and keeps every index at every level', () => {
    let node: any = leaf('bottom', 3);
    for (let i = 0; i < 5; i++) node = { ...cases(), id: `c${i}`, branches: [node, leaf(`x${i}`)], otherwise: 1 };
    let copy: any = copyWithNewIds(node);
    for (let i = 0; i < 5; i++) {
      expect(copy.otherwise).toBe(1);
      expect(copy.id).not.toBe(`c${4 - i}`);
      copy = copy.branches[0];
    }
    expect(copy).toMatchObject({ type: 'leaf', result_idx: 3 });
  });

  it('does not share branch objects with the original, so editing the copy cannot change it', () => {
    const original = cases();
    const copy: any = copyWithNewIds(original);
    expect(copy.branches).not.toBe(original.branches);
    expect(copy.branches[0]).not.toBe(original.branches[0]);
  });

  it('passes parameter references and UI fields on unchanged', () => {
    const original: any = { id: 'u', type: 'unary', editMode: true, condition: { op: '>', feature: 'f', threshold: { param: 'a' } } };
    const copy: any = copyWithNewIds(original);
    expect(copy.condition).toEqual({ op: '>', feature: 'f', threshold: { param: 'a' } });
    expect(copy.editMode).toBe(true);
  });

  it('keeps a composite condition list as it was, nested composites included', () => {
    const conditions = [{ type: 'composite', op: 'not', conditions: [{ type: 'unary', feature: 'f', op: '>', threshold: { param: 'a' } }] }];
    const copy: any = copyWithNewIds({ id: 'outer', type: 'composite', op: 'and', conditions, then: leaf('t') } as any);
    expect(copy.conditions).toEqual(conditions);
  });
});
