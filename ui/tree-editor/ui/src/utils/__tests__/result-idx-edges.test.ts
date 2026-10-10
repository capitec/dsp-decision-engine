import { describe, it, expect } from 'vitest';
import { shiftLeafResults } from '../result-idx';

const leaf = (result_idx: number) => ({ type: 'leaf', result_idx });

describe('shiftLeafResults edge cases', () => {
  it('removing the last row points its leaves at the default and leaves earlier ones', () => {
    const out = shiftLeafResults([leaf(0), leaf(1), leaf(2)], 2);
    expect(out.map(l => l.result_idx)).toEqual([0, 1, -1]);
  });

  it('removing row 0 shifts every later row down and keeps the default', () => {
    expect(shiftLeafResults([leaf(0), leaf(1), leaf(5), leaf(-1)], 0).map(l => l.result_idx)).toEqual([-1, 0, 4, -1]);
  });

  it('leaves non-leaf nodes that happen to have a result_idx field alone', () => {
    const node = { type: 'unary', result_idx: 3 };
    expect(shiftLeafResults(node, 0)).toEqual({ type: 'unary', result_idx: 3 });
  });

  it('leaves a leaf whose result_idx is not a number alone', () => {
    expect(shiftLeafResults({ type: 'leaf', result_idx: undefined }, 0)).toEqual({ type: 'leaf', result_idx: undefined });
  });

  it('does not touch the cases branch-index fields (then, otherwise as numbers)', () => {
    const cases = { type: 'cases', conditions: [{ when: { min: 0, max: 1 }, then: 1 }], branches: [leaf(1), leaf(2)], otherwise: 1 };
    const out: any = shiftLeafResults(cases, 0);
    expect(out.conditions[0].then).toBe(1);
    expect(out.otherwise).toBe(1);
    expect(out.branches.map((b: any) => b.result_idx)).toEqual([0, 1]);
  });

  it('does not change its input and passes primitives and null through', () => {
    const input = [leaf(1)];
    shiftLeafResults(input, 0);
    expect(input[0].result_idx).toBe(1);
    expect(shiftLeafResults(null, 0)).toBeNull();
    expect(shiftLeafResults(3, 0)).toBe(3);
    expect(shiftLeafResults('x', 0)).toBe('x');
  });
});
