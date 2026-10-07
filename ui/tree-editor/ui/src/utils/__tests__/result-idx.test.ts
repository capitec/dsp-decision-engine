import { describe, it, expect } from 'vitest';
import { shiftLeafResults } from '../result-idx';

const leaf = (result_idx: number) => ({ type: 'leaf', result_idx });

describe('shiftLeafResults', () => {
  it('points leaves on the removed row at the default and shifts later ones down', () => {
    const nodes = [
      { id: 'a', data: leaf(0) },
      { id: 'b', data: leaf(1) },
      { id: 'c', data: leaf(2) },
      { id: 'd', data: leaf(-1) },
      { id: 'e', data: { type: 'unary', condition: { threshold: 1 } } },
    ];
    expect(shiftLeafResults(nodes, 1).map((n: any) => n.data.result_idx)).toEqual([0, -1, 1, -1, undefined]);
  });

  it('reaches leaves nested in then/otherwise/branches of flat rules', () => {
    const root = {
      type: 'unary',
      then: leaf(2),
      otherwise: { type: 'cases', branches: [leaf(0), leaf(3)], otherwise: leaf(1) },
    };
    const out: any = shiftLeafResults(root, 0);
    expect(out.then.result_idx).toBe(1);
    expect(out.otherwise.branches.map((b: any) => b.result_idx)).toEqual([-1, 2]);
    expect(out.otherwise.otherwise.result_idx).toBe(0);
    expect(root.then.result_idx).toBe(2);
  });
});
