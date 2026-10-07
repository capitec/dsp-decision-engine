import { describe, it, expect } from 'vitest';
import { createsCycle } from '../util';

const edge = (source: string, target: string) => ({ id: `${source}:${target}`, source, target, data: { sourceIndex: [0] } }) as any;

describe('createsCycle', () => {
  const edges = [edge('a', 'b'), edge('b', 'c')];

  it('rejects an edge back to an ancestor', () => {
    expect(createsCycle(edge('c', 'a'), edges)).toBe(true);
    expect(createsCycle(edge('b', 'a'), edges)).toBe(true);
  });

  it('rejects a self-loop', () => {
    expect(createsCycle(edge('a', 'a'), edges)).toBe(true);
  });

  it('allows an edge to a descendant or an unrelated node', () => {
    expect(createsCycle(edge('a', 'c'), edges)).toBe(false);
    expect(createsCycle(edge('c', 'd'), edges)).toBe(false);
  });
});
