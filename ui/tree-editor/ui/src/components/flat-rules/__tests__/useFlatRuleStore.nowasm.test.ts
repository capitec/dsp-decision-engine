import { describe, it, expect, vi } from 'vitest';

// The browser could not load the merge-core wasm: every live op throws.
vi.mock('@/lib/live-ops', () => {
  const unloaded = () => { throw new Error('merge-core wasm is not loaded'); };
  return { diff: unloaded, apply: unloaded, loadLiveOps: unloaded, liveOpsLoaded: () => false };
});
vi.mock('../api', () => ({ flatRuleApi: { updateProject: vi.fn() } }));
vi.mock('@/hooks/use-toast', () => ({ toast: vi.fn() }));
vi.mock('@/lib/background-token-refresh', () => ({ subscribeToStoreActivity: vi.fn() }));

import { useFlatRuleStore } from '../useFlatRuleStore';
import { flatRuleApi } from '../api';

const state = () => useFlatRuleStore.getState();
const rule = (id: string, name: string, threshold = 1) => ({
  meta: { name },
  rule: { id, type: 'unary', condition: { op: '>', feature: 'x', threshold }, then: { id: `${id}1`, type: 'leaf', result_idx: 0 } },
});
const data = (rules: any[]) => ({
  type: 'prioritized_flat_rule', input_schema: [['x', 'Float64']], parameters: {}, mode: 'first_match',
  output: { dtypes: [['Action', 'String']], type_defs: {}, data: [{ Action: 'A' }], default: { Action: 'D' } },
  rules,
});

describe('the flat-rule store without the wasm', () => {
  it('opens, shows a server document keeping unchanged rules, and saves over REST', async () => {
    state().initializeFromProject({ id: 'p', name: 'P', version: 1, project_data: data([rule('a', 'A'), rule('b', 'B')]) } as any);
    const [a, b] = state().trees;

    state().applyDocument(data([rule('a', 'A'), rule('b', 'B', 2)]), 2);

    expect(state().trees[0]).toBe(a);
    expect(state().trees[1]).not.toBe(b);
    expect(state().trees[1].id).toBe(b.id);
    expect(state().project.version).toBe(2);

    const update = flatRuleApi.updateProject as unknown as ReturnType<typeof vi.fn>;
    update.mockResolvedValue({ version: 3, validation_errors: [] });
    expect(await state().saveProject()).toBe(true);
    expect(update.mock.calls[0][1].base_version).toBe(2);
    expect(state().project.version).toBe(3);
  });

  it('a server document that changes nothing outside the rules keeps the project data object', () => {
    state().initializeFromProject({ id: 'p', name: 'P', version: 1, project_data: data([rule('a', 'A')]) } as any);
    const before = state().project.project_data;

    state().applyDocument(data([rule('a', 'Renamed')]), 2);

    expect(state().project.project_data).toBe(before);
    expect(state().trees[0].name).toBe('Renamed');
  });
});
