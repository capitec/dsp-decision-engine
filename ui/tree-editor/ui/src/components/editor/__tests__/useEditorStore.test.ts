import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@/api', () => ({ getProjectApiV1ProjectsProjectIdGet: vi.fn(), updateProjectApiV1ProjectsProjectIdPut: vi.fn() }));
vi.mock('@/hooks/use-toast', () => ({ toast: vi.fn() }));
vi.mock('@/lib/background-token-refresh', () => ({ subscribeToStoreActivity: vi.fn() }));

import { useEditorStore } from '../useEditorStore';
import { toDocument } from '../document';
import { updateProjectApiV1ProjectsProjectIdPut } from '@/api';
import { toast } from '@/hooks/use-toast';

const put = updateProjectApiV1ProjectsProjectIdPut as unknown as ReturnType<typeof vi.fn>;
const state = () => useEditorStore.getState();

const unary = (id: string, threshold = 10, x = 0) => ({
  id, position: { x, y: 0 }, data: { type: 'unary', condition: { op: '<=', feature: 'amount', threshold } },
});
const leaf = (id: string, result_idx = 0) => ({ id, position: { x: 0, y: 200 }, data: { type: 'leaf', result_idx } });
const doc = (over: any = {}) => ({
  type: 'ui-tree', formatVersion: 3,
  nodes: [unary('root'), leaf('yes', 0), leaf('no', 1)],
  edges: [
    { id: 'root:yes', source: 'root', target: 'yes', data: { sourceIndex: [0] } },
    { id: 'root:no', source: 'root', target: 'no', data: { sourceIndex: [1] } },
  ],
  subtrees: [{ id: 'root', name: 'Main' }],
  input_schema: [['amount', 'Float64']],
  parameters: {},
  output: { dtypes: [['decision', 'String']], type_defs: {}, data: [{ decision: 'a' }, { decision: 'b' }], default: { decision: 'd' } },
  metadata: { name: 'kept' },
  ...over,
});
const open = (data: any = doc(), over: any = {}) => {
  state().reset();
  useEditorStore.setState({ projectId: 'p' });
  state().initializeFromProject({ id: 'p', name: 'P', description: null, version: 4, project_data: data, validation_errors: [], ...over } as any);
};
const held = () => toDocument(state());

beforeEach(() => {
  put.mockReset();
  (toast as any).mockReset();
  open();
});

describe('the node editor document', () => {
  it('is what was loaded: edges get source:target ids and sourceIndex lists, unknown keys are kept', () => {
    open(doc({ edges: [{ source: 'root', target: 'yes', data: { sourceIndex: 0 } }] }));

    expect(held().edges).toEqual([{ id: 'root:yes', source: 'root', target: 'yes', data: { sourceIndex: [0] } }]);
    expect(held().metadata).toEqual({ name: 'kept' });
    expect(held().nodes).toEqual(doc().nodes);
  });

  it('holds no canvas state: selection and measured sizes are not part of it', () => {
    state().setNodes((nodes) => nodes.map((n) => ({ ...n, selected: true, measured: { width: 10, height: 10 } })), false);

    expect(held().nodes).toEqual(doc().nodes);
  });
});

describe('showing a document from the server', () => {
  it('keeps unchanged nodes and edges as they are and a changed node its selection and size', () => {
    state().setNodes((nodes) => nodes.map((n) => (n.id === 'root' ? { ...n, selected: true, measured: { width: 9, height: 9 } } : n)), false);
    const [root, yes] = state().nodes;
    const edges = state().edges;

    state().applyDocument(doc({ nodes: [unary('root', 99), leaf('yes', 0), leaf('no', 1)] }), 5);

    const [root2, yes2] = state().nodes;
    expect(yes2).toBe(yes);
    expect(state().edges).toEqual(edges);
    expect(state().edges[0]).toBe(edges[0]);
    expect(root2).not.toBe(root);
    expect(root2).toMatchObject({ selected: true, measured: { width: 9, height: 9 } });
    expect((root2.data as any).condition.threshold).toBe(99);
    expect(state().version).toBe(5);
    expect(state().isDirty).toBe(false);
  });

  it('adds new nodes and edges, drops deleted ones and keeps the subtree names', () => {
    state().applyDocument(doc({
      nodes: [unary('root'), leaf('yes', 0), leaf('extra', 1)],
      edges: [{ id: 'root:yes', source: 'root', target: 'yes', data: { sourceIndex: [0] } },
              { id: 'root:extra', source: 'root', target: 'extra', data: { sourceIndex: [1] } }],
    }), 5);

    expect(state().nodes.map((n) => n.id)).toEqual(['root', 'yes', 'extra']);
    expect(state().edges.map((e) => e.id)).toEqual(['root:yes', 'root:extra']);
    expect(state().subtrees.map((s) => [s.id, s.name])).toEqual([['root', 'Main']]);
  });

  it('takes the rest of the document: output, parameters, input schema and unknown keys', () => {
    state().applyDocument(doc({ parameters: { lo: { type: 'Float64', default_value: 1 } }, metadata: { name: 'new' } }), 5);

    expect(state().parameters).toEqual({ lo: { type: 'Float64', default_value: 1 } });
    expect(held().metadata).toEqual({ name: 'new' });
  });

  it('places the server validation errors on their nodes', () => {
    state().setValidationErrors([{ loc: [], msg: "tree node 'yes': result_idx 5 is not a row", type: 'build_error' }]);

    expect([...state().invalidNodeIds]).toEqual(['yes']);
    expect(state().validationErrors).toHaveLength(1);
  });
});

describe('saving', () => {
  it('without live editing PUTs the document with the loaded version and takes the new one', async () => {
    put.mockResolvedValue({ data: { version: 5, validation_errors: [] } });
    state().updateNodeData('root', { condition: { op: '<=', feature: 'amount', threshold: 3 } });

    await state().saveProject();

    expect(put).toHaveBeenCalledTimes(1);
    expect(put.mock.calls[0][0].body.base_version).toBe(4);
    expect(put.mock.calls[0][0].body.project_data.nodes[0].data.condition.threshold).toBe(3);
    expect(state().version).toBe(5);
    expect(state().isDirty).toBe(false);
  });

  it('a stale save is refused with a toast and the edits stay unsaved', async () => {
    put.mockRejectedValue({ detail: { code: 'stale_version', current_version: 9 } });
    state().updateNodeData('root', { condition: { op: '<=', feature: 'amount', threshold: 3 } });

    await state().saveProject();

    expect(toast).toHaveBeenCalledWith(expect.objectContaining({ title: 'Save Refused' }));
    expect(state().isDirty).toBe(true);
    expect(state().version).toBe(4);
  });

  it('with live editing waits for the server instead of PUTting, then reports the project saved', async () => {
    const settle = vi.fn().mockResolvedValue('saved');
    useEditorStore.setState({ live: 'live', liveSettle: settle });
    state().updateNodeData('root', { condition: { op: '<=', feature: 'amount', threshold: 3 } });

    await state().saveProject();

    expect(settle).toHaveBeenCalled();
    expect(put).not.toHaveBeenCalled();
    expect(toast).toHaveBeenCalledWith(expect.objectContaining({ title: 'Project Saved' }));
    expect(state().isDirty).toBe(false);
    expect(state().saving).toBe(false);
  });

  it('with live editing, a server that does not confirm in time leaves the edits unsaved and says so', async () => {
    useEditorStore.setState({ live: 'reconnecting', liveSettle: vi.fn().mockResolvedValue('unsaved') });
    state().updateNodeData('root', { condition: { op: '<=', feature: 'amount', threshold: 3 } });

    await state().saveProject();

    expect(toast).toHaveBeenCalledWith(expect.objectContaining({ title: 'Save Failed' }));
    expect(state().isDirty).toBe(true);
  });

  it('with live editing, edits the server stored but has not validated yet are saved and the toast says so', async () => {
    useEditorStore.setState({ live: 'live', liveSettle: vi.fn().mockResolvedValue('unvalidated') });
    state().updateNodeData('root', { condition: { op: '<=', feature: 'amount', threshold: 3 } });

    await state().saveProject();

    expect(toast).toHaveBeenCalledWith(expect.objectContaining({ title: 'Saved, not validated yet' }));
    expect(state().isDirty).toBe(false);
  });
});

describe('auto-arrange while another session edits', () => {
  it('turns off when a server version moves a node, so this browser does not lay their move out again', () => {
    useEditorStore.setState({ autoRearrange: true });

    state().applyDocument(doc({ nodes: [unary('root', 99), leaf('yes', 0), leaf('no', 1)] }), 5);
    expect(state().autoRearrange).toBe(true);

    state().applyDocument(doc({ nodes: [unary('root', 99), leaf('yes', 0), { ...leaf('no', 1), position: { x: 400, y: 300 } }] }), 6);
    expect(state().autoRearrange).toBe(false);
  });
});
