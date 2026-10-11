import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

const wasm = vi.hoisted(() => ({ loaded: true }));
vi.mock('@/lib/live-ops', async (original) => ({ ...(await original<object>()), liveOpsLoaded: () => wasm.loaded }));
vi.mock('@/api', () => ({ getProjectApiV1ProjectsProjectIdGet: vi.fn(), updateProjectApiV1ProjectsProjectIdPut: vi.fn() }));
vi.mock('@/hooks/use-toast', () => ({ toast: vi.fn() }));
vi.mock('@/lib/background-token-refresh', () => ({ subscribeToStoreActivity: vi.fn() }));

import { useEditorStore } from '../useEditorStore';
import { useNodeLive } from '../useNodeLive';
import { toast } from '@/hooks/use-toast';

class FakeSocket {
  static all: FakeSocket[] = [];
  sent: any[] = [];
  onopen: any = null;
  onmessage: any = null;
  onclose: any = null;
  constructor(public url: string) { FakeSocket.all.push(this); }
  send(text: string) { this.sent.push(JSON.parse(text)); }
  close() {}
  receive(message: object) { act(() => this.onmessage({ data: JSON.stringify(message) })); }
  ops() { return this.sent.filter((m) => m.type === 'ops'); }
}

const state = () => useEditorStore.getState();
const node = (id: string, threshold = 10) => ({
  id, position: { x: 0, y: 0 }, data: { type: 'unary', condition: { op: '<=', feature: 'amount', threshold } },
});
const leaf = (id: string) => ({ id, position: { x: 0, y: 200 }, data: { type: 'leaf', result_idx: -1 } });
const doc = () => ({
  type: 'ui-tree', formatVersion: 3, nodes: [node('a'), leaf('b')],
  edges: [{ id: 'a:b', source: 'a', target: 'b', data: { sourceIndex: [0] } }], subtrees: [{ id: 'a', name: 'A' }],
  input_schema: [['amount', 'Float64']], parameters: {},
  output: { dtypes: [['d', 'String']], type_defs: {}, data: [], default: { d: 'x' } },
});
const open = (live_collab = true) => {
  state().reset();
  useEditorStore.setState({ projectId: 'p' });
  state().initializeFromProject({ id: 'p', name: 'P', version: 3, project_data: doc(), validation_errors: [], live_collab } as any);
};
const connected = () => {
  const socket = FakeSocket.all[FakeSocket.all.length - 1];
  act(() => socket.onopen());
  socket.receive({ type: 'ready', version: 3 });
  return socket;
};

beforeEach(() => {
  FakeSocket.all = [];
  wasm.loaded = true;
  vi.stubGlobal('WebSocket', FakeSocket);
  vi.useFakeTimers();
  (toast as any).mockReset();
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('the node editor live', () => {
  it('pushes a node edit after the pause, addressed by the node id, and is saved once the server has it', () => {
    open();
    renderHook(() => useNodeLive());
    const socket = connected();
    expect(state().live).toBe('live');

    act(() => state().updateNodeData('a', { condition: { op: '<=', feature: 'amount', threshold: 7 } }));
    expect(state().isDirty).toBe(true);
    act(() => { vi.advanceTimersByTime(150); });

    const [sent] = socket.ops();
    expect(sent.ops).toEqual([{ op: 'replace', path: '/nodes/a/data/condition/threshold', value: 7 }]);
    socket.receive({ type: 'applied', version: 4, client_id: socket.sent[0].client_id, msg_id: sent.id, ops: sent.ops });
    expect(state().isDirty).toBe(false);
    expect(state().version).toBe(4);
  });

  it('shows another session\'s edit without marking the project unsaved and keeps the node\'s canvas state', () => {
    open();
    renderHook(() => useNodeLive());
    const socket = connected();
    act(() => state().setNodes((nodes) => nodes.map((n) => ({ ...n, selected: n.id === 'a' })), false));

    socket.receive({ type: 'applied', version: 4, client_id: 'other', msg_id: 'm',
                     ops: [{ op: 'replace', path: '/nodes/a/data/condition/threshold', value: 42 }] });

    const a: any = state().nodes.find((n) => n.id === 'a');
    expect(a.data.condition.threshold).toBe(42);
    expect(a.selected).toBe(true);
    expect(state().version).toBe(4);
    expect(state().isDirty).toBe(false);
    act(() => { vi.advanceTimersByTime(500); });
    expect(socket.ops()).toEqual([]);
  });

  it('keeps my un-pushed edit to one node when another session changed a different node', () => {
    open();
    renderHook(() => useNodeLive());
    const socket = connected();
    act(() => state().updateNodeData('a', { condition: { op: '<=', feature: 'amount', threshold: 7 } }));

    socket.receive({ type: 'applied', version: 4, client_id: 'other', msg_id: 'm',
                     ops: [{ op: 'replace', path: '/nodes/b/data/result_idx', value: 0 }] });

    const [a, b]: any[] = state().nodes;
    expect(a.data.condition.threshold).toBe(7);
    expect(b.data.result_idx).toBe(0);
    expect(socket.ops()[0].ops).toEqual([{ op: 'replace', path: '/nodes/a/data/condition/threshold', value: 7 }]);
    expect(state().isDirty).toBe(true);
  });

  it('tells me when my edit to a node someone else deleted was discarded', () => {
    open();
    renderHook(() => useNodeLive());
    const socket = connected();
    socket.receive({ type: 'dropped', msg_id: 'x', version: 4, ops: [{ op: 'replace', path: '/nodes/a/data/condition/threshold', value: 1 }] });

    expect(toast).toHaveBeenCalledWith(expect.objectContaining({
      title: 'Edit Discarded', description: expect.stringContaining('amount ≤ 10'),
    }));
  });

  it('shows the server\'s validation errors on the nodes they name', () => {
    open();
    renderHook(() => useNodeLive());
    const socket = connected();
    socket.receive({ type: 'validation', version: 3, errors: [{ loc: [], msg: "tree node 'b': bad", type: 'build_error' }] });

    expect([...state().invalidNodeIds]).toEqual(['b']);
  });

  it('does not connect when live editing is off, or when this browser could not load the wasm', () => {
    open(false);
    renderHook(() => useNodeLive());
    wasm.loaded = false;
    open(true);
    renderHook(() => useNodeLive());

    expect(FakeSocket.all).toEqual([]);
    expect(state().live).toBe(null);
    expect(state().liveSettle).toBe(null);
  });
});
