import { useLiveEditing } from '@/hooks/use-live-editing';
import type { Json } from '@/lib/live-ops';
import { useEditorStore } from './useEditorStore';
import { toDocument } from './document';
import { generateDefaultNodeName } from './util';

/**
 * Sync the node editor's store live with the server while the loaded project is open (when the server edits live
 * and this browser loaded the wasm). Returns the last refused offline batch (null when none) and a function that
 * dismisses it.
 */
export function useNodeLive() {
  const loaded = useEditorStore((state) => state.loaded);
  return useLiveEditing(loaded, () => {
    const store = useEditorStore;
    // The last name each node had here, so a discarded edit names its node even after the node is gone.
    const names = new Map<string, string>();
    return {
      store,
      read: () => toDocument(store.getState()) as Json,
      show: (doc, version) => store.getState().applyDocument(doc, version),
      saved: () => store.setState({ isDirty: false }),
      validation: (errors) => store.getState().setValidationErrors(errors as any),
      watch: (changed) => store.subscribe(
        (s) => [s.nodes, s.edges, s.subtrees, s.input_schema, s.output, s.parameters] as const,
        // Measuring and selecting nodes change them too, but only an edit sends: the editor's own form of a stored
        // document (one subtree per root, edge ids) would otherwise be pushed by merely opening it.
        () => { if (store.getState().isDirty) changed(); },
        { equalityFn: (a, b) => a.every((part, i) => part === b[i]) },
      ),
      remember: (doc: any) => doc?.nodes?.forEach((n: any) => {
        try { names.set(n.id, generateDefaultNodeName(n)); } catch { /* a half-edited node has no name yet */ }
      }),
      label: (path) => {
        const [, list, key] = path.split('/');
        const id = key?.replace(/~1/g, '/').replace(/~0/g, '~');
        if (list === 'nodes' && id && id !== '#order') return `the node ${names.get(id) ?? id}`;
        if (list === 'edges') return 'a connection';
        if (list === 'subtrees') return 'the subtrees';
        return list || 'the whole document';
      },
    };
  });
}
