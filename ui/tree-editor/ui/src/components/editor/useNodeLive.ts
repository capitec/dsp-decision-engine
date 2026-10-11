import { useEffect, useState } from 'react';
import { toast } from '@/hooks/use-toast';
import { LiveSync, liveUrl } from '@/lib/live-sync';
import { liveOpsLoaded, type Json, type Op } from '@/lib/live-ops';
import type { OfflineConflict } from '../flat-rules/useFlatRuleLive';
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
  const enabled = !!loaded?.live_collab && liveOpsLoaded();
  const [conflict, setConflict] = useState<OfflineConflict | null>(null);

  useEffect(() => {
    if (!enabled || !loaded) return;
    const store = useEditorStore;
    // The last name each node had here, so a discarded edit names its node even after the node is gone.
    const names = new Map<string, string>();
    const remember = (doc: any) => doc?.nodes?.forEach((n: any) => {
      try { names.set(n.id, generateDefaultNodeName(n)); } catch { /* a half-edited node has no name yet */ }
    });
    remember(loaded.project_data);
    let showing = false;
    const label = (path: string) => {
      const [, list, key] = path.split('/');
      const id = key?.replace(/~1/g, '/').replace(/~0/g, '~');
      if (list === 'nodes' && id && id !== '#order') return `the node ${names.get(id) ?? id}`;
      if (list === 'edges') return 'a connection';
      if (list === 'subtrees') return 'the subtrees';
      return list || 'the whole document';
    };

    const markSavedIfIdle = () => {
      if (sync.idle()) store.setState({ isDirty: false });
    };

    const sync: LiveSync = new LiveSync(liveUrl(loaded.id), loaded.project_data as unknown as Json, loaded.version, {
      read: () => {
        const doc = toDocument(store.getState());
        remember(doc);
        return doc as Json;
      },
      show: (doc) => {
        showing = true;
        try {
          store.getState().applyDocument(doc, sync.version);
        } finally {
          showing = false;
        }
        remember(doc);
        markSavedIfIdle();
      },
      status: (live) => store.setState({ live }),
      dropped: (ops: Op[]) => toast({
        title: "Edit Discarded",
        description: `Your edit to ${[...new Set(ops.map((op) => label(op.path)))].join(', ')} was discarded: someone else changed or deleted it.`,
        variant: "destructive",
        duration: 10000,
      }),
      rejected: (reason) => toast({ title: "Edit Refused", description: reason, variant: "destructive", duration: 10000 }),
      merged: () => toast({ title: "Offline Edits Merged", description: "Your offline edits were merged." }),
      offlineRejected: ({ reason, conflicts, doc }) => {
        remember(doc);
        setConflict({ names: [...new Set(conflicts.map(label))], reason, doc });
      },
      validation: (errors) => store.getState().setValidationErrors(errors as any),
      settled: markSavedIfIdle,
    });

    store.setState({ live: 'connecting', liveSettle: () => sync.settle() });
    const unsubscribe = store.subscribe(
      (s) => [s.nodes, s.edges, s.subtrees, s.input_schema, s.output, s.parameters] as const,
      // Measuring and selecting nodes change them too, but only an edit sends: the editor's own form of a stored
      // document (one subtree per root, edge ids) would otherwise be pushed by merely opening it.
      () => { if (!showing && store.getState().isDirty) sync.edit(); },
      { equalityFn: (a, b) => a.every((part, i) => part === b[i]) },
    );
    return () => {
      setConflict(null);
      unsubscribe();
      sync.stop();
      store.setState({ live: null, liveSettle: null });
    };
  }, [enabled, loaded]);

  return { conflict, dismiss: () => setConflict(null) };
}
