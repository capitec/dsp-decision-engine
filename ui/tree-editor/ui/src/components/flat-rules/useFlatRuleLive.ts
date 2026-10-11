import { useEffect, useState } from 'react';
import { toast } from '@/hooks/use-toast';
import { LiveSync, liveUrl } from '@/lib/live-sync';
import type { Json, Op } from '@/lib/live-ops';
import { buildDocument, useFlatRuleStore } from './useFlatRuleStore';
import type { FlatRuleProjectDetail } from './types';

/** Offline edits the server refused: what they collided with, and my version of the document with them. */
export interface OfflineConflict {
  names: string[];
  reason: string;
  doc: Json;
}

/**
 * Sync the flat-rule store live with the server while `project` (as loaded, before the editor's defaults) is open.
 * Returns the last refused offline batch (null when none) and a function that dismisses it.
 */
export function useFlatRuleLive(project: FlatRuleProjectDetail | null) {
  const enabled = !!project?.live_collab;
  const projectId = project?.id;
  const [conflict, setConflict] = useState<OfflineConflict | null>(null);

  useEffect(() => {
    if (!enabled || !project) return;
    const store = useFlatRuleStore;
    // The last name each rule had here, so a discarded edit names its rule even after the rule is gone.
    const names = new Map<string, string>();
    const remember = (doc: any) => doc?.rules?.forEach((r: any) => r?.rule?.id && names.set(r.rule.id, r.meta?.name));
    let showing = false;
    // What a path names for a person: the rule's name, or the part of the document.
    const label = (path: string) => {
      const [, list, key] = path.split('/');
      if (list !== 'rules') return list || 'the whole document';
      if (key === '#order') return 'the rule order';
      return key ? names.get(key.replace(/~1/g, '/').replace(/~0/g, '~')) ?? 'a rule' : 'the rules';
    };

    const markSavedIfIdle = () => {
      if (!sync.idle()) return;
      const { trees, project } = store.getState();
      store.setState({ savedTrees: trees, savedProjectData: project.project_data });
    };

    let ended: ReturnType<typeof toast> | undefined;
    const sync: LiveSync = new LiveSync(liveUrl(project.id), project.project_data as unknown as Json, project.version, {
      read: () => {
        const doc = buildDocument(store.getState());
        remember(doc);
        return doc as unknown as Json;
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
      dropped: (ops: Op[]) => {
        const rules = ops.map((op) => label(op.path));
        toast({
          title: "Edit Discarded",
          description: `Your edit to ${[...new Set(rules)].join(', ')} was discarded: someone else changed or deleted it.`,
          variant: "destructive",
          duration: 10000,
        });
      },
      rejected: (reason) => toast({ title: "Edit Refused", description: reason, variant: "destructive", duration: 10000 }),
      merged: () => toast({ title: "Offline Edits Merged", description: "Your offline edits were merged." }),
      offlineRejected: ({ reason, conflicts, doc }) => {
        remember(doc);
        setConflict({ names: [...new Set(conflicts.map(label))], reason, doc });
      },
      validation: (errors) => store.getState().setValidationErrors(errors),
      settled: markSavedIfIdle,
      ended: (reason) => {
        ended = toast({
          title: "Live Editing Stopped",
          description: `${reason[0].toUpperCase()}${reason.slice(1)}. Edits not saved yet stay in this editor until you leave it; Export keeps a copy.`,
          variant: "destructive",
          duration: Infinity,
        });
      },
    });
    // Signing in again through the session dialog picks live editing up where it stopped.
    const resume = () => {
      ended?.dismiss();
      sync.resume();
    };
    window.addEventListener('session-refreshed', resume);

    store.setState({ live: 'connecting', liveSettle: () => sync.settle() });
    const unsubscribe = store.subscribe(
      (s) => [s.trees, s.project.project_data] as const,
      () => { if (!showing) sync.edit(); },
      { equalityFn: (a, b) => a[0] === b[0] && a[1] === b[1] },
    );
    return () => {
      window.removeEventListener('session-refreshed', resume);
      setConflict(null);
      unsubscribe();
      sync.stop();
      store.setState({ live: null, liveSettle: null });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, projectId]);

  return { conflict, dismiss: () => setConflict(null) };
}
