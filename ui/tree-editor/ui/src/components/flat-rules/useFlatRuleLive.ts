import { useEffect } from 'react';
import { toast } from '@/hooks/use-toast';
import { LiveSync, liveUrl } from '@/lib/live-sync';
import type { Json, Op } from '@/lib/live-ops';
import { buildDocument, useFlatRuleStore } from './useFlatRuleStore';
import type { FlatRuleProjectDetail } from './types';

/** Sync the flat-rule store live with the server while `project` (as loaded, before the editor's defaults) is open. */
export function useFlatRuleLive(project: FlatRuleProjectDetail | null) {
  const enabled = !!project?.live_collab;
  const projectId = project?.id;

  useEffect(() => {
    if (!enabled || !project) return;
    const store = useFlatRuleStore;
    // The last name each rule had here, so a discarded edit names its rule even after the rule is gone.
    const names = new Map<string, string>();
    const remember = (doc: any) => doc?.rules?.forEach((r: any) => r?.rule?.id && names.set(r.rule.id, r.meta?.name));
    let showing = false;

    const markSavedIfIdle = () => {
      if (!sync.idle()) return;
      const { trees, project } = store.getState();
      store.setState({ savedTrees: trees, savedProjectData: project.project_data });
    };

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
        const rules = ops.map((op) => {
          const [, list, key] = op.path.split('/');
          return list === 'rules' && key ? names.get(key.replace(/~1/g, '/').replace(/~0/g, '~')) ?? 'a rule' : `/${list}`;
        });
        toast({
          title: "Edit Discarded",
          description: `Your edit to ${[...new Set(rules)].join(', ')} was discarded: someone else changed or deleted it.`,
          variant: "destructive",
          duration: 10000,
        });
      },
      rejected: (reason) => toast({ title: "Edit Refused", description: reason, variant: "destructive", duration: 10000 }),
      validation: (errors) => store.getState().setValidationErrors(errors),
      settled: markSavedIfIdle,
    });

    store.setState({ live: 'connecting', liveSettle: () => sync.settle() });
    const unsubscribe = store.subscribe(
      (s) => [s.trees, s.project.project_data] as const,
      () => { if (!showing) sync.edit(); },
      { equalityFn: (a, b) => a[0] === b[0] && a[1] === b[1] },
    );
    return () => {
      unsubscribe();
      sync.stop();
      store.setState({ live: null, liveSettle: null });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, projectId]);
}
