import { useLiveEditing } from '@/hooks/use-live-editing';
import type { Json } from '@/lib/live-ops';
import { buildDocument, useFlatRuleStore } from './useFlatRuleStore';
import type { FlatRuleProjectDetail } from './types';

/**
 * Sync the flat-rule store live with the server while `project` (as loaded, before the editor's defaults) is open.
 * Returns the last refused offline batch (null when none) and a function that dismisses it.
 */
export function useFlatRuleLive(project: FlatRuleProjectDetail | null) {
  return useLiveEditing(project, () => {
    const store = useFlatRuleStore;
    // The last name each rule had here, so a discarded edit names its rule even after the rule is gone.
    const names = new Map<string, string>();
    return {
      store,
      read: () => buildDocument(store.getState()) as unknown as Json,
      show: (doc, version) => store.getState().applyDocument(doc, version),
      saved: () => {
        const { trees, project } = store.getState();
        store.setState({ savedTrees: trees, savedProjectData: project.project_data });
      },
      validation: (errors) => store.getState().setValidationErrors(errors as any),
      watch: (changed) => store.subscribe(
        (s) => [s.trees, s.project.project_data] as const,
        changed,
        { equalityFn: (a, b) => a[0] === b[0] && a[1] === b[1] },
      ),
      remember: (doc: any) => doc?.rules?.forEach((r: any) => r?.rule?.id && names.set(r.rule.id, r.meta?.name)),
      label: (path) => {
        const [, list, key] = path.split('/');
        if (list !== 'rules') return list || 'the whole document';
        if (key === '#order') return 'the rule order';
        return key ? names.get(key.replace(/~1/g, '/').replace(/~0/g, '~')) ?? 'a rule' : 'the rules';
      },
    };
  });
}
