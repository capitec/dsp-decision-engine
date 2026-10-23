import { useEffect, useState } from 'react';
import { toast } from '@/hooks/use-toast';
import { LiveSync, liveUrl, type LiveStatus } from '@/lib/live-sync';
import { liveOpsLoaded, type Json, type Op } from '@/lib/live-ops';

/** Offline edits the server refused: what they collided with, and my version of the document with them. */
export interface OfflineConflict {
  names: string[];
  reason: string;
  doc: Json;
}

/** What live editing needs from one editor. */
export interface LiveEditor {
  /** The editor's store; live editing sets its `live` status and the `liveSettle` that Save waits on. */
  store: { setState(state: Partial<{ live: LiveStatus | null; liveSettle: (() => Promise<boolean>) | null }>): void };
  /** The document the editor holds now. */
  read: () => Json;
  /** Show the server's document (at `version`) in the editor. */
  show: (doc: Json, version: number) => void;
  /** Every edit is on the server: the editor has nothing unsaved. */
  saved: () => void;
  validation: (errors: unknown[]) => void;
  /** Call `changed` after every change to the document; returns the unsubscribe. */
  watch: (changed: () => void) => () => void;
  /** Note the names in `doc`, so `label` can still name a part after it is gone. */
  remember: (doc: Json) => void;
  /** What a path in the document is for a person, as in "Your edit to the rule Pricing". */
  label: (path: string) => string;
}

/**
 * Edit `project` live with every other session while it is open, when the server edits live and this browser
 * loaded the wasm (else Save goes over REST). `editor` is called once per connection. Returns the last refused
 * offline batch (null when none) and a function that dismisses it.
 *
 * @example
 * const offline = useLiveEditing(project, () => ({ store, read, show, saved, validation, watch, remember, label }));
 * {offline.conflict && <OfflineConflictBanner conflict={offline.conflict} onDismiss={offline.dismiss} ... />}
 */
export function useLiveEditing(
  project: { id: string; version: number; project_data: unknown; live_collab?: boolean } | null,
  editor: () => LiveEditor,
) {
  const enabled = !!project?.live_collab && liveOpsLoaded();
  const projectId = project?.id;
  const [conflict, setConflict] = useState<OfflineConflict | null>(null);

  useEffect(() => {
    if (!enabled || !project) return;
    const e = editor();
    const doc = project.project_data as Json;
    e.remember(doc);
    let showing = false;
    let ended: ReturnType<typeof toast> | undefined;
    const savedIfIdle = () => { if (sync.idle()) e.saved(); };
    const names = (paths: string[]) => [...new Set(paths.map(e.label))];

    const sync: LiveSync = new LiveSync(liveUrl(project.id), doc, project.version, {
      read: () => {
        const now = e.read();
        e.remember(now);
        return now;
      },
      show: (now) => {
        showing = true;
        try {
          e.show(now, sync.version);
        } finally {
          showing = false;
        }
        e.remember(now);
        savedIfIdle();
      },
      status: (live) => e.store.setState({ live }),
      dropped: (ops: Op[]) => toast({
        title: "Edit Discarded",
        description: `Your edit to ${names(ops.map((op) => op.path)).join(', ')} was discarded: someone else changed or deleted it.`,
        variant: "destructive",
        duration: 10000,
      }),
      rejected: (reason) => toast({ title: "Edit Refused", description: reason, variant: "destructive", duration: 10000 }),
      merged: () => toast({ title: "Offline Edits Merged", description: "Your offline edits were merged." }),
      offlineRejected: ({ reason, conflicts, doc }) => {
        e.remember(doc);
        setConflict({ names: names(conflicts), reason, doc });
      },
      validation: (errors) => e.validation(errors),
      settled: savedIfIdle,
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

    e.store.setState({ live: 'connecting', liveSettle: () => sync.settle() });
    const unsubscribe = e.watch(() => { if (!showing) sync.edit(); });
    return () => {
      window.removeEventListener('session-refreshed', resume);
      ended?.dismiss();
      setConflict(null);
      unsubscribe();
      sync.stop();
      e.store.setState({ live: null, liveSettle: null });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, projectId]);

  return { conflict, dismiss: () => setConflict(null) };
}
