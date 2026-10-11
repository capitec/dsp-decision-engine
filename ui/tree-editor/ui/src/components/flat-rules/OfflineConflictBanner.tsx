import React from 'react';
import { Button } from "@/components/ui/button";
import { downloadJson } from './api';
import type { OfflineConflict } from './useFlatRuleLive';

/** Persistent notice that offline edits were refused, with the user's own version to download. */
export const OfflineConflictBanner: React.FC<{
  project: { id: string, name: string },
  projectType?: 'flat_rule' | 'tree',
  conflict: OfflineConflict,
  onDismiss: () => void,
}> = ({ project, projectType = 'flat_rule', conflict, onDismiss }) => {
  const download = () => {
    // The shape Export gives, so the file imports back as a project.
    const doc = conflict.doc as Record<string, any>;
    downloadJson({
      ...doc,
      metadata: {
        name: project.name,
        ...(doc.metadata ?? {}),
        project_id: project.id,
        project_type: projectType,
        exported_at: new Date().toISOString(),
      },
    }, `${project.name.replace(/[^a-zA-Z0-9]/g, '_')}_my_offline_version.json`);
  };

  return (
    <div role="alert" data-testid="offline-conflict" className="bg-red-50 border-b border-red-200 px-6 py-3">
      <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-red-800">
          Your offline edits were not saved:{' '}
          {conflict.names.length
            ? `someone else changed ${conflict.names.join(', ')} while you were offline.`
            : conflict.reason}{' '}
          The editor shows the saved version; download yours to keep your edits.
        </p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={download}>Download my version</Button>
          <Button variant="ghost" size="sm" onClick={onDismiss}>Dismiss</Button>
        </div>
      </div>
    </div>
  );
};
