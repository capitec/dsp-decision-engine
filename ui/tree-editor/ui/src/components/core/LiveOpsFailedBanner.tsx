import React, { useState } from 'react';
import { Button } from "@/components/ui/button";

/** Shown when the live-editing wasm did not load: the editors still work, saving with Save. */
export const LiveOpsFailedBanner: React.FC<{ reason: string }> = ({ reason }) => {
  const [open, setOpen] = useState(true);
  if (!open) return null;
  // Fixed at the bottom: the node editor's header is fixed at the top and would cover it.
  return (
    <div role="alert" data-testid="live-ops-failed" className="fixed bottom-0 left-0 right-0 z-[60] bg-amber-50 border-t border-amber-300 px-6 py-3">
      <div className="max-w-6xl mx-auto flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-amber-900">
          Live editing is unavailable in this browser: part of the editor failed to load ({reason}). You can keep
          editing; Save stores your changes, and other people's changes show after a reload.
        </p>
        <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>Dismiss</Button>
      </div>
    </div>
  );
};
