import React from 'react';
import { Badge } from "@/components/ui/badge";
import type { LiveStatus } from '@/lib/live-sync';

const LOOK: Record<LiveStatus, [string, string]> = {
  live: ['Live', 'bg-green-100 text-green-800 border-green-200'],
  connecting: ['Connecting...', 'bg-amber-100 text-amber-800 border-amber-200'],
  reconnecting: ['Offline - edits will be merged when you reconnect', 'bg-amber-100 text-amber-800 border-amber-200'],
  ended: ['Not live - edits are not saved', 'bg-red-100 text-red-800 border-red-200'],
};

/** The editor's live editing connection, shown beside the project name (nothing when it does not edit live). */
export const LiveStatusBadge: React.FC<{ live: LiveStatus | null }> = ({ live }) => {
  if (!live) return null;
  const [text, colours] = LOOK[live];
  return <Badge variant="secondary" data-testid="live-status" className={`text-xs ${colours}`}>{text}</Badge>;
};
