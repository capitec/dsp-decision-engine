import { toast } from '@/hooks/use-toast';

/** True for the 409 the server answers when the project was saved by someone else since it was loaded. */
export const isStaleSave = (err: any): boolean => err?.detail?.code === 'stale_version';

export const staleSaveToast = () =>
  toast({
    title: "Save Refused",
    description: "Someone else saved this project since you opened it. Reload to see their changes; your edits are not saved.",
    variant: "destructive",
  });
