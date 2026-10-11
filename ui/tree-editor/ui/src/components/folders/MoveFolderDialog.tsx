import React, { useState, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { toast } from '@/hooks/use-toast';
import { useFolders } from '@/contexts/FolderContext';
import { FolderIcon, Home, AlertTriangle, Info } from 'lucide-react';
import type { Folder } from '@/types/folder-types';

interface MoveFolderDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  folder: Folder | null;
  onFolderMoved?: () => void;
}

export const MoveFolderDialog: React.FC<MoveFolderDialogProps> = ({
  open,
  onOpenChange,
  folder,
  onFolderMoved,
}) => {
  const { moveFolder, folders, refreshFolders } = useFolders();
  const [selectedParentId, setSelectedParentId] = useState<string | undefined>();
  const [moving, setMoving] = useState(false);

  // Get all descendant folder IDs to prevent circular references
  const getDescendantIds = (folderId: string): Set<string> => {
    const descendants = new Set<string>([folderId]);
    const addDescendants = (id: string) => {
      folders
        .filter(f => f.parent_id === id)
        .forEach(child => {
          descendants.add(child.id);
          addDescendants(child.id);
        });
    };
    addDescendants(folderId);
    return descendants;
  };

  // Get valid target folders (excluding self and descendants)
  const validTargetFolders = useMemo(() => {
    if (!folder) return [];

    const invalidIds = getDescendantIds(folder.id);
    return folders.filter(f => !invalidIds.has(f.id) && f.id !== folder.id);
  }, [folder, folders]);

  // Build folder tree for display
  const buildFolderOptions = () => {
    const buildTree = (parentId: string | undefined, level: number = 0): React.ReactElement[] => {
      return folders
        .filter(f => f.parent_id === parentId && validTargetFolders.some(vf => vf.id === f.id))
        .sort((a, b) => a.name.localeCompare(b.name))
        .flatMap(f => [
          <div key={f.id} className="flex items-center py-2">
            <RadioGroupItem value={f.id} id={`folder-${f.id}`} />
            <label
              htmlFor={`folder-${f.id}`}
              className="flex items-center gap-2 ml-2 flex-1 cursor-pointer"
              style={{ paddingLeft: `${level * 20}px` }}
            >
              <FolderIcon className="h-4 w-4 text-gray-500" />
              <span className="text-sm">{f.name}</span>
              {f.is_shared && (
                <span className="text-xs text-purple-600" title="Shared folder">
                  (Shared)
                </span>
              )}
            </label>
          </div>,
          ...buildTree(f.id, level + 1),
        ]);
    };

    return buildTree(undefined);
  };

  const handleMove = async () => {
    if (!folder) return;

    // Prevent moving to the same location
    if (selectedParentId === folder.parent_id) {
      toast({
        title: 'No change',
        description: 'Folder is already in this location.',
        variant: 'destructive',
      });
      return;
    }

    try {
      setMoving(true);

      await moveFolder(folder.id, {
        new_parent_id: selectedParentId,
      });

      toast({
        title: 'Folder moved',
        description: `"${folder.name}" has been moved successfully.`,
      });

      await refreshFolders();
      onFolderMoved?.();
      onOpenChange(false);
    } catch (error: any) {
      console.error('Failed to move folder:', error);
      toast({
        title: 'Move failed',
        description: error.message || 'Failed to move folder. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setMoving(false);
    }
  };

  const handleCancel = () => {
    if (!moving) {
      setSelectedParentId(undefined);
      onOpenChange(false);
    }
  };

  if (!folder) return null;

  const targetFolder = selectedParentId ? folders.find(f => f.id === selectedParentId) : null;
  const hasWriteAccess = targetFolder ? targetFolder.access_mode === 'write' || targetFolder.access_mode === 'admin' : true;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>Move Folder</DialogTitle>
          <DialogDescription>
            Choose a new location for "{folder.name}"
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto space-y-4 py-4">
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              Moving a folder will also move all its subfolders and projects. Permission inheritance will update accordingly.
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <Label>Select destination folder</Label>
            <RadioGroup value={selectedParentId || 'root'} onValueChange={(val: string) => setSelectedParentId(val === 'root' ? undefined : val)}>
              {/* Root option */}
              <div className="flex items-center py-2">
                <RadioGroupItem value="root" id="folder-root" />
                <label htmlFor="folder-root" className="flex items-center gap-2 ml-2 flex-1 cursor-pointer">
                  <Home className="h-4 w-4 text-gray-500" />
                  <span className="text-sm font-medium">Root (No parent)</span>
                </label>
              </div>

              {/* Folder tree */}
              {buildFolderOptions()}
            </RadioGroup>

            {validTargetFolders.length === 0 && (
              <p className="text-sm text-gray-500 py-4">No valid destination folders available.</p>
            )}
          </div>

          {/* Permission warning */}
          {targetFolder && !hasWriteAccess && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                You only have READ access to "{targetFolder.name}". You cannot move folders into it.
              </AlertDescription>
            </Alert>
          )}

          {/* Current location info */}
          {folder.parent_id && (
            <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-md">
              <span className="font-medium">Current location:</span>{' '}
              {folders.find(f => f.id === folder.parent_id)?.name || 'Unknown'}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel} disabled={moving}>
            Cancel
          </Button>
          <Button onClick={handleMove} disabled={moving || (targetFolder && !hasWriteAccess) || selectedParentId === folder.parent_id}>
            {moving ? 'Moving...' : 'Move Folder'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
