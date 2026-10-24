import React, { useState } from 'react';
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogCancel,
  AlertDialogAction,
} from '@/components/ui/alert-dialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { toast } from '@/hooks/use-toast';
import { useFolders } from '@/contexts/FolderContext';
import { AlertTriangle, Info } from 'lucide-react';
import type { Folder, DeleteFolderOptions } from '@/types/folder-types';

interface DeleteFolderDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  folder: Folder | null;
  onFolderDeleted?: () => void;
}

export const DeleteFolderDialog: React.FC<DeleteFolderDialogProps> = ({
  open,
  onOpenChange,
  folder,
  onFolderDeleted,
}) => {
  const { deleteFolder, refreshFolders } = useFolders();
  const [deleteOption, setDeleteOption] = useState<'move_to_root' | 'delete_projects'>('move_to_root');
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async () => {
    if (!folder) return;

    const hasContents = (folder.project_count || 0) > 0 || (folder.subfolder_count || 0) > 0;

    try {
      setDeleting(true);

      const options: DeleteFolderOptions = hasContents
        ? {
            move_to_root: deleteOption === 'move_to_root',
            delete_projects: deleteOption === 'delete_projects',
          }
        : {};

      await deleteFolder(folder.id, options);

      toast({
        title: 'Folder deleted',
        description: `"${folder.name}" has been deleted successfully.`,
      });

      await refreshFolders();
      onFolderDeleted?.();
      onOpenChange(false);
    } catch (error: any) {
      console.error('Failed to delete folder:', error);
      toast({
        title: 'Delete failed',
        description: error.message || 'Failed to delete folder. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setDeleting(false);
    }
  };

  const handleCancel = () => {
    if (!deleting) {
      setDeleteOption('move_to_root');
      onOpenChange(false);
    }
  };

  if (!folder) return null;

  const hasProjects = (folder.project_count || 0) > 0;
  const hasSubfolders = (folder.subfolder_count || 0) > 0;
  const hasContents = hasProjects || hasSubfolders;

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent className="sm:max-w-[500px]">
        <AlertDialogHeader>
          <AlertDialogTitle>Delete Folder</AlertDialogTitle>
          <AlertDialogDescription>
            {hasContents ? (
              <>This folder contains {hasProjects && `${folder.project_count} project${folder.project_count !== 1 ? 's' : ''}`}
                {hasProjects && hasSubfolders && ' and '}
                {hasSubfolders && `${folder.subfolder_count} subfolder${folder.subfolder_count !== 1 ? 's' : ''}`}. Choose how to handle them:</>
            ) : (
              <>Are you sure you want to delete "{folder.name}"? This action cannot be undone.</>
            )}
          </AlertDialogDescription>
        </AlertDialogHeader>

        {hasContents && (
          <div className="space-y-4 py-4">
            <RadioGroup value={deleteOption} onValueChange={(val: string) => setDeleteOption(val as 'move_to_root' | 'delete_projects')}>
              <div className="flex items-start space-x-2 p-3 border rounded-md hover:bg-gray-50">
                <RadioGroupItem value="move_to_root" id="move-to-root" className="mt-1" />
                <div className="flex-1">
                  <Label htmlFor="move-to-root" className="font-medium cursor-pointer">
                    Move contents to root
                  </Label>
                  <p className="text-sm text-gray-600 mt-1">
                    Projects and subfolders will be moved to the root level before deleting this folder.
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-2 p-3 border rounded-md border-red-200 hover:bg-red-50">
                <RadioGroupItem value="delete_projects" id="delete-projects" className="mt-1" />
                <div className="flex-1">
                  <Label htmlFor="delete-projects" className="font-medium cursor-pointer text-red-700">
                    Delete all contents
                  </Label>
                  <p className="text-sm text-red-600 mt-1">
                    Permanently delete all projects and subfolders. This cannot be undone.
                  </p>
                </div>
              </div>
            </RadioGroup>

            {deleteOption === 'delete_projects' && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>Warning:</strong> This will permanently delete {hasProjects && `${folder.project_count} project${folder.project_count !== 1 ? 's' : ''}`}
                  {hasProjects && hasSubfolders && ' and '}
                  {hasSubfolders && `${folder.subfolder_count} subfolder${folder.subfolder_count !== 1 ? 's' : ''}`}. This action cannot be undone.
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}

        {!hasContents && (
          <Alert className="my-4">
            <Info className="h-4 w-4" />
            <AlertDescription>This folder is empty and can be safely deleted.</AlertDescription>
          </Alert>
        )}

        <AlertDialogFooter>
          <AlertDialogCancel onClick={handleCancel} disabled={deleting}>
            Cancel
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={handleDelete}
            disabled={deleting}
            className={deleteOption === 'delete_projects' ? 'bg-red-600 hover:bg-red-700' : ''}
          >
            {deleting ? 'Deleting...' : deleteOption === 'delete_projects' ? 'Delete All' : 'Delete Folder'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};
