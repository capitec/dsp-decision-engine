import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { toast } from '@/hooks/use-toast';
import { useFolders } from '@/contexts/FolderContext';
import type { Folder } from '@/types/folder-types';

interface RenameFolderDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  folder: Folder | null;
  onFolderRenamed?: () => void;
}

export const RenameFolderDialog: React.FC<RenameFolderDialogProps> = ({
  open,
  onOpenChange,
  folder,
  onFolderRenamed,
}) => {
  const { updateFolder, folders, refreshFolders } = useFolders();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [updating, setUpdating] = useState(false);
  const [errors, setErrors] = useState<{ name?: string; description?: string }>({});

  // Initialize form when folder changes
  useEffect(() => {
    if (folder) {
      setName(folder.name);
      setDescription(folder.description || '');
      setErrors({});
    }
  }, [folder]);

  const validateForm = () => {
    const newErrors: { name?: string; description?: string } = {};

    if (!name.trim()) {
      newErrors.name = 'Folder name is required';
    } else if (name.trim().length < 2) {
      newErrors.name = 'Folder name must be at least 2 characters';
    } else if (name.trim().length > 255) {
      newErrors.name = 'Folder name must be less than 255 characters';
    }

    if (description && description.length > 500) {
      newErrors.description = 'Description must be less than 500 characters';
    }

    // Check for duplicate names in the same parent (excluding self)
    if (folder) {
      const siblings = folders.filter(f => f.parent_id === folder.parent_id && f.id !== folder.id);
      const duplicate = siblings.find(f => f.name.toLowerCase() === name.trim().toLowerCase());
      if (duplicate) {
        newErrors.name = 'A folder with this name already exists in this location';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!folder || !validateForm()) {
      return;
    }

    // Check if anything changed
    if (name.trim() === folder.name && (description.trim() || undefined) === folder.description) {
      onOpenChange(false);
      return;
    }

    try {
      setUpdating(true);

      await updateFolder(folder.id, {
        name: name.trim(),
        description: description.trim() || undefined,
      });

      toast({
        title: 'Folder updated',
        description: 'Folder has been updated successfully.',
      });

      await refreshFolders();
      onFolderRenamed?.();
      onOpenChange(false);
    } catch (error: any) {
      console.error('Failed to update folder:', error);
      toast({
        title: 'Update failed',
        description: error.message || 'Failed to update folder. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setUpdating(false);
    }
  };

  const handleCancel = () => {
    if (!updating) {
      if (folder) {
        setName(folder.name);
        setDescription(folder.description || '');
      }
      setErrors({});
      onOpenChange(false);
    }
  };

  if (!folder) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Rename Folder</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="folder-name">Folder Name *</Label>
            <Input
              id="folder-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter folder name"
              disabled={updating}
              autoFocus
              className={errors.name ? 'border-red-500' : ''}
            />
            {errors.name && <p className="text-sm text-red-600">{errors.name}</p>}
          </div>

          <div className="space-y-2">
            <Label htmlFor="folder-description">Description</Label>
            <Textarea
              id="folder-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter folder description (optional)"
              disabled={updating}
              rows={3}
              className={errors.description ? 'border-red-500' : ''}
            />
            {errors.description && <p className="text-sm text-red-600">{errors.description}</p>}
            <p className="text-xs text-gray-500">{description.length}/500 characters</p>
          </div>
        </form>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel} disabled={updating}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={updating || !name.trim()}>
            {updating ? 'Saving...' : 'Save Changes'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
